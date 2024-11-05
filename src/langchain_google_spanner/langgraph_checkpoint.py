# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Cloud Spanner-based langgraph checkpointer"""
import threading
from contextlib import closing, contextmanager
from typing import Any, Dict, Iterator, Optional, Sequence, Tuple, TypedDict

from google.cloud.spanner_dbapi import Cursor  # type: ignore[import-untyped]
from langchain_core.load.dump import dumps
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
)
from langgraph.checkpoint.serde.base import SerializerProtocol

MetadataInput = Optional[dict[str, Any]]

from .version import __version__

USER_AGENT_CHECKPOINTER = f"langchain-google-spanner-python:checkpointer/{__version__}"
OPERATION_TIMEOUT_SECONDS = 240


def _config(thread_id, checkpoint_ns, checkpoint_id) -> RunnableConfig:
    return RunnableConfig(
        configurable={
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
            "checkpoint_id": checkpoint_id,
        }
    )


class _SpannerTableNames(TypedDict):
    checkpoints: str
    checkpoint_writes: str


class SpannerCheckpointSaver(BaseCheckpointSaver[str]):
    """A checkpoint saver that stores LangGraph checkpoints in a Spanner database.

    Checkpointers allow LangGraph agents to persist their state
    within and across multiple interactions.

    Args:
        instance_id (str):
            Required. The Spanner instance to use for saving checkpoints.
        database_id (str):
            Required. The Spanner database to use for saving checkpoints.
        project_id (str):
            Optional. The GCP project which owns the instances, tables and data.
            If not provided, will attempt to determine from the environment.
        table_names (dict[str, str]):
            Optional. The Spanner table names to use for saving checkpoints, for example,
            {"checkpoints": "chkpts", "checkpoint_writes": "chkpt_writes"} will
            correspond to the table names "chkpts" and "chkpt_writes".
        user_agent (str):
            Optional. User agent to be used with this checkpointer's requests.
        autocommit (bool):
            Optional. A boolean indicating whether or not the connection is in autocommit mode.
            Defaults to True.
        connect_kwargs (dict[str, Any]):
            Optional. Additional kwargs to provide to the DB API connection. Defaults to an
            empty dictionary. See google.cloud.spanner_dbapi.connection.connect for details.

    Examples:

        >>> from langchain_google_spanner import SpannerCheckpointSaver
        >>> from langgraph.graph import StateGraph
        >>>
        >>> builder = StateGraph(int)
        >>> builder.add_node("add_one", lambda x: x + 1)
        >>> builder.set_entry_point("add_one")
        >>> builder.set_finish_point("add_one")
        >>> checkpointer = SpannerCheckpointSaver(instance_id, database_id, project_id)
        >>> graph = builder.compile(checkpointer=checkpointer)
        >>> config = {"configurable": {"thread_id": "1"}}
        >>> graph.get_state(config)
        >>> result = graph.invoke(3, config)
        >>> graph.get_state(config)
        StateSnapshot(values=4, next=(), config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '0c62ca34-ac19-445d-bbb0-5b4984975b2a'}}, parent_config=None)
    """  # noqa

    lock: threading.Lock

    def __init__(
        self,
        instance_id: str,
        database_id: str,
        project_id: Optional[str] = None,
        table_names: _SpannerTableNames = _SpannerTableNames(
            checkpoints="checkpoints", checkpoint_writes="checkpoint_writes"
        ),
        user_agent: str = USER_AGENT_CHECKPOINTER,
        autocommit: bool = True,
        connect_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        from google.cloud.spanner_dbapi.connection import (  # type: ignore[import-untyped]
            connect,
        )

        super().__init__()
        self.conn = connect(
            instance_id,
            database_id,
            project_id,
            user_agent=user_agent,
            **(connect_kwargs or {}),
        )
        self.conn.autocommit = autocommit
        self.lock = threading.Lock()
        self.table_names = table_names
        self.CREATE_CHECKPOINT_DDL = (
            f"CREATE TABLE IF NOT EXISTS {self.table_names['checkpoints']} ("
            + """
            thread_id STRING(1024) NOT NULL,
            checkpoint_ns STRING(1024) NOT NULL DEFAULT (''),
            checkpoint_id STRING(1024) NOT NULL,
            parent_checkpoint_id STRING(1024),
            checkpoint STRING(MAX) NOT NULL,
            metadata STRING(MAX) NOT NULL DEFAULT ('{}'),
        ) PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
        """
        )
        self.CREATE_CHECKPOINT_WRITES_DDL = (
            f"CREATE TABLE IF NOT EXISTS {self.table_names['checkpoint_writes']} ("
            + """
            thread_id STRING(1024) NOT NULL,
            checkpoint_ns STRING(1024) NOT NULL DEFAULT (''),
            checkpoint_id STRING(1024) NOT NULL,
            task_id STRING(1024) NOT NULL,
            idx INT64 NOT NULL,
            channel STRING(1024) NOT NULL,
            value STRING(MAX) NOT NULL,
        ) PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
        """
        )

    def setup(self) -> None:
        """Set up the checkpoint database.

        This method creates the necessary tables in the Spanner database if they don't
        already exist.
        """
        with self.cursor() as cur:
            cur.execute(self.CREATE_CHECKPOINT_DDL)
            cur.execute(self.CREATE_CHECKPOINT_WRITES_DDL)

    @contextmanager
    def cursor(self) -> Iterator[Cursor]:
        """Get a cursor for the Spanner database.

        This method returns a cursor for the Spanner database.

        Yields:
            Cursor: A cursor for the Spanner database.
        """
        with self.lock:
            cur = self.conn.cursor()
            try:
                yield cur
            finally:
                cur.close()

    def list(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        """List checkpoints from the Spanner database.

        This method retrieves a list of checkpoint tuples from the Spanner database based
        on the provided config. The checkpoints are ordered by checkpoint ID in descending order (newest first).

        Args:
            config (RunnableConfig): The config to use for listing the checkpoints.
            filter (Optional[Dict[str, Any]]): Additional filtering criteria for metadata. Ignored for now.
            before (Optional[RunnableConfig]): If provided, only checkpoints before the specified checkpoint ID are returned. Defaults to None.
            limit (Optional[int]): The maximum number of checkpoints to return. Defaults to None.

        Yields:
            Iterator[CheckpointTuple]: An iterator of checkpoint tuples.

        Examples:
            >>> from langchain_google_spanner import SpannerCheckpointSaver
            >>> checkpointer = SpannerCheckpointSaver(instance_id, database_id, project_id)
            ... # Run a graph, then list the checkpoints
            >>> config = {"configurable": {"thread_id": "1"}}
            >>> checkpoints = list(checkpointer.list(config, limit=2))
            >>> print(checkpoints)
            [CheckpointTuple(...), CheckpointTuple(...)]

            >>> config = {"configurable": {"thread_id": "1"}}
            >>> before = {"configurable": {"checkpoint_id": "1ef4f797-8335-6428-8001-8a1503f9b875"}}
            >>> checkpointer = SpannerCheckpointSaver(instance_id, database_id, project_id)
            ... # Run a graph, then list the checkpoints
            >>> checkpoints = list(checkpointer.list(config, before=before))
            >>> print(checkpoints)
            [CheckpointTuple(...), ...]
        """
        where, param_values = _search_where(config, before)
        query = f"""SELECT thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, checkpoint, metadata
        FROM {self.table_names['checkpoints']}
        {where}
        ORDER BY checkpoint_id DESC"""
        if limit:
            query += f" LIMIT {limit}"
        with self.cursor() as cur, closing(self.conn.cursor()) as wcur:
            cur.execute(query, param_values)
            for (
                thread_id,
                checkpoint_ns,
                checkpoint_id,
                parent_checkpoint_id,
                checkpoint,
                metadata,
            ) in cur:
                wcur.execute(
                    f"SELECT task_id, channel, value FROM {self.table_names['checkpoint_writes']} WHERE thread_id = %s AND checkpoint_ns = %s AND checkpoint_id = %s ORDER BY task_id, idx",
                    (thread_id, checkpoint_ns, checkpoint_id),
                )
                yield _load_checkpoint_tuple(
                    serde=self.serde,
                    cur=wcur,
                    config=_config(thread_id, checkpoint_ns, checkpoint_id),
                    thread_id=thread_id,
                    checkpoint=checkpoint,
                    checkpoint_ns=checkpoint_ns,
                    parent_checkpoint_id=parent_checkpoint_id,
                    metadata=metadata,
                )

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:  # type: ignore[return]
        """Get a checkpoint tuple from the Spanner database.

        This method retrieves a checkpoint tuple from the Spanner database based on the
        provided config. If the config contains a "checkpoint_id" key, the checkpoint with
        the matching thread ID and checkpoint ID is retrieved. Otherwise, the latest checkpoint
        for the given thread ID is retrieved.

        Args:
            config (RunnableConfig): The config to use for retrieving the checkpoint.

        Returns:
            Optional[CheckpointTuple]: The retrieved checkpoint tuple, or None if no matching checkpoint was found.

        Examples:

            Basic:
            >>> from langchain_google_spanner import SpannerCheckpointSaver
            >>> checkpointer = SpannerCheckpointSaver(instance_id, database_id, project_id)
            >>> config = {"configurable": {"thread_id": "1"}}
            >>> checkpoint_tuple = checkpointer.get_tuple(config)
            >>> print(checkpoint_tuple)
            CheckpointTuple(...)

            With checkpoint ID:

            >>> config = {
            ...    "configurable": {
            ...        "thread_id": "1",
            ...        "checkpoint_ns": "",
            ...        "checkpoint_id": "1ef4f797-8335-6428-8001-8a1503f9b875",
            ...    }
            ... }
            >>> checkpoint_tuple = checkpointer.get_tuple(config)
            >>> print(checkpoint_tuple)
            CheckpointTuple(...)
        """  # noqa
        _checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        with self.cursor() as cur:
            if checkpoint_id := get_checkpoint_id(config):
                sql = f"SELECT thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, checkpoint, metadata FROM {self.table_names['checkpoints']} WHERE thread_id = %s AND checkpoint_ns = %s AND checkpoint_id = %s"
                sql_params = (  # type: ignore[assignment]
                    config["configurable"]["thread_id"],
                    _checkpoint_ns,
                    checkpoint_id,
                )
            else:  # find the latest checkpoint for the thread_id
                sql = f"SELECT thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, checkpoint, metadata FROM {self.table_names['checkpoints']} WHERE thread_id = %s AND checkpoint_ns = %s ORDER BY checkpoint_id DESC LIMIT 1"
                sql_params = (config["configurable"]["thread_id"], _checkpoint_ns)  # type: ignore[assignment]
            cur.execute(sql, sql_params)
            if _checkpoint := cur.fetchone():
                (
                    thread_id,
                    checkpoint_ns,
                    checkpoint_id,
                    parent_checkpoint_id,
                    checkpoint,
                    metadata,
                ) = _checkpoint
                if not get_checkpoint_id(config):
                    config = _config(thread_id, checkpoint_ns, checkpoint_id)
                cur.execute(  # find any pending writes
                    f"SELECT task_id, channel, value FROM {self.table_names['checkpoint_writes']} WHERE thread_id = %s AND checkpoint_ns = %s AND checkpoint_id = %s ORDER BY task_id, idx",
                    (thread_id, checkpoint_ns, checkpoint_id),
                )
                return _load_checkpoint_tuple(
                    serde=self.serde,
                    cur=cur,
                    config=config,
                    thread_id=thread_id,
                    checkpoint=checkpoint,
                    checkpoint_ns=checkpoint_ns,
                    parent_checkpoint_id=parent_checkpoint_id,
                    metadata=metadata,
                )

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Save a checkpoint to the Spanner database.

        This method saves a checkpoint to the Spanner database. The checkpoint is associated
        with the provided config and its parent config (if any).

        Args:
            config (RunnableConfig): The config to associate with the checkpoint.
            checkpoint (Checkpoint): The checkpoint to save.
            metadata (CheckpointMetadata): Additional metadata to save with the checkpoint.
            new_versions (ChannelVersions): Ignored for now.

        Returns:
            RunnableConfig: Updated configuration after storing the checkpoint.

        Examples:

            >>> from langchain_google_spanner import SpannerCheckpointSaver
            >>> checkpointer = SpannerCheckpointSaver(instance_id, database_id, project_id)
            >>> config = {"configurable": {"thread_id": "1", "checkpoint_ns": ""}}
            >>> checkpoint = {"ts": "2024-05-04T06:32:42.235444+00:00", "id": "1ef4f797-8335-6428-8001-8a1503f9b875", "channel_values": {"key": "value"}}
            >>> saved_config = checkpointer.put(config, checkpoint, {"source": "input", "step": 1, "writes": {"key": "value"}}, {})
            >>> print(saved_config)
            {'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef4f797-8335-6428-8001-8a1503f9b875'}}
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"]["checkpoint_ns"]
        with self.cursor() as cur:
            cur.execute(
                f"INSERT OR UPDATE INTO {self.table_names['checkpoints']} (thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, checkpoint, metadata) VALUES (%s, %s, %s, %s, %s, %s)",
                (
                    thread_id,
                    checkpoint_ns,
                    checkpoint["id"],
                    config["configurable"].get("checkpoint_id"),
                    dumps(checkpoint),
                    dumps(metadata),
                ),
            )
        return _config(thread_id, checkpoint_ns, checkpoint["id"])

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
    ) -> None:
        """Store intermediate writes linked to a checkpoint.

        This method saves intermediate writes associated with a checkpoint to the Spanner database.

        Args:
            config (RunnableConfig): Configuration of the related checkpoint.
            writes (Sequence[Tuple[str, Any]]): List of writes to store, each as (channel, value) pair.
            task_id (str): Identifier for the task creating the writes.
        """
        query = (
            f"INSERT OR REPLACE INTO {self.table_names['checkpoint_writes']} (thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel, value) VALUES (%s, %s, %s, %s, %s, %s, %s)"
            if all(w[0] in WRITES_IDX_MAP for w in writes)
            else f"INSERT OR IGNORE INTO {self.table_names['checkpoint_writes']} (thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel, value) VALUES (%s, %s, %s, %s, %s, %s, %s)"
        )
        with self.cursor() as cur:
            cur.executemany(
                query,
                [
                    (
                        config["configurable"]["thread_id"],
                        config["configurable"]["checkpoint_ns"],
                        config["configurable"]["checkpoint_id"],
                        task_id,
                        WRITES_IDX_MAP.get(channel, idx),
                        channel,
                        dumps(value),
                    )
                    for idx, (channel, value) in enumerate(writes)
                ],
            )


def _load_checkpoint_tuple(
    serde: SerializerProtocol,
    cur: Cursor,
    config: RunnableConfig,
    thread_id: str,
    checkpoint: str,
    checkpoint_ns: str,
    parent_checkpoint_id: str,
    metadata: str,
) -> CheckpointTuple:
    return CheckpointTuple(
        config,
        serde.loads(checkpoint),  # type: ignore[arg-type]
        serde.loads(metadata),  # type: ignore[arg-type]
        (
            _config(thread_id, checkpoint_ns, parent_checkpoint_id)
            if parent_checkpoint_id
            else None
        ),
        [(task_id, channel, serde.loads(_value)) for task_id, channel, _value in cur],
    )


def _search_where(
    config: Optional[RunnableConfig],
    before: Optional[RunnableConfig] = None,
) -> Tuple[str, Sequence[Any]]:
    """Return WHERE clause predicates for search() given `before` config.

    This method returns a tuple of a string and a tuple of values. The string
    is the parametered WHERE clause predicate (including the WHERE keyword):
    "WHERE column1 = %s AND column2 IS %s". The tuple of values contains the
    values for each of the corresponding parameters.
    """
    wheres = []
    param_values = []

    if config is not None:
        wheres.append("thread_id = %s")
        param_values.append(config["configurable"]["thread_id"])
        checkpoint_ns = config["configurable"].get("checkpoint_ns")
        if checkpoint_ns is not None:
            wheres.append("checkpoint_ns = %s")
            param_values.append(checkpoint_ns)

        if checkpoint_id := get_checkpoint_id(config):
            wheres.append("checkpoint_id = %s")
            param_values.append(checkpoint_id)

    if before is not None:
        wheres.append("checkpoint_id < %s")
        param_values.append(get_checkpoint_id(before))

    return ("WHERE " + " AND ".join(wheres) if wheres else "", param_values)
