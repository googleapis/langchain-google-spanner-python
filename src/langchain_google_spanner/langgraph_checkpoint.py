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
from typing import Any, Dict, Iterator, Optional, Sequence, Tuple

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


class SpannerCheckpointSaver(BaseCheckpointSaver[str]):
    CREATE_CHECKPOINT_DDL = """
    CREATE TABLE IF NOT EXISTS checkpoints (
        thread_id STRING(1024) NOT NULL,
        checkpoint_ns STRING(1024) NOT NULL DEFAULT (''),
        checkpoint_id STRING(1024) NOT NULL,
        parent_checkpoint_id STRING(1024),
        checkpoint STRING(MAX) NOT NULL,
        metadata STRING(MAX) NOT NULL DEFAULT ('{}'),
    ) PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
    """
    CREATE_CHECKPOINT_WRITES_DDL = """
    CREATE TABLE IF NOT EXISTS checkpoint_writes (
        thread_id STRING(1024) NOT NULL,
        checkpoint_ns STRING(1024) NOT NULL DEFAULT (''),
        checkpoint_id STRING(1024) NOT NULL,
        task_id STRING(1024) NOT NULL,
        idx INT64 NOT NULL,
        channel STRING(1024) NOT NULL,
        value STRING(MAX) NOT NULL,
    ) PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
    """
    lock: threading.Lock

    def __init__(
        self,
        instance_id: str,
        database_id: str,
        project_id: str,
        user_agent=USER_AGENT_CHECKPOINTER,
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

    def setup(self) -> None:
        with self.cursor() as cur:
            cur.execute(self.CREATE_CHECKPOINT_DDL)
            cur.execute(self.CREATE_CHECKPOINT_WRITES_DDL)

    @contextmanager
    def cursor(self) -> Iterator[Cursor]:
        """Get a cursor for the Spanner database.

        This method returns a cursor for the Spanner database. It is used internally
        by SpannerSaver and should not be called directly by the user.

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
        where, param_values = _search_where(config, before)
        query = f"""SELECT thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, checkpoint, metadata
        FROM checkpoints
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
                    "SELECT task_id, channel, value FROM checkpoint_writes WHERE thread_id = %s AND checkpoint_ns = %s AND checkpoint_id = %s ORDER BY task_id, idx",
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
        _checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        with self.cursor() as cur:
            if checkpoint_id := get_checkpoint_id(config):
                sql = "SELECT thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, checkpoint, metadata FROM checkpoints WHERE thread_id = %s AND checkpoint_ns = %s AND checkpoint_id = %s"
                sql_params = (  # type: ignore[assignment]
                    config["configurable"]["thread_id"],
                    _checkpoint_ns,
                    checkpoint_id,
                )
            else:  # find the latest checkpoint for the thread_id
                sql = "SELECT thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, checkpoint, metadata FROM checkpoints WHERE thread_id = %s AND checkpoint_ns = %s ORDER BY checkpoint_id DESC LIMIT 1"
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
                    "SELECT task_id, channel, value FROM checkpoint_writes WHERE thread_id = %s AND checkpoint_ns = %s AND checkpoint_id = %s ORDER BY task_id, idx",
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
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"]["checkpoint_ns"]
        with self.cursor() as cur:
            cur.execute(
                "INSERT OR UPDATE INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, checkpoint, metadata) VALUES (%s, %s, %s, %s, %s, %s)",
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
        writes: Sequence[tuple[str, Any]],
        task_id: str,
    ) -> None:
        query = (
            "INSERT OR REPLACE INTO checkpoint_writes (thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel, value) VALUES (%s, %s, %s, %s, %s, %s, %s)"
            if all(w[0] in WRITES_IDX_MAP for w in writes)
            else "INSERT OR IGNORE INTO checkpoint_writes (thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel, value) VALUES (%s, %s, %s, %s, %s, %s, %s)"
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
