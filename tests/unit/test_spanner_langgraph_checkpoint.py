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

import contextlib
import json
import operator
from typing import Annotated, Any, Iterator, Optional, Sequence, Tuple, Union
from unittest import mock

import pytest
from google.auth import credentials as auth_credentials  # type: ignore[import-untyped]
from google.cloud.spanner_dbapi import connection  # type: ignore[import-untyped]
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    create_checkpoint,
    empty_checkpoint,
)
from langgraph.checkpoint.serde.types import ChannelProtocol
from langgraph.graph.graph import START
from langgraph.graph.state import StateGraph

from langchain_google_spanner import langgraph_checkpoint
from langchain_google_spanner.langgraph_checkpoint import SpannerCheckpointSaver

_TEST_CREDENTIALS = mock.Mock(spec=auth_credentials.AnonymousCredentials())
_TEST_INSTANCE_ID = "test-instance-id"
_TEST_DATABASE_ID = "test-database-id"
_TEST_PROJECT_ID = "test-project-id"
_TEST_CHECKPOINT_ID = "1ef97be6-668b-62a2-8003-9376ddec2770"
_TEST_CHECKPOINT_NS = ""
_TEST_PARENT_CHECKPOINT_ID = "1ef7b8c8-f35a-60de-8000-c1ac6a4bafd5"
_TEST_THREAD_ID = "thread-1"
_TEST_CHECKPOINT = {
    "id": _TEST_CHECKPOINT_ID,
    "channel_values": {},
    "channel_versions": {},
    "pending_sends": [],
    "versions_seen": {},
}
_TEST_CHECKPOINT_METADATA = {"step": 0}
_TEST_PARENT_CONFIG = {
    "configurable": {
        "thread_id": _TEST_THREAD_ID,
        "checkpoint_ns": _TEST_CHECKPOINT_NS,
        "checkpoint_id": _TEST_PARENT_CHECKPOINT_ID,
    }
}
_TEST_CURSOR_RETURN_VALUE = (
    _TEST_THREAD_ID,
    _TEST_CHECKPOINT_NS,
    _TEST_CHECKPOINT_ID,
    _TEST_PARENT_CHECKPOINT_ID,
    json.dumps(_TEST_CHECKPOINT),
    json.dumps(_TEST_CHECKPOINT_METADATA),
)
_TEST_CURSOR_WRITES_RETURN_VALUE = (  # pending_writes
    "",  # task_id
    "",  # channel
    "{}",  # value
)


@pytest.fixture(scope="module")
def dbapi_connect_mock():
    with mock.patch.object(connection, "connect") as dbapi_connect_mock:
        dbapi_connect_mock.return_value.cursor.return_value.fetchone.return_value = (
            _TEST_CURSOR_RETURN_VALUE
        )
        yield dbapi_connect_mock


@pytest.fixture(scope="module")
def contextlib_closing_mock():
    with mock.patch.object(contextlib, "closing") as contextlib_closing_mock:
        yield contextlib_closing_mock


class FakeCursor:
    def __init__(self, return_value):
        self._index = 0
        self._return_value = return_value
        self._execute_statements = []

    def __iter__(self):
        return self

    def __next__(self):
        if self._index > 0:
            raise StopIteration
        self._index += 1
        return self._return_value

    def execute(self, sql, *args, **kwargs):
        self._execute_statements.append(sql)
        return None


class FakeCursorCheckpointer(SpannerCheckpointSaver):

    @contextlib.contextmanager
    def cursor(self) -> Iterator:
        self._cursor = FakeCursor(_TEST_CURSOR_RETURN_VALUE)
        yield self._cursor


class FaultyGetCheckpointer(SpannerCheckpointSaver):

    def get_tuple(
        self,
        config: RunnableConfig,
    ) -> Optional[CheckpointTuple]:
        raise ValueError("Faulty get_tuple")


class FaultyPutCheckpointer(SpannerCheckpointSaver):
    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: Optional[dict[str, Union[str, int, float]]] = None,
    ) -> RunnableConfig:
        raise ValueError("Faulty put")


class FaultyPutWritesCheckpointer(SpannerCheckpointSaver):
    def put_writes(  # type: ignore[override]
        self,
        config: RunnableConfig,
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
    ) -> RunnableConfig:
        raise ValueError("Faulty put_writes")


class FaultyVersionCheckpointer(SpannerCheckpointSaver):
    def get_next_version(
        self,
        current: Optional[str],
        channel: ChannelProtocol[Any, Any, Any],
    ) -> str:
        raise ValueError("Faulty get_next_version")


def logic(inp: str) -> str:
    return ""


def _test_builder() -> StateGraph:
    builder = StateGraph(Annotated[str, operator.add])  # type: ignore[arg-type]
    builder.add_node("agent", logic)
    builder.add_edge(START, "agent")
    return builder


class TestSpannerLanggraphCheckpoint:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        # objects for test setup
        self.config_1: RunnableConfig = {
            "configurable": {
                "thread_id": "thread-1",
                "thread_ts": "1",  # for backwards compatibility testing
                "checkpoint_ns": "",
            }
        }
        self.config_2: RunnableConfig = {
            "configurable": {
                "thread_id": "thread-2",
                "checkpoint_id": "2",
                "checkpoint_ns": "",
            }
        }
        self.config_3: RunnableConfig = {
            "configurable": {
                "thread_id": "thread-2",
                "checkpoint_id": "2-inner",
                "checkpoint_ns": "inner",
            }
        }

        self.chkpnt_1: Checkpoint = empty_checkpoint()
        self.chkpnt_2: Checkpoint = create_checkpoint(self.chkpnt_1, {}, 1)
        self.chkpnt_3: Checkpoint = empty_checkpoint()

        self.metadata_1: CheckpointMetadata = {
            "source": "input",
            "step": 2,
            "writes": {},
        }
        self.metadata_2: CheckpointMetadata = {
            "source": "loop",
            "step": 1,
            "writes": {"foo": "bar"},
        }
        self.metadata_3: CheckpointMetadata = {}

    def test_setup(self, dbapi_connect_mock) -> None:
        saver = FakeCursorCheckpointer(
            instance_id=_TEST_INSTANCE_ID,
            database_id=_TEST_DATABASE_ID,
            project_id=_TEST_PROJECT_ID,
            connect_kwargs={"credentials": _TEST_CREDENTIALS},
        )
        saver.setup()
        assert saver._cursor._execute_statements == [
            """
            CREATE TABLE IF NOT EXISTS checkpoints (
                thread_id STRING(1024) NOT NULL,
                checkpoint_ns STRING(1024) NOT NULL DEFAULT (''),
                checkpoint_id STRING(1024) NOT NULL,
                parent_checkpoint_id STRING(1024),
                checkpoint STRING(MAX) NOT NULL,
                metadata STRING(MAX) NOT NULL DEFAULT ('{}'),
            ) PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
            """,
            """
            CREATE TABLE IF NOT EXISTS checkpoint_writes (
                thread_id STRING(1024) NOT NULL,
                checkpoint_ns STRING(1024) NOT NULL DEFAULT (''),
                checkpoint_id STRING(1024) NOT NULL,
                task_id STRING(1024) NOT NULL,
                idx INT64 NOT NULL,
                channel STRING(1024) NOT NULL,
                value STRING(MAX) NOT NULL,
            ) PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
            """,
        ]

    def test_put(self, dbapi_connect_mock) -> None:
        saver = FakeCursorCheckpointer(
            instance_id=_TEST_INSTANCE_ID,
            database_id=_TEST_DATABASE_ID,
            project_id=_TEST_PROJECT_ID,
            connect_kwargs={"credentials": _TEST_CREDENTIALS},
        )
        saver.setup()
        saver.put(self.config_1, self.chkpnt_1, self.metadata_1, {})
        saver.put(self.config_2, self.chkpnt_2, self.metadata_2, {})
        saver.put(self.config_3, self.chkpnt_3, self.metadata_3, {})

        # search_results_0 = list(saver.list(None))
        # assert len(search_results_0) == 0

    def test_list(self, dbapi_connect_mock) -> None:
        saver = FakeCursorCheckpointer(
            instance_id=_TEST_INSTANCE_ID,
            database_id=_TEST_DATABASE_ID,
            project_id=_TEST_PROJECT_ID,
            connect_kwargs={"credentials": _TEST_CREDENTIALS},
        )
        assert len(list(saver.list(self.config_1))) == 1
        assert len(list(saver.list(self.config_2, limit=2))) == 1
        assert len(list(saver.list(None, before=self.config_3))) == 1

    def test_search_where_none_config(self) -> None:
        expected_predicate_1 = "WHERE checkpoint_id < %s"
        expected_param_values_1 = ["1"]
        assert langgraph_checkpoint._search_where(
            config=None,
            before=self.config_1,
        ) == (
            expected_predicate_1,
            expected_param_values_1,
        )

    def test_search_where_none_before(self) -> None:
        expected_predicate = "WHERE thread_id = %s AND checkpoint_ns = %s AND checkpoint_id = %s"
        expected_param_values = ["thread-1", "", "1"]
        assert langgraph_checkpoint._search_where(config=self.config_1) == (
            expected_predicate,
            expected_param_values,
        )

    def test_get_tuple(self, dbapi_connect_mock):
        saver = SpannerCheckpointSaver(
            instance_id=_TEST_INSTANCE_ID,
            database_id=_TEST_DATABASE_ID,
            project_id=_TEST_PROJECT_ID,
            connect_kwargs={"credentials": _TEST_CREDENTIALS},
        )
        assert saver.get_tuple(self.config_1) == CheckpointTuple(
            config=self.config_1,
            checkpoint=_TEST_CHECKPOINT,
            metadata=_TEST_CHECKPOINT_METADATA,
            parent_config=_TEST_PARENT_CONFIG,
            pending_writes=[],
        )
        assert saver.get_tuple(self.config_2) == CheckpointTuple(
            config=self.config_2,
            checkpoint=_TEST_CHECKPOINT,
            metadata=_TEST_CHECKPOINT_METADATA,
            parent_config=_TEST_PARENT_CONFIG,
            pending_writes=[],
        )
        assert saver.get_tuple(self.config_3) == CheckpointTuple(
            config=self.config_3,
            checkpoint=_TEST_CHECKPOINT,
            metadata=_TEST_CHECKPOINT_METADATA,
            parent_config=_TEST_PARENT_CONFIG,
            pending_writes=[],
        )

    def test_yield_checkpoint_write(self, dbapi_connect_mock):
        saver = SpannerCheckpointSaver(
            instance_id=_TEST_INSTANCE_ID,
            database_id=_TEST_DATABASE_ID,
            project_id=_TEST_PROJECT_ID,
            connect_kwargs={"credentials": _TEST_CREDENTIALS},
        )
        results = list(
            langgraph_checkpoint._yield_checkpoint_write(
                saver.serde,
                FakeCursor(_TEST_CURSOR_RETURN_VALUE),
                FakeCursor(_TEST_CURSOR_WRITES_RETURN_VALUE),
            )
        )
        assert results == [
            CheckpointTuple(
                config={
                    "configurable": {
                        "thread_id": self.config_1["configurable"]["thread_id"],
                        "checkpoint_ns": self.config_1["configurable"]["checkpoint_ns"],
                        "checkpoint_id": _TEST_CHECKPOINT_ID,
                    }
                },
                checkpoint=_TEST_CHECKPOINT,
                metadata=_TEST_CHECKPOINT_METADATA,
                parent_config=_TEST_PARENT_CONFIG,
                pending_writes=[("", "", {})],
            )
        ]


# https://github.com/langchain-ai/langgraph/pull/2089#issuecomment-2417606590
class TestSpannerLanggraphCheckpointErrors:

    def test_get_tuple_error(self, dbapi_connect_mock):

        graph = _test_builder().compile(
            checkpointer=FaultyGetCheckpointer(
                instance_id=_TEST_INSTANCE_ID,
                database_id=_TEST_DATABASE_ID,
                project_id=_TEST_PROJECT_ID,
                connect_kwargs={"credentials": _TEST_CREDENTIALS},
            )
        )
        with pytest.raises(ValueError, match="Faulty get_tuple"):
            graph.invoke("", {"configurable": {"thread_id": "thread-1"}})

    def test_put_error(self, dbapi_connect_mock):
        graph = _test_builder().compile(
            checkpointer=FaultyPutCheckpointer(
                instance_id=_TEST_INSTANCE_ID,
                database_id=_TEST_DATABASE_ID,
                project_id=_TEST_PROJECT_ID,
                connect_kwargs={"credentials": _TEST_CREDENTIALS},
            )
        )
        with pytest.raises(ValueError, match="Faulty put"):
            graph.invoke("", {"configurable": {"thread_id": "thread-1"}})

    def test_get_next_version_error(self, dbapi_connect_mock):
        graph = _test_builder().compile(
            checkpointer=FaultyVersionCheckpointer(
                instance_id=_TEST_INSTANCE_ID,
                database_id=_TEST_DATABASE_ID,
                project_id=_TEST_PROJECT_ID,
                connect_kwargs={"credentials": _TEST_CREDENTIALS},
            )
        )
        with pytest.raises(ValueError, match="Faulty get_next_version"):
            graph.invoke("", {"configurable": {"thread_id": "thread-1"}})

    def test_put_writes_error(self, dbapi_connect_mock):
        builder = _test_builder()
        builder.add_node("parallel", logic)
        builder.add_edge(START, "parallel")
        graph = builder.compile(
            checkpointer=FaultyPutWritesCheckpointer(
                instance_id=_TEST_INSTANCE_ID,
                database_id=_TEST_DATABASE_ID,
                project_id=_TEST_PROJECT_ID,
                connect_kwargs={"credentials": _TEST_CREDENTIALS},
            )
        )
        with pytest.raises(ValueError, match="Faulty put_writes"):
            graph.invoke("", {"configurable": {"thread_id": "thread-1"}})
