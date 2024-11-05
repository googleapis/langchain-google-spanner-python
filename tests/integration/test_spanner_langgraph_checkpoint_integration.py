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

import os
import uuid

import pytest  # noqa

from langchain_google_spanner import SpannerCheckpointSaver

project_id = os.environ["PROJECT_ID"]
instance_id = os.environ["INSTANCE_ID"]

_TEST_CHECKPOINT = {
    "v": 1,
    "ts": "2024-07-31T20:14:19.804150+00:00",
    "id": "1ef4f797-8335-6428-8001-8a1503f9b875",
    "channel_values": {"my_key": "meow", "node": "node"},
    "channel_versions": {"__start__": 2, "my_key": 3, "start:node": 3, "node": 3},
    "versions_seen": {
        "__input__": {},
        "__start__": {"__start__": 1},
        "node": {"start:node": 2},
    },
    "pending_sends": [],
}
_TEST_WRITE_CONFIG = {"configurable": {"thread_id": "1", "checkpoint_ns": ""}}
_TEST_READ_CONFIG = {"configurable": {"thread_id": "1"}}


@pytest.fixture(scope="module")
def setup():
    for env in ["GOOGLE_DATABASE"]:  # , "PG_DATABASE"]:
        checkpointer = SpannerCheckpointSaver(
            instance_id=instance_id,
            database_id=os.environ.get(env, ""),
            project_id=project_id,
        )
        with checkpointer.cursor() as cur:
            cur.executemany(
                "DROP TABLE IF EXISTS %s",
                ["checkpoints", "checkpoint_writes"],
            )
        checkpointer.setup()


def test_chat_message_history(setup) -> None:
    for env in ["GOOGLE_DATABASE"]:  # , "PG_DATABASE"]:
        checkpointer = SpannerCheckpointSaver(
            instance_id=instance_id,
            database_id=os.environ.get(env, ""),
            project_id=project_id,
        )
        checkpointer.put(_TEST_WRITE_CONFIG, _TEST_CHECKPOINT, {}, {})  # type: ignore[arg-type]
        checkpoint = checkpointer.get(_TEST_READ_CONFIG)  # type: ignore[arg-type]
        assert checkpoint == _TEST_CHECKPOINT
        assert len(list(checkpointer.list(_TEST_READ_CONFIG))) == 1  # type: ignore[arg-type]
