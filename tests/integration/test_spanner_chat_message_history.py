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
from google.cloud.spanner import Client  # type: ignore
from langchain_core.messages.ai import AIMessage
from langchain_core.messages.human import HumanMessage

from langchain_google_spanner import SpannerChatMessageHistory

project_id = os.environ["PROJECT_ID"]
instance_id = os.environ["INSTANCE_ID"]
table_name = "test_table" + str(uuid.uuid4()).replace("-", "_")

OPERATION_TIMEOUT_SECONDS = 240


@pytest.fixture(scope="module")
def client() -> Client:
    return Client(project=project_id)


@pytest.fixture(scope="module")
def setup(client):
    for env in ["GOOGLE_DATABASE", "PG_DATABASE"]:
        database_id = os.environ.get(env)
        SpannerChatMessageHistory.create_chat_history_table(
            instance_id, database_id, table_name
        )
    yield
    for env in ["GOOGLE_DATABASE", "PG_DATABASE"]:
        database_id = os.environ.get(env)
        database = client.instance(instance_id).database(database_id)
        operation = database.update_ddl([f"DROP TABLE IF EXISTS {table_name}"])
        operation.result(OPERATION_TIMEOUT_SECONDS)


def test_chat_message_history(setup) -> None:
    for env in ["GOOGLE_DATABASE", "PG_DATABASE"]:
        database_id = os.environ.get(env)
        assert database_id is not None
        history = SpannerChatMessageHistory(
            instance_id=instance_id,
            database_id=database_id,
            session_id="test-session",
            table_name=table_name,
        )
        history.add_user_message("hi!")
        history.add_ai_message("whats up?")
        messages = history.messages

        # verify messages are correct
        assert messages[0].content == "hi!"
        assert type(messages[0]) is HumanMessage
        assert messages[1].content == "whats up?"
        assert type(messages[1]) is AIMessage

        # verify clear() clears message history
        history.clear()
        assert len(history.messages) == 0
