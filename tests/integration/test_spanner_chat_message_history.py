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

from langchain_core.messages.ai import AIMessage
from langchain_core.messages.human import HumanMessage

from langchain_google_spanner import SpannerChatMessageHistory

instance_id = os.environ["INSTANCE_ID"]
database_id = os.environ["DATABASE_ID"]


def test_chat_message_history() -> None:
    history = SpannerChatMessageHistory(
        instance_id=instance_id, database_id=database_id, session_id="test"
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
