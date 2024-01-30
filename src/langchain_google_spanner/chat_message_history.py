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

"""Cloud Spanner-based chat message history"""
from __future__ import annotations

import uuid
from typing import List, Optional

from google.cloud import spanner
from google.cloud.spanner_admin_database_v1.types import DatabaseDialect  # type: ignore
from google.cloud.spanner_v1 import param_types
from google.cloud.spanner_v1.data_types import JsonObject
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, message_to_dict, messages_from_dict

OPERATION_TIMEOUT_SECONDS = 240

COLUMN_FAMILY = "langchain"
COLUMN_NAME = "history"


class SpannerChatMessageHistory(BaseChatMessageHistory):
    """Chat message history that stores history in Spanner.

    Args:
        instance_id: The Spanner instance to use for chat message history.
        database_id: The Spanner database to use for chat message history.
        table_name: The Spanner table to use for chat message history.
        session_id: Optional. The existing session ID.
    """

    def __init__(
        self,
        instance_id: str,
        database_id: str,
        table_name: str = "message_store",
        session_id: Optional[str] = None,
        client: Optional[spanner.Client] = None,
    ) -> None:
        self.instance_id = instance_id
        self.database_id = database_id
        self.table_name = table_name
        self.client = client or spanner.Client()
        self.instance = self.client.instance(instance_id)
        if not self.instance.exists():
            raise Exception("Instance doesn't exist.")
        self.database = self.instance.database(database_id)
        if not self.database.exists():
            raise Exception("Database doesn't exist.")
        self.database.reload()
        self.dialect = self.database.database_dialect
        self.session_id = session_id or uuid.uuid4().hex
        self._create_table_if_not_exists()

    def _create_table_if_not_exists(self) -> None:
        google_schema = f"""CREATE TABLE IF NOT EXISTS {self.table_name} (
                            id STRING(36) DEFAULT (GENERATE_UUID()),
                            created_at TIMESTAMP NOT NULL OPTIONS (allow_commit_timestamp=true),
                            session_id STRING(MAX) NOT NULL,
                            message JSON NOT NULL,
                         ) PRIMARY KEY (session_id, created_at ASC, id)"""

        pg_schema = f"""CREATE TABLE IF NOT EXISTS {self.table_name}  (
                             id varchar(36) DEFAULT (spanner.generate_uuid()),
                             created_at SPANNER.COMMIT_TIMESTAMP NOT NULL,
                             session_id TEXT NOT NULL,
                             message JSONB NOT NULL,
                             PRIMARY KEY (session_id, created_at ASC, id)
                         );"""

        ddl = pg_schema if self.dialect == DatabaseDialect.POSTGRESQL else google_schema
        database = self.client.instance(self.instance_id).database(self.database_id)
        operation = database.update_ddl([ddl])
        operation.result(OPERATION_TIMEOUT_SECONDS)
        return operation

    @property
    def messages(self) -> List[BaseMessage]:  # type: ignore
        """Retrieve the messages from Cloud Spanner"""
        place_holder = "$1" if self.dialect == DatabaseDialect.POSTGRESQL else "@p1"
        query = f"SELECT message FROM {self.table_name} WHERE session_id = {place_holder} ORDER BY created_at;"
        param = {"p1": self.session_id}
        param_type = {"p1": param_types.STRING}

        with self.database.snapshot() as snapshot:
            results = snapshot.execute_sql(
                query,
                params=param,
                param_types=param_type,
            )
        items = [result[0] for result in results]
        messages = messages_from_dict(items)
        return messages

    def add_message(self, message: BaseMessage) -> None:
        """Append the message to the record in Cloud Spanner"""
        with self.database.batch() as batch:
            batch.insert(
                table=self.table_name,
                columns=("session_id", "created_at", "message"),
                values=[
                    (
                        self.session_id,
                        spanner.COMMIT_TIMESTAMP,
                        JsonObject(message_to_dict(message)),
                    ),
                ],
            )

    def clear(self) -> None:
        """Clear session memory from Cloud Spanner"""
        place_holder = "$1" if self.dialect == DatabaseDialect.POSTGRESQL else "@p1"
        query = f"DELETE FROM {self.table_name} WHERE session_id = {place_holder};"
        param = {"p1": self.session_id}
        param_type = {"p1": param_types.STRING}
        self.database.execute_partitioned_dml(query, param, param_type)
