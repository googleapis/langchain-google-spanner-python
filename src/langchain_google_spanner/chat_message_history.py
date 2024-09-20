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

from typing import List, Optional

from google.cloud import spanner
from google.cloud.spanner_admin_database_v1.types import DatabaseDialect  # type: ignore
from google.cloud.spanner_v1 import param_types
from google.cloud.spanner_v1.data_types import JsonObject
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, messages_from_dict

from .version import __version__

USER_AGENT_CHAT = "langchain-google-spanner-python:chat_history/" + __version__

OPERATION_TIMEOUT_SECONDS = 240

COLUMN_FAMILY = "langchain"
COLUMN_NAME = "history"


def client_with_user_agent(
    client: Optional[spanner.Client], user_agent: str
) -> spanner.Client:
    if not client:
        client = spanner.Client()
    client_agent = client._client_info.user_agent
    if not client_agent:
        client._client_info.user_agent = user_agent
    elif user_agent not in client_agent:
        client._client_info.user_agent = " ".join([client_agent, user_agent])
    return client


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
        session_id: str,
        table_name: str,
        client: Optional[spanner.Client] = None,
    ) -> None:
        self.instance_id = instance_id
        self.database_id = database_id
        self.session_id = session_id
        self.table_name = table_name
        self.client = client_with_user_agent(client, USER_AGENT_CHAT)
        self.instance = self.client.instance(instance_id)
        if not self.instance.exists():
            raise Exception("Instance doesn't exist.")
        self.database = self.instance.database(database_id)
        if not self.database.exists():
            raise Exception("Database doesn't exist.")
        self.database.reload()
        self.dialect = self.database.database_dialect
        self._verify_schema()

    def _verify_schema(self) -> None:
        """Verify table exists with required schema for SpannerChatMessageHistory class.
        Use helper method MSSQLEngine.create_chat_history_table(...) to create
        table with valid schema.
        """
        # check table exists
        column_names = []  # type: List[str]
        with self.database.snapshot() as snapshot:
            results = snapshot.execute_sql(
                f"SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.columns WHERE table_name = '{self.table_name}'"
            )
            for row in results:
                column_names.append(*row)

        # check that all required columns are present
        required_columns = ["id", "session_id", "created_at", "message"]
        if len(column_names) == 0:
            raise AttributeError(
                f"Table '{self.table_name}' does not exist. Please create "
                "it before initializing SpannerChatMessageHistory. See "
                "SpannerEngine.create_chat_history_table() for a helper method."
            )
        else:
            if not (all(x in column_names for x in required_columns)):
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
                                             PRIMARY KEY (session_id, created_at, id)"""
                ddl = (
                    pg_schema
                    if self.dialect == DatabaseDialect.POSTGRESQL
                    else google_schema
                )
                raise IndexError(
                    f"Table '{self.table_name}' has incorrect schema. Got "
                    f"column names '{column_names}' but required column names "
                    f"'{required_columns}'.\nPlease create table with following schema:"
                    f"{ddl};"
                )

    @staticmethod
    def create_chat_history_table(
        instance_id: str,
        database_id: str,
        table_name: str,
        client: Optional[spanner.Client] = None,
    ) -> None:
        """
        Create a chat history table in a Cloud Spanner database.

        Args:
            instance_id (str): The ID of the Cloud Spanner instance.
            database_id (str): The ID of the Cloud Spanner database.
            table_name (str): The name of the table to be created.
            client (spanner.Client, optional): An instance of the Cloud Spanner client. Defaults to None.

        Raises:
            Exception: If the specified instance or database does not exist.

        Returns:
            Operation: The operation to create the table.
        """

        client = client_with_user_agent(client, USER_AGENT_CHAT)

        instance = client.instance(instance_id)

        if not instance.exists():
            raise Exception("Instance with id:  {} doesn't exist.".format(instance_id))

        database = instance.database(database_id)

        if not database.exists():
            raise Exception("Database with id: {} doesn't exist.".format(database_id))

        database.reload()

        dialect = database.database_dialect

        google_schema = f"""CREATE TABLE IF NOT EXISTS {table_name} (
                            id STRING(36) DEFAULT (GENERATE_UUID()),
                            created_at TIMESTAMP NOT NULL OPTIONS (allow_commit_timestamp=true),
                            session_id STRING(MAX) NOT NULL,
                            message JSON NOT NULL,
                         ) PRIMARY KEY (session_id, created_at ASC, id)"""

        pg_schema = f"""CREATE TABLE IF NOT EXISTS {table_name}  (
                             id varchar(36) DEFAULT (spanner.generate_uuid()),
                             created_at SPANNER.COMMIT_TIMESTAMP NOT NULL,
                             session_id TEXT NOT NULL,
                             message JSONB NOT NULL,
                             PRIMARY KEY (session_id, created_at, id)
                         );"""

        ddl = pg_schema if dialect == DatabaseDialect.POSTGRESQL else google_schema

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
        items = []  # type: List[dict]
        for row in results:
            items.append({"data": row[0], "type": row[0]["type"]})
        messages = messages_from_dict(items)
        return messages

    def add_message(self, message: BaseMessage) -> None:
        """Append the message to the record in Cloud Spanner"""
        with self.database.batch() as batch:
            batch.insert_or_update(
                table=self.table_name,
                columns=("session_id", "created_at", "message"),
                values=[
                    (
                        self.session_id,
                        spanner.COMMIT_TIMESTAMP,
                        JsonObject(message.dict()),
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
