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

import json
from typing import List, Optional

from google.cloud.spanner import Client, KeySet  # type: ignore
from google.cloud.spanner_admin_database_v1.types import DatabaseDialect  # type: ignore
from google.cloud.spanner_v1.data_types import JsonObject  # type: ignore
from langchain_community.document_loaders.base import BaseLoader
from langchain_core.documents import Document

OPERATION_TIMEOUT_SECONDS = 240

CONTENT_COL_NAME = "PageContent"
METADATA_COL_NAME = "LangchainMetadata"


class SpannerDocumentSaver:
    def __init__(
        self,
        instance_name: str,
        database_name: str,
        table_name: str,
        client: Optional[Client] = None,
    ):
        self.instance_name = instance_name
        self.database_name = database_name
        self.table_name = table_name
        self.client = client if client else Client()
        instance = self.client.instance(instance_name)
        if not instance.exists():
            raise Exception("Instance doesn't exist.")

        database = instance.database(database_name)
        if not database.exists():
            raise Exception("Database doesn't exist.")
        database.reload()
        self.dialect = database.database_dialect
        # Create table if doesn't exist
        if not instance.database(database_name).table(table_name).exists():
            self.create_table()

    def add_documents(self, documents: List[Document]):
        db = self.client.instance(self.instance_name).database(self.database_name)
        if self.dialect == DatabaseDialect.POSTGRESQL:
            values = [(doc.page_content, json.dumps(doc.metadata)) for doc in documents]
        else:
            values = [(doc.page_content, JsonObject(doc.metadata)) for doc in documents]
        with db.batch() as batch:
            batch.insert(
                table=self.table_name,
                columns=(CONTENT_COL_NAME, METADATA_COL_NAME),
                values=values,
            )
        # return

    def delete(self, documents: List[Document]):
        database = self.client.instance(self.instance_name).database(self.database_name)
        keys = [doc.page_content for doc in documents]
        docs_to_delete = KeySet(keys=keys)
        with database.batch() as batch:
            batch.delete(self.table_name, docs_to_delete)
        # def delete_documents(transaction):
        #     row_ct = transaction.execute_update(
        #         f"DELETE FROM {self.table_name} WHERE FirstName = 'Alice'"
        #     )

        #     print("{} record(s) deleted.".format(row_ct))

        # database.run_in_transaction(delete_documents)
        return

    @staticmethod
    def init_document_table(
        cls,
        instance_name: str,
        database_name: str,
        table_name: str,
        content_column: List[str] = [CONTENT_COL_NAME],
        metadata_columns: List[str] = [METADATA_COL_NAME],
        store_metadata: bool = True,
    ):
        saver = cls(instance_name, database_name, table_name)
        saver.create_table(content_column, metadata_columns)
        return

    def create_table(
        self,
        content_columns: Optional[List[str]] = None,
        metadata_columns: Optional[List[str]] = None,
    ):
        google_schema = f"""CREATE TABLE {self.table_name} (
                        {CONTENT_COL_NAME}  STRING(1024) NOT NULL,
                        {METADATA_COL_NAME} JSON NOT NULL,
                        ) PRIMARY KEY ({CONTENT_COL_NAME})"""

        pg_schema = f"""CREATE TABLE {self.table_name} (
                        {CONTENT_COL_NAME}  VARCHAR(1024) NOT NULL,
                        {METADATA_COL_NAME} JSONb NOT NULL,
                        PRIMARY KEY ({CONTENT_COL_NAME})
                        );"""

        ddl = pg_schema if self.dialect == DatabaseDialect.POSTGRESQL else google_schema
        database = self.client.instance(self.instance_name).database(self.database_name)
        operation = database.update_ddl([ddl])
        operation.result(OPERATION_TIMEOUT_SECONDS)
        return operation


class SpannerLoader(BaseLoader):
    """Loads data from Google Cloud Spanner."""

    def __init__(
        self,
        instance_name: str,
        database_name: str,
        table_name: Optional[str] = None,
        query: Optional[str] = None,
        content_columns: List[str] = [CONTENT_COL_NAME],
        metadata_columns: List[str] = [METADATA_COL_NAME],
        format: str = "text",
        read_only: bool = True,
        client: Client = Client(),
    ):
        self.instance_name = instance_name
        self.database_name = database_name
        self.table_name = table_name
        self.query = query
        if (table_name and query) or (not table_name and not query):
            raise Exception("Use one of 'table_name' or 'query'.")
        self.content_columns = content_columns
        self.metadata_columns = metadata_columns
        self.format = format
        formats = ["JSON", "text", "YAML", "CSV"]
        if self.format not in formats:
            raise Exception("Use on of 'text', 'JSON', 'YAML, 'CSV'")
        self.read_only = read_only
        self.client = client
        if not self.client.instance(self.instance_name).exists():
            raise Exception("Instance doesn't exist.")
        if (
            not self.client.instance(self.instance_name)
            .database(self.database_name)
            .exists()
        ):
            raise Exception("Database doesn't exist.")

    def load(self) -> List[Document]:
        """Load documents."""
        instance = self.client.instance(self.instance_name)
        db = instance.database(self.database_name)
        if self.table_name:
            with db.snapshot() as snapshot:
                keyset = KeySet(all_=True)
                results = snapshot.read(
                    table=self.table_name,
                    columns=tuple(self.content_columns + self.metadata_columns),
                    keyset=keyset,
                )
                formatted_results = [
                    Document(page_content=row[0], metadata=row[1]) for row in results
                ]
                return formatted_results
        else:
            with db.snapshot() as snapshot:
                keyset = KeySet(all_=True)
                results = snapshot.execute_sql(self.query)
                formatted_results = [
                    Document(page_content=row[0], metadata=row[1]) for row in results
                ]
                return formatted_results
            # return list(self.lazy_load())

    # def lazy_load(self) -> List[Document]:
    #     """A lazy loader for Documents."""
    #     docs = []
    #     # self.client.
    #     return docs
