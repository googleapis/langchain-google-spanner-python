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

import datetime
import json
from typing import Dict, Iterator, List, Optional

from google.cloud.spanner import Client, KeySet
from google.cloud.spanner_admin_database_v1.types import DatabaseDialect  # type: ignore
from google.cloud.spanner_v1.data_types import JsonObject  # type: ignore
from langchain_community.document_loaders.base import BaseLoader
from langchain_core.documents import Document

OPERATION_TIMEOUT_SECONDS = 240
MUTATION_BATCH_SIZE = 1000

CONTENT_COL_NAME = "page_content"
METADATA_COL_NAME = "langchain_metadata"

def _load_row_to_doc(format: str, content_columns: List[str], metadata_columns: List[str], row: Dict) -> Document:
    include_headers = ["JSON", "YAML"]
    if format in include_headers:
        page_content = " ".join(
            f"{c}: {str(row[c])}" for c in content_columns if c in row
        )
    else:
        page_content = " ".join(str(row[c]) for c in content_columns if c in row)

    if not page_content:
        raise Exception("column page_content doesn't exist.")

    metadata: Dict[str, Any] = {}
    if METADATA_COL_NAME in metadata_columns and row.get(METADATA_COL_NAME):
        metadata = {**metadata, **row[METADATA_COL_NAME]}

    for c in metadata_columns:
        if c in row and c != METADATA_COL_NAME:
            metadata[c] = row[c]

    return Document(page_content=page_content, metadata=metadata)

def _load_doc_to_row(column_names: List[str], doc: Document, metadata_json_column: Optional[str]) -> Dict:
    doc_metadata = doc.metadata.copy()
    row: Dict[str, Any] = {"page_content": doc.page_content}
    for entry in doc.metadata:
        if entry in column_names:
            row[entry] = doc_metadata[entry]
            del doc_metadata[entry]
    # store extra metadata in metadata_json_column in json format
    if metadata_json_column and metadata_json_column in column_names:
        row[metadata_json_column] = doc_metadata
    elif METADATA_COL_NAME in column_names:
        row[METADATA_COL_NAME] = doc_metadata
    return row

def _batch(datas, size=1):
    data_length = len(datas)
    for current in range(0, data_length, size):
        yield datas[current:min(current+size, data_length)]

class SpannerDocumentSaver:
    """Save docs to Google Cloud Spanner."""

    def __init__(
        self,
        instance: str,
        database: str,
        table_name: str,
        client: Optional[Client] = Client(),
        content_column: Optional[str],
        metadata_columns: List[str] = [],
        metadata_json_column: Optional[str],
    ):
        """Initialize Spanner document saver.

        Args:
            instance: The Spanner instance to load data to.
            database: The Spanner database to load data to.
            table_name: The table name to load data to.
            client: Optional. The connection object to use. This can be used to customized project id and credentials.
            content_columns: Optional. The name of the content column. Defaulted to the first column.
            metadata_columns: Optional. This is for user to opt-in a selection of columns to use. Defaulted to use
                              all columns.
            metadata_json_column: Optional. The name of the special JSON column. Defaulted to use "langchain_metadata".
        """
        self.instance = instance
        self.database = database
        self.table_name = table_name
        self.client = client
        self.content_columns = content_columns
        self.metadata_columns = metadata_columns
        self.metadata_json_column = metadata_json_column
        spanner_instance = self.client.instance(instance)
        if not spanner_instance.exists():
            raise Exception("Instance doesn't exist.")
        spanner_database = spanner_instance.database(database)
        if not spanner_database.exists():
            raise Exception("Database doesn't exist.")
        spanner_database.reload()
        self.dialect = spanner_database.database_dialect
        # Create table if doesn't exist
        if not spanner_instance.database(database).table(table_name).exists():
            self.create_table()

    def add_documents(self, documents: List[Document]):
        """Add documents to the Spanner table."""
        db = self.client.instance(self.instance).database(self.database)
        values = [ _load_doc_to_row(column_names, doc, self.metadata_json_column) for doc in documents]
        # TEST THIS OUT
        columns = (self.content_column or CONTENT_COL_NAME)
        for metadata in self.metadata_columns:
            columns += (metadata,)
        columns += (self.metadata_json_column or METADATA_COL_NAME,)

        for values_batch in _batch(values, MUTATION_BATCH_SIZE):
            with db.batch() as batch:
                batch.insert(
                    table=self.table_name,
                    columns=columns,
                    values=values_batch,
                )

    def delete(self, documents: List[Document]):
        """Delete documents from the table."""
        database = self.client.instance(self.instance).database(self.database)
        keys = [[doc.page_content] for doc in documents]
        for keys_batch in _batch(keys, MUTATION_BATCH_SIZE):
            docs_to_delete = KeySet(keys=keys_batch)
            with database.batch() as batch:
                # Delete based on comparing the whole document instead of just the key
                batch.delete(self.table_name, docs_to_delete)

    @classmethod
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
        database = self.client.instance(self.instance).database(self.database)
        operation = database.update_ddl([ddl])
        operation.result(OPERATION_TIMEOUT_SECONDS)


class SpannerLoader(BaseLoader):
    """Loads data from Google CLoud Spanner."""

    def __init__(
        self,
        instance: str,
        database: str,
        query: str,
        client: Optional[Client] = Client(),
        staleness: Optional[int] = 0,
        content_columns: List[str] = [],
        metadata_columns: List[str] = [],
        format: str = "text",
        databoost: bool = False,
    ):
        """Initialize Spanner document loader.

        Args:
            instance: The Spanner instance to load data from.
            database: The Spanner database to load data from.
            query: A GoogleSQL or PostgreSQL query. Users must match dialect to their database.
            client: Optional. The connection object to use. This can be used to customize project id and credentials.
            staleness: Optional. The time bound for stale read.
            content_columns: The list of column(s) or field(s) to use for a Document's page content.
                              Page content is the default field for embeddings generation.
            metadata_columns: The list of column(s) or field(s) to use for metadata.
            format: Set the format of page content if using multiple columns or fields.
                    Format included: 'text', 'JSON', 'YAML', 'CSV'.
            databoost: Use data boost on read. Note: needs extra IAM permissions and higher cost.
        """
        self.instance = instance
        self.database = database
        self.query = query
        self.client = client
        self.staleness = staleness
        self.content_columns = content_columns
        self.metadata_columns = metadata_columns
        self.format = format
        formats = ["JSON", "text", "YAML", "CSV"]
        if self.format not in formats:
            raise Exception("Use on of 'text', 'JSON', 'YAML', 'CSV'")
        self.databoost = databoost
        if not self.client.instance(self.instance).exists():
            raise Exception("Instance doesn't exist.")
        if not self.client.instance(self.instance).database(self.database).exists():
            raise Exception("Database doesn't exist.")

    def load(self) -> List[Document]:
        """
        Load langchain documents from a Spanner database.

        Returns:
            (List[langchain_core.documents.Document]): a list of Documents with metadata
            from specific columns.
        """
        return list(self.lazy_load())

    def lazy_load(self) -> Iterator[Document]:
        """
        A lazy loader for langchain documents from a Spanner database. Use lazy load to avoid
        caching all documents in memory at once.

        Returns:
            (Iterator[langchain_core.documents.Document]): a list of Documents with metadata
            from specific columns.
        """
        instance = self.client.instance(self.instance)
        db = instance.database(self.database)
        duration = datetime.timedelta(seconds=self.staleness)
        snapshot = db.batch_snapshot(exact_staleness=duration)
        keyset = KeySet(all_=True)
        partitions = snapshot.generate_query_batches(
            sql=self.query, data_boost_enabled=self.databoost
        )
        for partition in partitions:
            r = snapshot.process_query_batch(partition)
            results = r.to_dict_list()
            if len(results) == 0:
                break
            column_names = [f.name for f in r.fields]
            content_columns = self.content_columns or [column_names[0]]
            metadata_columns = self.metadata_columns or [
                col for col in column_names if col not in content_columns
            ]
            for row in results:
                yield _load_row_to_doc(self.format, content_columns, metadata_columns, row)
