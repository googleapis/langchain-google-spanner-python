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


def _load_row_to_doc(
    format: str,
    content_columns: List[str],
    metadata_columns: List[str],
    metadata_json_column: str,
    row: Dict,
) -> Document:
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
    if metadata_json_column in metadata_columns and row.get(metadata_json_column):
        metadata = {**metadata, **row[metadata_json_column]}

    for c in metadata_columns:
        if c in row and c != metadata_json_column:
            metadata[c] = row[c]

    return Document(page_content=page_content, metadata=metadata)


def _load_doc_to_row(table_fields: List[str], doc: Document, metadata_json_column: str):
    doc_metadata = doc.metadata.copy()
    row = (doc.page_content,)
    # store metadata
    for col in table_fields:
        if (
            col != CONTENT_COL_NAME
            and col != metadata_json_column
            and col in doc_metadata
        ):
            row += (doc_metadata[col],)
            del doc_metadata[col]
    if metadata_json_column in table_fields:
        metadata_json = {}
        print(f"metadata json column is {metadata_json_column}")
        if metadata_json_column in doc_metadata:
            metadata_json = doc_metadata[metadata_json_column]
            del doc_metadata[metadata_json_column]
        metadata_json = {**metadata_json, **doc_metadata}
        row += (json.dumps(metadata_json),)
    return row


def _batch(datas: List[Any], size: int = 1) -> Iterator[List[Any]]:
    data_length = len(datas)
    for current in range(0, data_length, size):
        yield datas[current : min(current + size, data_length)]


class SpannerLoader(BaseLoader):
    """Loads data from Google CLoud Spanner."""

    def __init__(
        self,
        instance: str,
        database: str,
        query: str,
        content_columns: List[str] = [],
        metadata_columns: List[str] = [],
        format: str = "text",
        databoost: bool = False,
        metadata_json_column: str = "",
        client: Optional[Client] = Client(),
        staleness: Optional[int] = 0,
    ):
        """Initialize Spanner document loader.

        Args:
            instance: The Spanner instance to load data from.
            database: The Spanner database to load data from.
            query: A GoogleSQL or PostgreSQL query. Users must match dialect to their database.
            content_columns: The list of column(s) or field(s) to use for a Document's page content.
                              Page content is the default field for embeddings generation.
            metadata_columns: The list of column(s) or field(s) to use for metadata.
            format: Set the format of page content if using multiple columns or fields.
                    Format included: 'text', 'JSON', 'YAML', 'CSV'.
            databoost: Use data boost on read. Note: needs extra IAM permissions and higher cost.
            metadata_json_column: The name of the JSON column to use as the metadata's base dictionary.
            client: Optional. The connection object to use. This can be used to customize project id and credentials.
            staleness: Optional. The time bound for stale read.
        """
        self.instance = instance
        self.database = database
        self.query = query
        self.content_columns = content_columns
        self.metadata_columns = metadata_columns
        self.format = format
        self.metadata_json_column = metadata_json_column
        formats = ["JSON", "text", "YAML", "CSV"]
        if self.format not in formats:
            raise Exception("Use on of 'text', 'JSON', 'YAML', 'CSV'")
        self.databoost = databoost
        self.client = client
        self.staleness = staleness
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
            metadata_json_column = self.metadata_json_column or METADATA_COL_NAME
            for row in results:
                yield _load_row_to_doc(
                    self.format,
                    content_columns,
                    metadata_columns,
                    metadata_json_column,
                    row,
                )


class SpannerDocumentSaver:
    """Save docs to Google Cloud Spanner."""

    def __init__(
        self,
        instance: str,
        database: str,
        table_name: str,
        client: Client = Client(),
        content_column: Optional[str] = "",
        metadata_columns: Optional[List[str]] = [],
        metadata_json_column: Optional[str] = "",
    ):
        """Initialize Spanner document saver.

        Args:
            instance: The Spanner instance to load data to.
            database: The Spanner database to load data to.
            table_name: The table name to load data to.
            client: The connection object to use. This can be used to customized project id and credentials.
            content_column: Optional. The name of the content column. Defaulted to the first column.
            metadata_columns: Optional. This is for user to opt-in a selection of columns to use. Defaulted to use
                              all columns.
            store_metadata: If true, extra metadata will be stored in the "langchain_metadata" column.
            metadata_json_column: Optional. The name of the special JSON column. Defaulted to use "langchain_metadata".
        """
        self.instance = instance
        self.database = database
        self.table_name = table_name
        self.content_column = content_column
        self.metadata_columns = metadata_columns
        self.metadata_json_column = metadata_json_column or METADATA_COL_NAME
        self.client = client
        spanner_instance = self.client.instance(instance)
        if not spanner_instance.exists():
            raise Exception("Instance doesn't exist.")
        spanner_database = spanner_instance.database(database)
        if not spanner_database.exists():
            raise Exception("Database doesn't exist.")
        spanner_database.reload()
        self.dialect = spanner_database.database_dialect
        spanner_table = spanner_database.table(table_name)
        if not spanner_table.exists():
            raise Exception(
                "Table doesn't exist. Create table with SpannerDocumentSaver.init_document_table function."
            )
        self._table_fields = [
            n.name for n in spanner_table.schema if n.name != self.metadata_json_column
        ]
        self._table_fields.append(self.metadata_json_column)

    def add_documents(self, documents: List[Document]):
        """Add documents to the Spanner table."""
        db = self.client.instance(self.instance).database(self.database)
        values = [
            _load_doc_to_row(self._table_fields, doc, self.metadata_json_column)
            for doc in documents
        ]

        for values_batch in _batch(values, MUTATION_BATCH_SIZE):
            with db.batch() as batch:
                batch.insert(
                    table=self.table_name,
                    columns=self._table_fields,
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

    @staticmethod
    def init_document_table(
        instance: str,
        database: str,
        table_name: str,
        content_column: str = "",
        metadata_columns: List[Any] = [],
        primary_key: str = "",
        store_metadata: bool = True,
        metadata_json_column: str = "",
    ):
        """
        Create a new table to store docs with a custom schema.

        Args:
            instance_name: The Spanner instance to load data to.
            database_name: The Spanner database to load data to.
            table_name: The table name to load data to.
            content_column: The name of the content column.
            metadata_columns: The metadata columns for custom schema.
            primary_key: The name of the primary key.
            store_metadata: If true, extra metadata will be stored in the "langchain_metadata" column.
                            Defaulted to true.
        """
        content_column = content_column or CONTENT_COL_NAME
        primary_key = primary_key or content_column
        metadata_json_column = (
            (metadata_json_column or METADATA_COL_NAME) if store_metadata else None
        )
        client = Client()
        spanner_instance = client.instance(instance)
        if not spanner_instance.exists():
            raise Exception("Instance doesn't exist.")
        spanner_database = spanner_instance.database(database)
        if not spanner_database.exists():
            raise Exception("Database doesn't exist.")
        # create table with custom schema
        SpannerDocumentSaver.create_table(
            client,
            instance,
            database,
            table_name,
            primary_key,
            metadata_json_column,
            content_column,
            metadata_columns,
        )

    @staticmethod
    def create_table(
        client: Client,
        instance: str,
        database: str,
        table_name: str,
        primary_key: str,
        metadata_json_column: Optional[str],
        content_column: str,
        metadata_columns: List[Any],
    ):
        """Create a new table in Spanner database."""
        database = client.instance(instance).database(database)
        database.reload()
        dialect = database.database_dialect

        ddl = f"CREATE TABLE {table_name} ("
        if dialect == DatabaseDialect.POSTGRESQL:
            ddl += f"{content_column} VARCHAR(1024) NOT NULL,"
            for col in metadata_columns:
                null_string = "NOT NULL" if col[2] else ""
                ddl += f"{col[0]} {col[1]} {null_string},"
            if metadata_json_column:
                ddl += f"{metadata_json_column} JSONb NOT NULL,"
            ddl += f"PRIMARY KEY ({primary_key}));"
        else:
            ddl += f"{content_column} STRING(1024) NOT NULL,"
            for col in metadata_columns:
                null_string = "NOT NULL" if col[2] else ""
                ddl += f"{col[0]} {col[1]} {null_string},"
            if metadata_json_column:
                ddl += f"{metadata_json_column} JSON NOT NULL,"
            ddl += f") PRIMARY KEY ({primary_key})"

        operation = database.update_ddl([ddl])
        operation.result(OPERATION_TIMEOUT_SECONDS)
