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
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Union

from google.cloud.spanner import Client, KeySet  # type: ignore
from google.cloud.spanner_admin_database_v1.types import DatabaseDialect  # type: ignore
from google.cloud.spanner_v1.data_types import JsonObject  # type: ignore
from langchain_community.document_loaders.base import BaseLoader
from langchain_core.documents import Document

from .version import __version__

USER_AGENT_LOADER = "langchain-google-spanner-python:document_loader/" + __version__
USER_AGENT_SAVER = "langchain-google-spanner-python:document_saver/" + __version__

OPERATION_TIMEOUT_SECONDS = 240
MUTATION_BATCH_SIZE = 1000

CONTENT_COL_NAME = "page_content"
METADATA_COL_NAME = "langchain_metadata"


@dataclass
class Column:
    name: str
    data_type: str
    nullable: bool = True


def client_with_user_agent(client: Optional[Client], user_agent: str) -> Client:
    if not client:
        client = Client()

    client_agent = client._client_info.user_agent
    if not client_agent:
        client._client_info.user_agent = user_agent
    elif user_agent not in client_agent:
        client._client_info.user_agent = " ".join([client_agent, user_agent])
    return client


def _load_row_to_doc(
    format: str,
    content_columns: List[str],
    metadata_columns: List[str],
    metadata_json_column: str,
    row: Dict,
) -> Document:
    page_content = ""
    if format == "text":
        page_content = " ".join(str(row[c]) for c in content_columns if row[c])
    elif format == "YAML":
        page_content = "\n".join(f"{c}: {str(row[c])}" for c in content_columns)
    elif format == "JSON":
        j = {}
        for c in content_columns:
            j[c] = row[c]
        page_content = json.dumps(j)
    elif format == "CSV":
        page_content = ", ".join(str(row[c]) for c in content_columns if row[c])

    metadata: Dict[str, Any] = {}
    if metadata_json_column in metadata_columns and row.get(metadata_json_column):
        metadata = row[metadata_json_column]
    for c in metadata_columns:
        if c != metadata_json_column:
            metadata[c] = row[c]

    return Document(page_content=page_content, metadata=metadata)


def _load_doc_to_row(
    table_fields: List[str],
    doc: Document,
    content_column: str,
    metadata_json_column: str,
    parse_json: bool = True,
) -> tuple:
    """
    Load document to row.

    Args:
        table_fields: Spanner table fields names.
        doc: Document that is used.
        content_column: Name of the content column.
        metadata_json_column: Name of the special JSON column.
        parse_json: Parse json column to string or leave it as JSON object. String format is needed to for Spanner inserts.
                    JSON object is used to compare with Spanner reads.
    """
    doc_metadata = doc.metadata.copy()
    row = []
    for col in table_fields:
        if (
            col != content_column
            and col != metadata_json_column
            and col in doc_metadata
        ):
            row.append(doc_metadata[col])
            del doc_metadata[col]
        if col == content_column:
            row.append(doc.page_content)

    if metadata_json_column in table_fields:
        metadata_json = {}
        if metadata_json_column in doc_metadata:
            metadata_json = doc_metadata[metadata_json_column]
            del doc_metadata[metadata_json_column]
        metadata_json = {**metadata_json, **doc_metadata}
        j = json.dumps(metadata_json) if parse_json else metadata_json
        row.append(j)  # type: ignore

    return tuple(row)


def _batch(datas: List[Any], size: int = 1) -> Iterator[List[Any]]:
    data_length = len(datas)
    for current in range(0, data_length, size):
        yield datas[current : min(current + size, data_length)]


class SpannerLoader(BaseLoader):
    """Loads data from Google Cloud Spanner."""

    def __init__(
        self,
        instance_id: str,
        database_id: str,
        query: str,
        content_columns: List[str] = [],
        metadata_columns: List[str] = [],
        format: str = "text",
        databoost: bool = False,
        metadata_json_column: str = METADATA_COL_NAME,
        staleness: Union[float, datetime.datetime] = 0.0,
        client: Optional[Client] = None,
    ):
        """Initialize Spanner document loader.

        Args:
            instance_id: The Spanner instance to load data from.
            database_id: The Spanner database to load data from.
            query: A GoogleSQL or PostgreSQL query. Users must match dialect to their database.
            content_columns: The list of column(s) or field(s) to use for a Document's page content.
                              Page content is the default field for embeddings generation.
            metadata_columns: The list of column(s) or field(s) to use for metadata.
            format: Set the format of page content if using multiple columns or fields.
                    Format included: 'text', 'JSON', 'YAML', 'CSV'.
            databoost: Use data boost on read. Note: needs extra IAM permissions and higher cost.
            metadata_json_column: The name of the JSON column to use as the metadata's base dictionary.
            staleness: The time bound for stale read. Takes either a datetime or float.
            client: The connection object to use. This can be used to customize project id and credentials.
        """
        self.instance_id = instance_id
        self.database_id = database_id
        self.query = query
        self.content_columns = content_columns
        self.metadata_columns = metadata_columns
        self.format = format
        self.metadata_json_column = metadata_json_column
        self.databoost = databoost
        self.client = client_with_user_agent(client, USER_AGENT_LOADER)
        self.staleness = staleness

        formats = ["JSON", "text", "YAML", "CSV"]
        if self.format not in formats:
            raise Exception("Use one of 'text', 'JSON', 'YAML', 'CSV'.")

        instance = self.client.instance(instance_id)
        if not instance.exists():
            raise Exception("Instance doesn't exist.")

        database = instance.database(database_id)
        if not database.exists():
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
        instance = self.client.instance(self.instance_id)
        db = instance.database(self.database_id)
        timestamp = (
            self.staleness.replace(tzinfo=datetime.timezone.utc)
            if type(self.staleness) is datetime.datetime
            else None
        )
        duration = (
            datetime.timedelta(seconds=self.staleness)
            if type(self.staleness) is float
            else None
        )
        snapshot = db.batch_snapshot(exact_staleness=duration, read_timestamp=timestamp)
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
                yield _load_row_to_doc(
                    self.format,
                    content_columns,
                    metadata_columns,
                    self.metadata_json_column,
                    row,
                )


class SpannerDocumentSaver:
    """Save docs to Google Cloud Spanner."""

    def __init__(
        self,
        instance_id: str,
        database_id: str,
        table_name: str,
        content_column: str = CONTENT_COL_NAME,
        metadata_columns: List[str] = [],
        metadata_json_column: str = METADATA_COL_NAME,
        primary_key: Optional[str] = None,
        client: Optional[Client] = None,
    ):
        """Initialize Spanner document saver.

        Args:
            instance_id: The Spanner instance to load data to.
            database_id: The Spanner database to load data to.
            table_name: The table name to load data to.
            content_column: The name of the content column. Defaulted to the first column.
            metadata_columns: This is for user to opt-in a selection of columns to use. Defaulted to use
                              all columns.
            metadata_json_column: The name of the special JSON column. Defaulted to use "langchain_metadata".
            client: The connection object to use. This can be used to customized project id and credentials.
        """
        self.instance_id = instance_id
        self.database_id = database_id
        self.table_name = table_name
        self.content_column = content_column
        self.metadata_columns = metadata_columns
        self.metadata_json_column = metadata_json_column
        self.primary_key = primary_key or self.content_column
        self.client = client_with_user_agent(client, USER_AGENT_SAVER)

        instance = self.client.instance(instance_id)
        if not instance.exists():
            raise Exception("Instance doesn't exist.")

        database = instance.database(database_id)
        if not database.exists():
            raise Exception("Database doesn't exist.")

        database.reload()
        self.dialect = database.database_dialect

        table = database.table(table_name)
        if not table.exists():
            raise Exception(
                "Table doesn't exist. Create table with SpannerDocumentSaver.init_document_table function."
            )

        self._table_fields = [self.primary_key]
        for n in table.schema:
            if n.name != metadata_json_column and n.name != self.primary_key:
                self._table_fields.append(n.name)
        self._table_fields.append(metadata_json_column)

    def add_documents(self, documents: List[Document]):
        """Add documents to the Spanner table."""
        db = self.client.instance(self.instance_id).database(self.database_id)
        values = [
            _load_doc_to_row(
                self._table_fields,
                doc,
                self.content_column,
                self.metadata_json_column,
            )
            for doc in documents
        ]

        for values_batch in _batch(values, MUTATION_BATCH_SIZE):
            with db.batch() as batch:
                batch.insert_or_update(
                    table=self.table_name,
                    columns=self._table_fields,
                    values=values_batch,
                )

    def delete(self, documents: List[Document]):
        """Delete documents from the table."""
        database = self.client.instance(self.instance_id).database(self.database_id)
        # load documents to row
        docs = [
            _load_doc_to_row(
                self._table_fields,
                doc,
                self.content_column,
                self.metadata_json_column,
                False,
            )
            for doc in documents
        ]
        keys = [[doc[0]] for doc in docs]
        docs_keys = KeySet(keys=keys)
        snapshot = database.batch_snapshot()
        partitions = snapshot.generate_read_batches(
            table=self.table_name,
            columns=tuple(self._table_fields),
            keyset=docs_keys,
            partition_size_bytes=5000000,
        )

        for partition in partitions:
            keys_to_delete = []
            for row in snapshot.process_read_batch(partition):
                # compare whole document
                if tuple(row) in docs:
                    keys_to_delete.append([row[0]])
            with database.batch() as batch:
                docs_to_delete = KeySet(keys=keys_to_delete)
                batch.delete(self.table_name, docs_to_delete)

    @staticmethod
    def init_document_table(
        instance_id: str,
        database_id: str,
        table_name: str,
        content_column: str = CONTENT_COL_NAME,
        metadata_columns: List[Column] = [],
        primary_key: str = "",
        store_metadata: bool = True,
        metadata_json_column: str = METADATA_COL_NAME,
    ):
        """
        Create a new table to store docs with a custom schema.

        Args:
            instance_id: The Spanner instance to load data to.
            database_id: The Spanner database to load data to.
            table_name: The table name to load data to.
            content_column: The name of the content column.
            metadata_columns: The metadata columns for custom schema.
            primary_key: The name of the primary key.
            store_metadata: If true, extra metadata will be stored in the "langchain_metadata" column.
                            Defaulted to true.
            metadata_json_column: The name of the special JSON column. Defaulted to use "langchain_metadata".
        """
        client = Client()
        primary_key = primary_key or content_column
        metadata_json_column = metadata_json_column if store_metadata else ""

        instance = client.instance(instance_id)
        if not instance.exists():
            raise Exception("Instance doesn't exist.")

        database = instance.database(database_id)
        if not database.exists():
            raise Exception("Database doesn't exist.")

        # create table with custom schema
        SpannerDocumentSaver.create_table(
            client,
            instance_id,
            database_id,
            table_name,
            primary_key,
            metadata_json_column,
            content_column,
            metadata_columns,
        )

    @staticmethod
    def create_table(
        client: Client,
        instance_id: str,
        database_id: str,
        table_name: str,
        primary_key: str,
        metadata_json_column: str,
        content_column: str,
        metadata_columns: List[Column],
    ):
        """
        Create a new table in Spanner database.

        Args:
            client: The connection object to use.
            instance_id: The Spanner instance to load data to.
            database_id: The Spanner database to load data to.
            table_name: The table name to load data to.
            primary_key: The name of the primary key for the table.
            metadata_json_column: The name of the special JSON column.
            content_column: The name of the content column.
            metadata_columns: The metadata columns for custom schema.
        """
        database = client.instance(instance_id).database(database_id)
        database.reload()
        dialect = database.database_dialect

        ddl = f"CREATE TABLE {table_name} ("
        if dialect == DatabaseDialect.POSTGRESQL:
            ddl += f"{content_column} VARCHAR(1024) NOT NULL,"
            for col in metadata_columns:
                null_string = " NOT NULL" if col.nullable else ""
                ddl += f"{col.name} {col.data_type}{null_string},"
            if metadata_json_column:
                ddl += f"{metadata_json_column} JSONb NOT NULL,"
            ddl += f"PRIMARY KEY ({primary_key}));"
        else:
            ddl += f"{content_column} STRING(1024) NOT NULL,"
            for col in metadata_columns:
                null_string = " NOT NULL" if col.nullable else ""
                ddl += f"{col.name} {col.data_type}{null_string},"
            if metadata_json_column:
                ddl += f"{metadata_json_column} JSON NOT NULL,"
            ddl += f") PRIMARY KEY ({primary_key})"

        operation = database.update_ddl([ddl])
        operation.result(OPERATION_TIMEOUT_SECONDS)
