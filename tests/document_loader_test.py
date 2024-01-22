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

import pytest
from google.cloud.spanner import Client, KeySet  # type: ignore
from langchain_core.documents import Document

from langchain_google_spanner.document_loader import (
    SpannerDocumentSaver,
    SpannerLoader,
)

INSTANCE = "my-instance"
GOOGLE_DATABASE = "my-google-database"
PG_DATABASE = "my-database"
TABLE = "my_table"
expected_docs = [
    Document(page_content="Hello, World!", metadata={"source": "my-computer"}),
    Document(page_content="Taylor", metadata={"last_name": "Swift"}),
]
json_format_metadata = [
    Document(page_content="{'source': 'my-computer'}"),
    Document(page_content="{'last_name': 'Swift'}"),
]
yaml_format = [
    Document(page_content="source: my-computer"),
    Document(page_content="last_name: Swift"),
]
csv_format = [
    Document(page_content="Hello, World!,my-computer"),
    Document(page_content="Taylor, Swift"),
]


class TestSpannerDocumentSaver:
    def test_saver_google_sql(self):
        database = Client().instance(INSTANCE).database(GOOGLE_DATABASE)
        operation = database.update_ddl([f"DROP TABLE IF EXISTS {TABLE}"])
        # operation.result(OPERATION_TIMEOUT_SECONDS)
        # with database.batch() as batch:
        #     batch.delete(TABLE, KeySet(all_=True))

        saver = SpannerDocumentSaver(INSTANCE, GOOGLE_DATABASE, TABLE)
        # saver.add_documents(expected_docs)
        # saver.add_documents([])
        saver.delete(expected_docs)
        # assert

    def test_saver_pg(self):
        database = Client().instance(INSTANCE).database(PG_DATABASE)
        operation = database.update_ddl([f"DROP TABLE IF EXISTS {TABLE}"])
        # with database.batch() as batch:
        #     batch.delete(TABLE, KeySet(all_=True))

        saver = SpannerDocumentSaver(INSTANCE, PG_DATABASE, TABLE)
        saver.add_documents(expected_docs)
        saver.add_documents([])
        saver.delete(expected_docs)
        # assert

    def test_saver_with_bad_docs(self):
        saver = SpannerDocumentSaver(INSTANCE, GOOGLE_DATABASE, TABLE)
        with pytest.raises(Exception):
            saver.add_documents([1, 2, 3])


class TestSpannerDocumentLoader:
    # Default CUJs
    def test_loader_with_table(self):
        loader = SpannerLoader(INSTANCE, GOOGLE_DATABASE, table_name=TABLE)
        result = loader.load()

        assert result == expected_docs

    def test_loader_with_query(self):
        query = f"SELECT PageContent, LangchainMetadata FROM {TABLE};"
        loader = SpannerLoader(INSTANCE, GOOGLE_DATABASE, query=query)
        result = loader.load()

        assert result == expected_docs

    def test_loader_missing_table_and_query(self):
        with pytest.raises(Exception):
            SpannerLoader(INSTANCE, GOOGLE_DATABASE)

    # Custom CUJs
    def test_loader_custom_content(self):
        loader = SpannerLoader(INSTANCE, GOOGLE_DATABASE, table_name=TABLE)
        result = loader.load()
        formatted_expected_docs = []
        assert result == formatted_expected_docs

    def test_loader_custom_metadata(self):
        loader = SpannerLoader(INSTANCE, GOOGLE_DATABASE, table_name=TABLE)
        result = loader.load()
        formatted_expected_docs = []
        assert result == formatted_expected_docs

    def test_loader_custom_format_json(self):
        loader = SpannerLoader(
            INSTANCE, GOOGLE_DATABASE, table_name=TABLE, format="JSON"
        )
        result = loader.load()
        formatted_expected_docs = []
        assert result == formatted_expected_docs

    def test_loader_custom_format_yaml(self):
        loader = SpannerLoader(
            INSTANCE, GOOGLE_DATABASE, table_name=TABLE, format="YAML"
        )
        result = loader.load()
        formatted_expected_docs = []
        assert result == formatted_expected_docs

    def test_loader_custom_format_csv(self):
        loader = SpannerLoader(
            INSTANCE, GOOGLE_DATABASE, table_name=TABLE, format="CSV"
        )
        result = loader.load()
        formatted_expected_docs = []
        assert result == formatted_expected_docs

    def test_loader_custom_format_error(self):
        with pytest.raises(Exception):
            SpannerLoader(
                INSTANCE,
                GOOGLE_DATABASE,
                table_name=TABLE,
                format="NOT_A_FORMAT",
            )

    def test_custom_client(self):
        client = Client()
        loader = SpannerLoader(
            INSTANCE, GOOGLE_DATABASE, table_name=TABLE, client=client
        )
        result = loader.load()

        assert result == expected_docs
