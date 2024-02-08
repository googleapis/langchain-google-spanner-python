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
import os
import json
from google.cloud.spanner import Client, KeySet  # type: ignore
from langchain_core.documents import Document

from langchain_google_spanner.document_loader import SpannerDocumentSaver, SpannerLoader

project_id = os.environ["PROJECT_ID"]
instance=os.environ["INSTANCE_ID"]
google_database=os.environ["GOOGLE_DATABASE"]
pg_database=os.environ["PG_DATABASE"]
table_name=os.environ["TABLE_NAME"]

OPERATION_TIMEOUT_SECONDS = 240

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


@pytest.fixture(name="google_client")
def setup_google_client() -> Client:
    client = Client(project=project_id)
    database = client.instance(instance).database(google_database)
    operation = database.update_ddl([f"DROP table_name IF EXISTS {table_name}"])
    operation.result(OPERATION_TIMEOUT_SECONDS)
    yield client


@pytest.fixture(name="pg_client")
def setup_pg_client() -> Client:
    client = Client(project=project_id)
    database = client.instance(instance).database(pg_database)
    operation = database.update_ddl([f"DROP table_name IF EXISTS {table_name}"])
    operation.result(OPERATION_TIMEOUT_SECONDS)
    yield client


class TestSpannerDocumentSaver:
    def test_saver_google_sql(self, google_client):
        saver = SpannerDocumentSaver(instance, google_database, table_name, google_client)
        query = f"SELECT * FROM {table_name}"
        loader = SpannerLoader(client=google_client, instance=instance, database=google_database, query=query)
        saver.add_documents(expected_docs)
        assert loader.load() == expected_docs

    # def test_saver_google_sql_with_custom_schema():

    def test_saver_pg(self, pg_client):
        saver = SpannerDocumentSaver(instance, pg_database, table_name, pg_client)
        query = f"SELECT * FROM {table_name}"
        loader = SpannerLoader(client=pg_client, instance=instance, database=pg_database, query=query)
        saver.add_documents(expected_docs)
        assert loader.load() == expected_docs

    # def test_saver_pg_with_custom_schema():

    def test_delete(self, google_client):
        saver = SpannerDocumentSaver(instance, google_database, table_name, google_client)
        query = f"SELECT * FROM {table_name}"
        loader = SpannerLoader(client=google_client, instance=instance, database=google_database, query=query)
        saver.add_documents(expected_docs)
        assert loader.load() == expected_docs
        # delete one Document
        saver.delete([expected_docs[0]])
        assert loader.load() == [expected_docs[1]]

    def test_saver_with_bad_docs(self, google_client):
        saver = SpannerDocumentSaver(instance, google_database, table_name, google_client)
        with pytest.raises(Exception):
            saver.add_documents([1, 2, 3])

class TestSpannerDocumentLoader:
    @pytest.fixture(autouse=True)
    def setup_database(self, google_client):
        google_schema = f"""CREATE table_name {table_name} (
                        product_id STRING(1024) NOT NULL,
                        product_name STRING(1024),
                        description STRING(1024),
                        price NUMERIC,
                        ) PRIMARY KEY (product_id)"""
        database = google_client.instance(instance).database(google_database)
        operation = database.update_ddl([google_schema])
        operation.result(OPERATION_TIMEOUT_SECONDS)
        with database.batch() as batch:
            batch.insert(
                table=table_name,
                columns=("product_id", "product_name", "description", "price"),
                values=[
                    ("1", "cards", "playing cards are cool", 10),
                ],
            )

    # Default CUJs
    @pytest.mark.parametrize(
        "query, expected",
        [
            pytest.param(
                f"SELECT * FROM {table_name}",
                [
                    Document(
                        page_content="1",
                        metadata={
                            "product_name": "cards",
                            "description": "playing cards are cool",
                            "price": 10,
                        },
                    )
                ],
            ),
            pytest.param(
                f"SELECT product_name, description FROM {table_name}",
                [
                    Document(
                        page_content="cards",
                        metadata={
                            "description": "playing cards are cool",
                        },
                    )
                ],
            ),
        ],
    )
    def test_loader_with_query(self, google_client, query, expected):
        loader = SpannerLoader(instance, google_database, query, google_client)
        docs = loader.load()
        assert docs == expected

    def test_loader_missing_table_and_query(self):
        with pytest.raises(Exception):
            SpannerLoader(instance, google_database)

    # Custom CUJs
    def test_loader_custom_content(self, google_client):
        query = f"SELECT * FROM {table_name}"
        loader = SpannerLoader(
            instance,
            google_database,
            query,
            google_client,
            content_columns=["description", "price"],
        )
        docs = loader.load()
        assert docs == [
            Document(
                page_content="playing cards are cool 10",
                metadata={"product_id": "1", "product_name": "card"},
            ),
        ]

    def test_loader_custom_metadata(self, google_client):
        query = f"SELECT * FROM {table_name}"
        loader = SpannerLoader(
            instance,
            google_database,
            query,
            google_client,
            metadata_columns=["product_id", "price"],
        )
        docs = loader.load()
        assert docs == [
            Document(
                page_content="1",
                metadata={"product_id": "1", "price": 10},
            ),
        ]

    def test_loader_custom_content_and_metadata(self, google_client):
        query = f"SELECT * FROM {table_name}"
        loader = SpannerLoader(
            instance,
            google_database,
            query,
            google_client,
            content_columns=["product_name"]
            metadata_columns=["product_id", "price"],
        )
        docs = loader.load()
        assert docs == [
            Document(
                page_content="card",
                metadata={"product_id": "1", "price": 10},
            ),
        ]

    def test_loader_custom_json_metadata(self, google_client):
        query = f"SELECT * FROM {table_name}"
        loader = SpannerLoader(
            instance,
            google_database,
            query,
            google_client,
            metadata_json_column="description",
        )
        docs = loader.load()
        assert docs == [
            Document(
                page_content="1",
                metadata={"product_id": "1", "price": 10},
            ),
        ]

    def test_loader_custom_format_json(self, google_client):
        query = f"SELECT * FROM {table_name}"
        loader = SpannerLoader(
            instance, google_database, query, google_client, content_columns=["product_id", "product_name"], format="JSON"
        )
        docs = loader.load()
        assert docs == [
            Document(
                page_content="product_id: 1\nproduct_name: cards",
                metadata={
                    "description": "playing cards are cool",
                    "price": 10,
                },
            )
        ]

    def test_loader_custom_format_yaml(self, google_client):
        query = f"SELECT * FROM {table_name}"
        loader = SpannerLoader(
            instance, google_database, query, google_client, format="YAML"
        )
        docs = loader.load()
        assert docs == [
            Document(
                page_content="product_id: 1",
                metadata={
                    "product_name": "cards",
                    "description": "playing cards are cool",
                    "price": 10,
                },
            )
        ]

    def test_loader_custom_format_csv(self, google_client):
        query = f"SELECT * FROM {table_name}"
        loader = SpannerLoader(
            instance, google_database, query, google_client, format="CSV"
        )
        docs = loader.load()
        assert docs == [
            Document(
                page_content="1",
                metadata={
                    "product_name": "cards",
                    "description": "playing cards are cool",
                    "price": 10,
                },
            )
        ]

    def test_loader_custom_format_error(self, google_client):
        query = f"SELECT * FROM {table_name}"
        with pytest.raises(Exception):
            SpannerLoader(
                instance,
                google_database,
                query,
                google_client,
                format="NOT_A_FORMAT",
            )
