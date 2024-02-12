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
import os

import pytest
from google.cloud.spanner import Client, KeySet  # type: ignore
from langchain_core.documents import Document

from langchain_google_spanner.document_loader import SpannerDocumentSaver, SpannerLoader

project_id = os.environ["PROJECT_ID"]
instance = os.environ["INSTANCE_ID"]
google_database = os.environ["GOOGLE_DATABASE"]
pg_database = os.environ["PG_DATABASE"]
table_name = os.environ["TABLE_NAME"].replace("-", "_")

OPERATION_TIMEOUT_SECONDS = 240


@pytest.fixture(scope="module")
def client() -> Client:
    return Client(project=project_id)


class TestSpannerDocumentLoaderGoogleSQL:
    @pytest.fixture(autouse=True, scope="class")
    def setup_database(self, client):
        database = client.instance(instance).database(google_database)
        operation = database.update_ddl([f"DROP TABLE IF EXISTS {table_name}"])
        operation.result(OPERATION_TIMEOUT_SECONDS)
        SpannerDocumentSaver.init_document_table(
            instance,
            google_database,
            table_name,
            content_column="product_id",
            metadata_columns=[
                ("product_name", "STRING(1024)", True),
                ("description", "STRING(1024)", False),
                ("price", "INT64", False),
            ],
        )

        saver = SpannerDocumentSaver(
            instance,
            google_database,
            table_name,
            client,
            content_column="product_id",
            metadata_columns=["product_name", "description", "price"],
        )
        test_documents = [
            Document(
                page_content="1",
                metadata={
                    "product_name": "cards",
                    "description": "playing cards are cool",
                    "price": 10,
                    "extra_metadata": "foobar",
                    "langchain_metadata": {
                        "foo": "bar",
                    },
                },
            ),
        ]
        saver.add_documents(test_documents)

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
                            "extra_metadata": "foobar",
                            "foo": "bar",
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
    def test_loader_with_query(self, client, query, expected):
        loader = SpannerLoader(instance, google_database, query, client=client)
        docs = loader.load()
        assert docs == expected

    def test_loader_missing_table_and_query(self):
        with pytest.raises(Exception):
            SpannerLoader(instance, google_database)

    # Custom CUJs
    def test_loader_custom_content(self, client):
        query = f"SELECT * FROM {table_name}"
        loader = SpannerLoader(
            instance,
            google_database,
            query,
            client=client,
            content_columns=["description", "price"],
        )
        docs = loader.load()
        assert docs == [
            Document(
                page_content="playing cards are cool 10",
                metadata={
                    "extra_metadata": "foobar",
                    "foo": "bar",
                    "product_id": "1",
                    "product_name": "cards",
                },
            ),
        ]

    def test_loader_custom_metadata(self, client):
        query = f"SELECT * FROM {table_name}"
        loader = SpannerLoader(
            instance,
            google_database,
            query,
            client=client,
            metadata_columns=["product_name", "price"],
        )
        docs = loader.load()
        assert docs == [
            Document(
                page_content="1",
                metadata={"product_name": "cards", "price": 10},
            ),
        ]

    def test_loader_custom_content_and_metadata(self, client):
        query = f"SELECT * FROM {table_name}"
        loader = SpannerLoader(
            instance,
            google_database,
            query,
            client=client,
            content_columns=["product_name"],
            metadata_columns=["product_id", "price"],
        )
        docs = loader.load()
        assert docs == [
            Document(
                page_content="cards",
                metadata={"product_id": "1", "price": 10},
            ),
        ]

    def test_loader_custom_format_json(self, client):
        query = f"SELECT * FROM {table_name}"
        loader = SpannerLoader(
            instance,
            google_database,
            query,
            client=client,
            content_columns=["product_id", "product_name"],
            format="JSON",
        )
        docs = loader.load()
        assert docs == [
            Document(
                page_content="product_id: 1 product_name: cards",
                metadata={
                    "extra_metadata": "foobar",
                    "foo": "bar",
                    "description": "playing cards are cool",
                    "price": 10,
                },
            )
        ]

    def test_loader_custom_format_yaml(self, client):
        query = f"SELECT * FROM {table_name}"
        loader = SpannerLoader(
            instance, google_database, query, client=client, format="YAML"
        )
        docs = loader.load()
        assert docs == [
            Document(
                page_content="product_id: 1",
                metadata={
                    "product_name": "cards",
                    "description": "playing cards are cool",
                    "price": 10,
                    "foo": "bar",
                    "extra_metadata": "foobar",
                },
            )
        ]

    def test_loader_custom_format_csv(self, client):
        query = f"SELECT * FROM {table_name}"
        loader = SpannerLoader(
            instance, google_database, query, client=client, format="CSV"
        )
        docs = loader.load()
        assert docs == [
            Document(
                page_content="1",
                metadata={
                    "product_name": "cards",
                    "description": "playing cards are cool",
                    "price": 10,
                    "foo": "bar",
                    "extra_metadata": "foobar",
                },
            )
        ]

    def test_loader_custom_format_error(self, client):
        query = f"SELECT * FROM {table_name}"
        with pytest.raises(Exception):
            SpannerLoader(
                instance,
                google_database,
                query,
                client,
                format="NOT_A_FORMAT",
            )

    def test_loader_custom_json_metadata(self, client):
        database = client.instance(instance).database(google_database)
        operation = database.update_ddl([f"DROP TABLE IF EXISTS {table_name}"])
        operation.result(OPERATION_TIMEOUT_SECONDS)
        SpannerDocumentSaver.init_document_table(
            instance,
            google_database,
            table_name,
            content_column="product_id",
            metadata_columns=[
                ("product_name", "STRING(1024)", True),
                ("description", "STRING(1024)", False),
                ("price", "INT64", False),
            ],
            metadata_json_column="my_metadata",
        )

        saver = SpannerDocumentSaver(
            instance,
            google_database,
            table_name,
            client,
            content_column="product_id",
            metadata_columns=["product_name", "description", "price"],
            metadata_json_column="my_metadata",
        )
        test_documents = [
            Document(
                page_content="1",
                metadata={
                    "product_name": "cards",
                    "description": "playing cards are cool",
                    "price": 10,
                    "extra_metadata": "foobar",
                    "my_metadata": {
                        "foo": "bar",
                    },
                },
            ),
        ]
        saver.add_documents(test_documents)
        query = f"SELECT * FROM {table_name}"
        loader = SpannerLoader(
            instance,
            google_database,
            query,
            client=client,
            metadata_json_column="my_metadata",
        )
        docs = loader.load()
        assert docs == [
            Document(
                page_content="1",
                metadata={
                    "product_name": "cards",
                    "description": "playing cards are cool",
                    "price": 10,
                    "foo": "bar",
                    "extra_metadata": "foobar",
                },
            ),
        ]


class TestSpannerDocumentLoaderPostgreSQL:
    @pytest.fixture(autouse=True, scope="class")
    def setup_database(self, client):
        database = client.instance(instance).database(pg_database)
        operation = database.update_ddl([f"DROP TABLE IF EXISTS {table_name}"])
        operation.result(OPERATION_TIMEOUT_SECONDS)
        SpannerDocumentSaver.init_document_table(
            instance,
            pg_database,
            table_name,
            content_column="product_id",
            metadata_columns=[
                ("product_name", "VARCHAR(1024)", True),
                ("description", "VARCHAR(1024)", False),
                ("price", "INT", False),
            ],
        )

        saver = SpannerDocumentSaver(
            instance,
            pg_database,
            table_name,
            client,
            content_column="product_id",
            metadata_columns=["product_name", "description", "price"],
        )
        test_documents = [
            Document(
                page_content="1",
                metadata={
                    "product_name": "cards",
                    "description": "playing cards are cool",
                    "price": 10,
                    "extra_metadata": "foobar",
                    "langchain_metadata": {
                        "foo": "bar",
                    },
                },
            ),
        ]
        saver.add_documents(test_documents)

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
                            "extra_metadata": "foobar",
                            "foo": "bar",
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
    def test_loader_with_query(self, client, query, expected):
        loader = SpannerLoader(instance, pg_database, query, client=client)
        docs = loader.load()
        assert docs == expected

    def test_loader_missing_table_and_query(self):
        with pytest.raises(Exception):
            SpannerLoader(instance, pg_database)

    # Custom CUJs
    def test_loader_custom_content(self, client):
        query = f"SELECT * FROM {table_name}"
        loader = SpannerLoader(
            instance,
            pg_database,
            query,
            client=client,
            content_columns=["description", "price"],
        )
        docs = loader.load()
        assert docs == [
            Document(
                page_content="playing cards are cool 10",
                metadata={
                    "extra_metadata": "foobar",
                    "foo": "bar",
                    "product_id": "1",
                    "product_name": "cards",
                },
            ),
        ]

    def test_loader_custom_metadata(self, client):
        query = f"SELECT * FROM {table_name}"
        loader = SpannerLoader(
            instance,
            pg_database,
            query,
            client=client,
            metadata_columns=["product_name", "price"],
        )
        docs = loader.load()
        assert docs == [
            Document(
                page_content="1",
                metadata={"product_name": "cards", "price": 10},
            ),
        ]

    def test_loader_custom_content_and_metadata(self, client):
        query = f"SELECT * FROM {table_name}"
        loader = SpannerLoader(
            instance,
            pg_database,
            query,
            client=client,
            content_columns=["product_name"],
            metadata_columns=["product_id", "price"],
        )
        docs = loader.load()
        assert docs == [
            Document(
                page_content="cards",
                metadata={"product_id": "1", "price": 10},
            ),
        ]

    def test_loader_custom_format_json(self, client):
        query = f"SELECT * FROM {table_name}"
        loader = SpannerLoader(
            instance,
            pg_database,
            query,
            client=client,
            content_columns=["product_id", "product_name"],
            format="JSON",
        )
        docs = loader.load()
        assert docs == [
            Document(
                page_content="product_id: 1 product_name: cards",
                metadata={
                    "extra_metadata": "foobar",
                    "foo": "bar",
                    "description": "playing cards are cool",
                    "price": 10,
                },
            )
        ]

    def test_loader_custom_format_yaml(self, client):
        query = f"SELECT * FROM {table_name}"
        loader = SpannerLoader(
            instance, pg_database, query, client=client, format="YAML"
        )
        docs = loader.load()
        assert docs == [
            Document(
                page_content="product_id: 1",
                metadata={
                    "product_name": "cards",
                    "description": "playing cards are cool",
                    "price": 10,
                    "foo": "bar",
                    "extra_metadata": "foobar",
                },
            )
        ]

    def test_loader_custom_format_csv(self, client):
        query = f"SELECT * FROM {table_name}"
        loader = SpannerLoader(
            instance, pg_database, query, client=client, format="CSV"
        )
        docs = loader.load()
        assert docs == [
            Document(
                page_content="1",
                metadata={
                    "product_name": "cards",
                    "description": "playing cards are cool",
                    "price": 10,
                    "foo": "bar",
                    "extra_metadata": "foobar",
                },
            )
        ]

    def test_loader_custom_format_error(self, client):
        query = f"SELECT * FROM {table_name}"
        with pytest.raises(Exception):
            SpannerLoader(
                instance,
                pg_database,
                query,
                client,
                format="NOT_A_FORMAT",
            )

    def test_loader_custom_json_metadata(self, client):
        database = client.instance(instance).database(pg_database)
        operation = database.update_ddl([f"DROP TABLE IF EXISTS {table_name}"])
        operation.result(OPERATION_TIMEOUT_SECONDS)
        SpannerDocumentSaver.init_document_table(
            instance,
            pg_database,
            table_name,
            content_column="product_id",
            metadata_columns=[
                ("product_name", "VARCHAR(1024)", True),
                ("description", "VARCHAR(1024)", False),
                ("price", "INT", False),
            ],
            metadata_json_column="my_metadata",
        )

        saver = SpannerDocumentSaver(
            instance,
            pg_database,
            table_name,
            client,
            content_column="product_id",
            metadata_columns=["product_name", "description", "price"],
            metadata_json_column="my_metadata",
        )
        test_documents = [
            Document(
                page_content="1",
                metadata={
                    "product_name": "cards",
                    "description": "playing cards are cool",
                    "price": 10,
                    "extra_metadata": "foobar",
                    "my_metadata": {
                        "foo": "bar",
                    },
                },
            ),
        ]
        saver.add_documents(test_documents)
        query = f"SELECT * FROM {table_name}"
        loader = SpannerLoader(
            instance,
            pg_database,
            query,
            client=client,
            metadata_json_column="my_metadata",
        )
        docs = loader.load()
        assert docs == [
            Document(
                page_content="1",
                metadata={
                    "product_name": "cards",
                    "description": "playing cards are cool",
                    "price": 10,
                    "foo": "bar",
                    "extra_metadata": "foobar",
                },
            ),
        ]


class TestSpannerDocumentSaver:
    @pytest.fixture(name="google_client")
    def setup_google_client(self, client) -> Client:
        database = client.instance(instance).database(google_database)
        operation = database.update_ddl([f"DROP TABLE IF EXISTS {table_name}"])
        print("table dropped")
        operation.result(OPERATION_TIMEOUT_SECONDS)
        yield client

    @pytest.fixture(name="pg_client")
    def setup_pg_client(self, client) -> Client:
        database = client.instance(instance).database(pg_database)
        operation = database.update_ddl([f"DROP TABLE IF EXISTS {table_name}"])
        operation.result(OPERATION_TIMEOUT_SECONDS)
        yield client

    def test_saver_google_sql(self, google_client):
        SpannerDocumentSaver.init_document_table(instance, google_database, table_name)
        saver = SpannerDocumentSaver(
            instance, google_database, table_name, google_client
        )
        query = f"SELECT * FROM {table_name}"
        loader = SpannerLoader(
            client=google_client,
            instance=instance,
            database=google_database,
            query=query,
        )
        expected_docs = [
            Document(page_content="Hello, World!", metadata={"source": "my-computer"}),
            Document(page_content="Taylor", metadata={"last_name": "Swift"}),
        ]

        saver.add_documents(expected_docs)
        assert loader.load() == expected_docs
        assert saver._table_fields == ["page_content", "langchain_metadata"]

    def test_saver_pg(self, pg_client):
        SpannerDocumentSaver.init_document_table(instance, pg_database, table_name)
        saver = SpannerDocumentSaver(instance, pg_database, table_name, pg_client)
        query = f"SELECT * FROM {table_name}"
        loader = SpannerLoader(
            client=pg_client, instance=instance, database=pg_database, query=query
        )
        expected_docs = [
            Document(page_content="Hello, World!", metadata={"source": "my-computer"}),
            Document(page_content="Taylor", metadata={"last_name": "Swift"}),
        ]

        saver.add_documents(expected_docs)
        assert loader.load() == expected_docs
        assert saver._table_fields == ["page_content", "langchain_metadata"]

    def test_saver_google_sql_with_custom_schema(self, google_client):
        SpannerDocumentSaver.init_document_table(
            instance,
            google_database,
            table_name,
            content_column="my_page_content",
            metadata_columns=[
                ("category", "STRING(35)", True),
                ("price", "INT64", False),
            ],
            primary_key="my_page_content",
            store_metadata=True,
        )
        saver = SpannerDocumentSaver(
            instance, google_database, table_name, google_client
        )
        query = f"SELECT * FROM {table_name}"
        loader = SpannerLoader(
            client=google_client,
            instance=instance,
            database=google_database,
            query=query,
        )
        expected_docs = [
            Document(
                page_content="card",
                metadata={
                    "category": "games",
                    "price": 5,
                    "description": "these are fun",
                },
            )
        ]
        saver.add_documents(expected_docs)
        assert loader.load() == expected_docs
        assert saver._table_fields == [
            "my_page_content",
            "category",
            "price",
            "langchain_metadata",
        ]

    def test_saver_pg_with_custom_schema(self, pg_client):
        SpannerDocumentSaver.init_document_table(
            instance,
            pg_database,
            table_name,
            content_column="my_page_content",
            metadata_columns=[
                ("category", "VARCHAR(35)", True),
                ("price", "INT", False),
            ],
            primary_key="my_page_content",
            store_metadata=True,
        )
        saver = SpannerDocumentSaver(instance, pg_database, table_name, pg_client)
        query = f"SELECT * FROM {table_name}"
        loader = SpannerLoader(
            client=pg_client, instance=instance, database=pg_database, query=query
        )
        expected_docs = [
            Document(
                page_content="card",
                metadata={
                    "category": "games",
                    "price": 5,
                    "description": "these are fun",
                },
            )
        ]
        saver.add_documents(expected_docs)
        assert loader.load() == expected_docs
        assert saver._table_fields == [
            "my_page_content",
            "category",
            "price",
            "langchain_metadata",
        ]

    def test_delete(self, google_client):
        SpannerDocumentSaver.init_document_table(instance, google_database, table_name)
        saver = SpannerDocumentSaver(
            instance, google_database, table_name, google_client
        )
        query = f"SELECT * FROM {table_name}"
        loader = SpannerLoader(
            client=google_client,
            instance=instance,
            database=google_database,
            query=query,
        )
        expected_docs = [
            Document(page_content="Hello, World!", metadata={"source": "my-computer"}),
            Document(page_content="Taylor", metadata={"last_name": "Swift"}),
        ]

        saver.add_documents(expected_docs)
        assert loader.load() == expected_docs
        # delete one Document
        saver.delete([expected_docs[0]])
        assert loader.load() == [expected_docs[1]]

    def test_saver_with_bad_docs(self, google_client):
        SpannerDocumentSaver.init_document_table(instance, google_database, table_name)
        saver = SpannerDocumentSaver(
            instance, google_database, table_name, google_client
        )
        with pytest.raises(Exception):
            saver.add_documents([1, 2, 3])
