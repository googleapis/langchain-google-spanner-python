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
import os
import uuid
from typing import Dict

import pytest
from google.cloud.spanner import Client  # type: ignore
from langchain_community.document_loaders import HNLoader
from langchain_community.embeddings import FakeEmbeddings

from langchain_google_spanner.vector_store import (  # type: ignore
    DistanceStrategy,
    QueryParameters,
    SpannerVectorStore,
    TableColumn,
    VectorSearchIndex,
)

project_id = os.environ["PROJECT_ID"]
instance_id = os.environ["INSTANCE_ID"]
google_database = os.environ["GOOGLE_DATABASE"]
pg_database = os.environ["PG_DATABASE"]
table_name = "test_table" + str(uuid.uuid4()).replace("-", "_")
# Cloud Spanner takes 30+ minutes to create a Vector search index
# hence in order to make integration tests usable, before
# they fix the bad delay, let's reuse the same database and never DROP
# the database nor table to allow for effective reuse.
ann_db = os.environ.get("GOOGLE_SPANNER_ANN_DB", "my-spanner-db-ann")
table_name_ANN = "our_table_ann"


OPERATION_TIMEOUT_SECONDS = 240


@pytest.fixture(scope="module")
def client() -> Client:
    return Client(project=project_id)


@pytest.fixture()
def cleanupGSQL(client):
    yield

    print("\nPerforming GSQL cleanup after each test...")

    database = client.instance(instance_id).database(google_database)
    operation = database.update_ddl(
        [
            f"DROP TABLE IF EXISTS {table_name}",
        ]
    )
    operation.result(OPERATION_TIMEOUT_SECONDS)

    # Code to perform teardown after each test goes here
    print("\nGSQL Cleanup complete.")


@pytest.fixture()
def cleanupPGSQL(client):
    yield

    print("\nPerforming PGSQL cleanup after each test...")

    database = client.instance(instance_id).database(pg_database)
    operation = database.update_ddl([f"DROP TABLE IF EXISTS {table_name}"])
    operation.result(OPERATION_TIMEOUT_SECONDS)

    # Code to perform teardown after each test goes here
    print("\n PGSQL Cleanup complete.")


class TestStaticUtilityGoogleSQL_KNN:
    @pytest.fixture(autouse=True)
    def setup_database(self, client, cleanupGSQL):
        yield

    def test_init_vector_store_table1(self):
        SpannerVectorStore.init_vector_store_table(
            instance_id=instance_id,
            database_id=google_database,
            table_name=table_name,
            metadata_columns=[
                TableColumn(name="product_name", type="STRING(1024)", is_null=False),
                TableColumn(name="title", type="STRING(1024)"),
                TableColumn(name="price", type="INT64"),
            ],
        )

    def test_init_vector_store_table2(self):
        SpannerVectorStore.init_vector_store_table(
            instance_id=instance_id,
            database_id=google_database,
            table_name=table_name,
            id_column="custom_id1",
            content_column="custom_content_id1",
            embedding_column="custom_embedding_id1",
            metadata_columns=[
                TableColumn(name="product_name", type="STRING(1024)", is_null=False),
                TableColumn(name="title", type="STRING(1024)"),
                TableColumn(name="price", type="INT64"),
            ],
        )

    def test_init_vector_store_table3(self):
        SpannerVectorStore.init_vector_store_table(
            instance_id=instance_id,
            database_id=google_database,
            table_name=table_name,
            id_column=TableColumn(
                name="product_id", type="STRING(1024)", is_null=False
            ),
            embedding_column=TableColumn(
                name="custom_embedding_id1", type="ARRAY<FLOAT64>", is_null=True
            ),
            metadata_columns=[
                TableColumn(name="product_name", type="STRING(1024)", is_null=False),
                TableColumn(name="title", type="STRING(1024)"),
                TableColumn(name="metadata_json_column", type="JSON"),
            ],
        )

    def test_init_vector_store_table4(self):
        SpannerVectorStore.init_vector_store_table(
            instance_id=instance_id,
            database_id=google_database,
            table_name=table_name,
            id_column=TableColumn(
                name="product_id", type="STRING(1024)", is_null=False
            ),
            embedding_column=TableColumn(
                name="custom_embedding_id1", type="ARRAY<FLOAT64>", is_null=True
            ),
            metadata_columns=[
                TableColumn(name="product_name", type="STRING(1024)", is_null=False),
                TableColumn(name="title", type="STRING(1024)"),
                TableColumn(name="metadata_json_column", type="JSON"),
            ],
            primary_key="product_name, title, product_id",
        )


class TestStaticUtilityPGSQL:
    @pytest.fixture(autouse=True)
    def setup_database(self, client, cleanupPGSQL):
        yield

    def test_init_vector_store_table1(self):
        SpannerVectorStore.init_vector_store_table(
            instance_id=instance_id,
            database_id=pg_database,
            table_name=table_name,
            metadata_columns=[
                TableColumn(name="product_name", type="TEXT", is_null=False),
                TableColumn(name="title", type="varchar(36)"),
                TableColumn(name="price", type="bigint"),
            ],
        )

    def test_init_vector_store_table2(self):
        SpannerVectorStore.init_vector_store_table(
            instance_id=instance_id,
            database_id=pg_database,
            table_name=table_name,
            id_column="custom_id1",
            content_column="custom_content_id1",
            embedding_column="custom_embedding_id1",
            metadata_columns=[
                TableColumn(name="product_name", type="TEXT", is_null=False),
                TableColumn(name="title", type="varchar(36)"),
                TableColumn(name="price", type="bigint"),
            ],
        )

    def test_init_vector_store_table3(self):
        SpannerVectorStore.init_vector_store_table(
            instance_id=instance_id,
            database_id=pg_database,
            table_name=table_name,
            id_column=TableColumn(name="product_id", type="varchar(36)", is_null=False),
            embedding_column=TableColumn(
                name="custom_embedding_id1", type="float8[]", is_null=True
            ),
            metadata_columns=[
                TableColumn(name="product_name", type="TEXT", is_null=False),
                TableColumn(name="title", type="varchar(36)"),
                TableColumn(name="price", type="bigint"),
            ],
        )

    def test_init_vector_store_table4(self):
        SpannerVectorStore.init_vector_store_table(
            instance_id=instance_id,
            database_id=pg_database,
            table_name=table_name,
            id_column=TableColumn(name="product_id", type="varchar(36)", is_null=False),
            embedding_column=TableColumn(
                name="custom_embedding_id1", type="float8[]", is_null=True
            ),
            metadata_columns=[
                TableColumn(name="product_name", type="TEXT", is_null=False),
                TableColumn(name="title", type="varchar(36)"),
                TableColumn(name="price", type="bigint"),
            ],
            primary_key="product_name, title, product_id",
        )


class TestSpannerVectorStoreGoogleSQL_KNN:
    @pytest.fixture(scope="class")
    def setup_database(self, client):
        SpannerVectorStore.init_vector_store_table(
            instance_id=instance_id,
            database_id=google_database,
            table_name=table_name,
            id_column="row_id",
            metadata_columns=[
                TableColumn(name="metadata", type="JSON", is_null=True),
                TableColumn(name="title", type="STRING(MAX)", is_null=False),
            ],
        )

        loader = HNLoader("https://news.ycombinator.com/item?id=34817881")

        embeddings = FakeEmbeddings(size=3)

        yield loader, embeddings

        print("\nPerforming GSQL cleanup after each test...")

        database = client.instance(instance_id).database(google_database)
        operation = database.update_ddl([f"DROP TABLE IF EXISTS {table_name}"])
        operation.result(OPERATION_TIMEOUT_SECONDS)

        # Code to perform teardown after each test goes here
        print("\nGSQL Cleanup complete.")

    def test_spanner_vector_add_data1(self, setup_database):
        loader, embeddings = setup_database

        db = SpannerVectorStore(
            instance_id=instance_id,
            database_id=google_database,
            table_name=table_name,
            id_column="row_id",
            ignore_metadata_columns=[],
            embedding_service=embeddings,
            metadata_json_column="metadata",
        )

        docs = loader.load()
        ids = [str(uuid.uuid4()) for _ in range(len(docs))]
        ids_row_inserted = db.add_documents(documents=docs, ids=ids)
        assert ids == ids_row_inserted

    def test_spanner_vector_add_data2(self, setup_database):
        loader, embeddings = setup_database

        db = SpannerVectorStore(
            instance_id=instance_id,
            database_id=google_database,
            table_name=table_name,
            id_column="row_id",
            ignore_metadata_columns=[],
            embedding_service=embeddings,
            metadata_json_column="metadata",
        )

        texts = [
            "Langchain Test Text 1",
            "Langchain Test Text 2",
            "Langchain Test Text 3",
        ]
        ids = [str(uuid.uuid4()) for _ in range(len(texts))]
        ids_row_inserted = db.add_texts(
            texts=texts,
            ids=ids,
            metadatas=[
                {"title": "Title 1"},
                {"title": "Title 2"},
                {"title": "Title 3"},
            ],
        )
        assert ids == ids_row_inserted

    def test_spanner_vector_delete_data(self, setup_database):
        loader, embeddings = setup_database

        db = SpannerVectorStore(
            instance_id=instance_id,
            database_id=google_database,
            table_name=table_name,
            id_column="row_id",
            ignore_metadata_columns=[],
            embedding_service=embeddings,
            metadata_json_column="metadata",
        )

        docs = loader.load()

        deleted = db.delete(documents=[docs[0], docs[1]])

        assert deleted

    def test_spanner_vector_search_data1(self, setup_database):
        loader, embeddings = setup_database

        db = SpannerVectorStore(
            instance_id=instance_id,
            database_id=google_database,
            table_name=table_name,
            id_column="row_id",
            ignore_metadata_columns=[],
            embedding_service=embeddings,
            metadata_json_column="metadata",
        )

        docs = db.similarity_search(
            "Testing the langchain integration with spanner", k=2
        )

        assert len(docs) == 2

    def test_spanner_vector_search_data2(self, setup_database):
        loader, embeddings = setup_database

        db = SpannerVectorStore(
            instance_id=instance_id,
            database_id=google_database,
            table_name=table_name,
            id_column="row_id",
            ignore_metadata_columns=[],
            embedding_service=embeddings,
            metadata_json_column="metadata",
        )

        embeds = embeddings.embed_query(
            "Testing the langchain integration with spanner"
        )

        docs = db.similarity_search_by_vector(embeds, k=3, pre_filter="1 = 1")

        assert len(docs) == 3

    def test_spanner_vector_search_data3(self, setup_database):
        loader, embeddings = setup_database

        db = SpannerVectorStore(
            instance_id=instance_id,
            database_id=google_database,
            table_name=table_name,
            id_column="row_id",
            ignore_metadata_columns=[],
            embedding_service=embeddings,
            metadata_json_column="metadata",
            query_parameters=QueryParameters(
                distance_strategy=DistanceStrategy.COSINE,
                max_staleness=datetime.timedelta(seconds=15),
            ),
        )

        docs = db.similarity_search(
            "Testing the langchain integration with spanner", k=3
        )

        assert len(docs) == 3

    def test_spanner_vector_search_data4(self, setup_database):
        loader, embeddings = setup_database
        db = SpannerVectorStore(
            instance_id=instance_id,
            database_id=google_database,
            table_name=table_name,
            id_column="row_id",
            ignore_metadata_columns=[],
            embedding_service=embeddings,
            metadata_json_column="metadata",
            query_parameters=QueryParameters(
                distance_strategy=DistanceStrategy.COSINE,
                max_staleness=datetime.timedelta(seconds=15),
            ),
        )

        docs = db.max_marginal_relevance_search(
            "Testing the langchain integration with spanner", k=3
        )

        assert len(docs) == 3


title_vector_size = 3
title_vector_index_name = "title_v_index"
title_vector_embedding_column = TableColumn(
    name="title_embedding", type="ARRAY<FLOAT64>", is_null=True
)


class TestSpannerVectorStoreGoogleSQL_ANN:
    # Sadly currently Cloud Spanner Vector Search Indices being
    # created and destroyed takes a very long time hence we
    # are creating and tearing down the indices exactly one.
    initialized: Dict[str, int] = dict()

    @pytest.fixture(scope="class")
    def setup_database(self, client):
        if len(self.initialized) == 0:
            self.initialized["a"] = 1

            SpannerVectorStore.init_vector_store_table(
                instance_id=instance_id,
                database_id=ann_db,
                table_name=table_name_ANN,
                vector_size=title_vector_size,
                id_column="row_id",
                metadata_columns=[
                    TableColumn(name="metadata", type="JSON", is_null=True),
                    TableColumn(name="title", type="STRING(MAX)", is_null=False),
                ],
                embedding_column=title_vector_embedding_column,
                secondary_indexes=[
                    VectorSearchIndex(
                        index_name=title_vector_index_name,
                        columns=[title_vector_embedding_column.name],
                        distance_type=DistanceStrategy.COSINE,
                        nullable_column=True,
                        num_branches=1000,
                        tree_depth=3,
                        num_leaves=100000,
                    ),
                ],
            )

        loader = HNLoader("https://news.ycombinator.com/item?id=34817881")
        embeddings = FakeEmbeddings(size=title_vector_size)

        def cleanup_db():
            print("\nPerforming GSQL cleanup...")
            database = client.instance(instance_id).database(ann_db)

            def delete_from_table(txn):
                return txn.execute_update(f"DELETE FROM {table_name_ANN} WHERE 1=1")

            database.run_in_transaction(delete_from_table)

            # Cloud Spanner Vector index creation takes multitudes of time
            # hence trying to drop and recreate indices can make tests
            # run for more than 2+ hours, hence this comment.
            # TODO: Uncoment these operations when Cloud Spanner has fixed the problem.
            # operation = database.update_ddl([
            #     f"DROP VECTOR INDEX IF EXISTS {title_vector_index_name}",
            #     f"DROP TABLE IF EXISTS {table_name_ANN}",
            # ])
            # operation.result(OPERATION_TIMEOUT_SECONDS)
            print("\nGSQL Cleanup complete.")

        yield loader, embeddings, cleanup_db

    def test_add_documents(self, setup_database):
        loader, embeddings, _ = setup_database

        db = SpannerVectorStore(
            instance_id=instance_id,
            database_id=ann_db,
            table_name=table_name_ANN,
            id_column="row_id",
            ignore_metadata_columns=[],
            vector_index_name=title_vector_index_name,
            embedding_column=title_vector_embedding_column,
            embedding_service=embeddings,
            metadata_json_column="metadata",
        )

        docs = loader.load()
        ids = [str(uuid.uuid4()) for _ in range(len(docs))]
        ids_row_inserted = db.add_documents(documents=docs, ids=ids)
        assert ids == ids_row_inserted

    def test_add_texts(self, setup_database):
        loader, embeddings, _ = setup_database

        db = SpannerVectorStore(
            instance_id=instance_id,
            database_id=ann_db,
            table_name=table_name_ANN,
            id_column="row_id",
            ignore_metadata_columns=[],
            vector_index_name=title_vector_index_name,
            embedding_column=title_vector_embedding_column,
            embedding_service=embeddings,
            metadata_json_column="metadata",
        )

        texts = [
            "Langchain Test Text 1",
            "Langchain Test Text 2",
            "Langchain Test Text 3",
        ]
        ids = [str(uuid.uuid4()) for _ in range(len(texts))]
        ids_row_inserted = db.add_texts(
            texts=texts,
            ids=ids,
            metadatas=[
                {"title": "Title 1"},
                {"title": "Title 2"},
                {"title": "Title 3"},
            ],
        )
        assert ids == ids_row_inserted

    def test_delete(self, setup_database):
        loader, embeddings, _ = setup_database

        db = SpannerVectorStore(
            instance_id=instance_id,
            database_id=ann_db,
            table_name=table_name_ANN,
            id_column="row_id",
            ignore_metadata_columns=[],
            vector_index_name=title_vector_index_name,
            embedding_column=title_vector_embedding_column,
            embedding_service=embeddings,
            metadata_json_column="metadata",
        )

        docs = loader.load()
        deleted = db.delete(documents=[docs[0], docs[1]])

        assert deleted

    def test_similarity_search(self, setup_database):
        loader, embeddings, _ = setup_database

        db = SpannerVectorStore(
            instance_id=instance_id,
            database_id=ann_db,
            table_name=table_name_ANN,
            id_column="row_id",
            ignore_metadata_columns=[],
            vector_index_name=title_vector_index_name,
            embedding_service=embeddings,
            embedding_column=title_vector_embedding_column,
            metadata_json_column="metadata",
            query_parameters=QueryParameters(
                algorithm=QueryParameters.NearestNeighborsAlgorithm.APPROXIMATE_NEAREST_NEIGHBOR,
                distance_strategy=DistanceStrategy.COSINE,
                max_staleness=datetime.timedelta(seconds=15),
            ),
        )

        docs = loader.load()
        ids = [str(uuid.uuid4()) for _ in range(len(docs))]
        db.add_documents(documents=docs, ids=ids)

        docs = db.similarity_search(
            "Testing the langchain integration with spanner",
            k=2,
        )

        assert len(docs) == 2

    def test_similarity_search_by_vector(self, setup_database):
        loader, embeddings, _ = setup_database

        db = SpannerVectorStore(
            instance_id=instance_id,
            database_id=ann_db,
            table_name=table_name_ANN,
            id_column="row_id",
            ignore_metadata_columns=[],
            vector_index_name=title_vector_index_name,
            embedding_service=embeddings,
            embedding_column=title_vector_embedding_column,
            metadata_json_column="metadata",
            query_parameters=QueryParameters(
                algorithm=QueryParameters.NearestNeighborsAlgorithm.APPROXIMATE_NEAREST_NEIGHBOR,
                distance_strategy=DistanceStrategy.COSINE,
                max_staleness=datetime.timedelta(seconds=15),
            ),
        )

        docs = loader.load()
        ids = [str(uuid.uuid4()) for _ in range(len(docs))]
        db.add_documents(documents=docs, ids=ids)

        embeds = embeddings.embed_query(
            "Testing the langchain integration with spanner"
        )

        docs = db.similarity_search_by_vector(
            embeds,
            k=3,
        )

        assert len(docs) == 3

    def test_max_marginal_relevance_search_with_score_by_vector(self, setup_database):
        loader, embeddings, _ = setup_database

        db = SpannerVectorStore(
            instance_id=instance_id,
            database_id=ann_db,
            table_name=table_name_ANN,
            id_column="row_id",
            ignore_metadata_columns=[],
            vector_index_name=title_vector_index_name,
            embedding_service=embeddings,
            metadata_json_column="metadata",
            embedding_column=title_vector_embedding_column,
            query_parameters=QueryParameters(
                algorithm=QueryParameters.NearestNeighborsAlgorithm.APPROXIMATE_NEAREST_NEIGHBOR,
                distance_strategy=DistanceStrategy.COSINE,
                max_staleness=datetime.timedelta(seconds=15),
            ),
        )

        docs = loader.load()
        ids = [str(uuid.uuid4()) for _ in range(len(docs))]
        db.add_documents(documents=docs, ids=ids)

        embeds = embeddings.embed_query(
            "Testing the langchain integration with spanner"
        )

        docs = db.max_marginal_relevance_search_with_score_by_vector(
            embeds,
            k=3,
        )

        assert len(docs) == 3

    def test_last_for_cleanup(self, setup_database):
        loader, _, cleanup = setup_database
        _ = loader
        cleanup()


class TestSpannerVectorStorePGSQL:
    @pytest.fixture(scope="class")
    def setup_database(self, client):
        SpannerVectorStore.init_vector_store_table(
            instance_id=instance_id,
            database_id=pg_database,
            table_name=table_name,
            id_column="row_id",
            metadata_columns=[
                TableColumn(name="metadata", type="JSONB", is_null=True),
                TableColumn(name="title", type="TEXT", is_null=False),
            ],
        )

        loader = HNLoader("https://news.ycombinator.com/item?id=34817881")

        embeddings = FakeEmbeddings(size=3)

        yield loader, embeddings

        print("\nPerforming PGSQL cleanup after each test...")

        database = client.instance(instance_id).database(pg_database)
        operation = database.update_ddl([f"DROP TABLE IF EXISTS {table_name}"])
        operation.result(OPERATION_TIMEOUT_SECONDS)

        # Code to perform teardown after each test goes here
        print("\n PGSQL Cleanup complete.")

    def test_spanner_vector_add_data1(self, setup_database):
        loader, embeddings = setup_database

        db = SpannerVectorStore(
            instance_id=instance_id,
            database_id=pg_database,
            table_name=table_name,
            id_column="row_id",
            ignore_metadata_columns=[],
            embedding_service=embeddings,
            metadata_json_column="metadata",
        )

        docs = loader.load()
        ids = [str(uuid.uuid4()) for _ in range(len(docs))]
        ids_row_inserted = db.add_documents(documents=docs, ids=ids)
        assert ids == ids_row_inserted

    def test_spanner_vector_add_data2(self, setup_database):
        loader, embeddings = setup_database

        db = SpannerVectorStore(
            instance_id=instance_id,
            database_id=pg_database,
            table_name=table_name,
            id_column="row_id",
            ignore_metadata_columns=[],
            embedding_service=embeddings,
            metadata_json_column="metadata",
        )

        texts = [
            "Langchain Test Text 1",
            "Langchain Test Text 2",
            "Langchain Test Text 3",
        ]
        ids = [str(uuid.uuid4()) for _ in range(len(texts))]
        ids_row_inserted = db.add_texts(
            texts=texts,
            ids=ids,
            metadatas=[
                {"title": "Title 1"},
                {"title": "Title 2"},
                {"title": "Title 3"},
            ],
        )
        assert ids == ids_row_inserted

    def test_spanner_vector_delete_data(self, setup_database):
        loader, embeddings = setup_database

        db = SpannerVectorStore(
            instance_id=instance_id,
            database_id=pg_database,
            table_name=table_name,
            id_column="row_id",
            ignore_metadata_columns=[],
            embedding_service=embeddings,
            metadata_json_column="metadata",
        )

        docs = loader.load()

        deleted = db.delete(documents=[docs[0], docs[1]])

        assert deleted

    def test_spanner_vector_search_data1(self, setup_database):
        loader, embeddings = setup_database

        db = SpannerVectorStore(
            instance_id=instance_id,
            database_id=pg_database,
            table_name=table_name,
            id_column="row_id",
            ignore_metadata_columns=[],
            embedding_service=embeddings,
            metadata_json_column="metadata",
        )

        docs = db.similarity_search(
            "Testing the langchain integration with spanner", k=2
        )

        assert len(docs) == 2

    def test_spanner_vector_search_data2(self, setup_database):
        loader, embeddings = setup_database

        db = SpannerVectorStore(
            instance_id=instance_id,
            database_id=pg_database,
            table_name=table_name,
            id_column="row_id",
            ignore_metadata_columns=[],
            embedding_service=embeddings,
            metadata_json_column="metadata",
        )

        embeds = embeddings.embed_query(
            "Testing the langchain integration with spanner"
        )

        docs = db.similarity_search_by_vector(embeds, k=3, pre_filter="1 = 1")

        assert len(docs) == 3

    def test_spanner_vector_search_data3(self, setup_database):
        loader, embeddings = setup_database

        db = SpannerVectorStore(
            instance_id=instance_id,
            database_id=pg_database,
            table_name=table_name,
            id_column="row_id",
            ignore_metadata_columns=[],
            embedding_service=embeddings,
            metadata_json_column="metadata",
            query_parameters=QueryParameters(
                distance_strategy=DistanceStrategy.COSINE,
                max_staleness=datetime.timedelta(seconds=15),
            ),
        )

        docs = db.similarity_search(
            "Testing the langchain integration with spanner", k=3
        )

        assert len(docs) == 3

    def test_spanner_vector_search_data4(self, setup_database):
        loader, embeddings = setup_database
        db = SpannerVectorStore(
            instance_id=instance_id,
            database_id=pg_database,
            table_name=table_name,
            id_column="row_id",
            ignore_metadata_columns=[],
            embedding_service=embeddings,
            metadata_json_column="metadata",
            query_parameters=QueryParameters(
                distance_strategy=DistanceStrategy.COSINE,
                max_staleness=datetime.timedelta(seconds=15),
            ),
        )

        docs = db.max_marginal_relevance_search(
            "Testing the langchain integration with spanner",
            k=3,
        )

        assert len(docs) == 3
