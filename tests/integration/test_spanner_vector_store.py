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
pg_database = os.environ.get("PG_DATABASE", None)
zone = os.environ.get("GOOGLE_DATABASE_ZONE", "us-west2")
table_name = "test_table" + str(uuid.uuid4()).replace("-", "_")
table_name_ANN = "products"


OPERATION_TIMEOUT_SECONDS = 240


@pytest.fixture(scope="module")
def client() -> Client:
    return Client(project=project_id)


@pytest.fixture()
def cleanupGSQL(client):
    yield

    print("\nPerforming GSQL cleanup after each test...")

    database = client.instance(instance_id).database(google_database)
    operation = database.update_ddl([f"DROP TABLE IF EXISTS {table_name}"])
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


class TestStaticUtilityGoogleSQL:
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


class TestSpannerVectorStoreGoogleSQL_ANN:
    @pytest.fixture(scope="class")
    def setup_database(self, client):
        distance_strategy = DistanceStrategy.COSINE
        SpannerVectorStore.init_vector_store_table(
            instance_id=instance_id,
            database_id=google_database,
            table_name=table_name_ANN,
            id_column=TableColumn("productId", type="INT64"),
            vector_size=758,
            embedding_column=TableColumn(
                name="productDescriptionEmbedding",
                type="ARRAY<FLOAT32>",
                is_null=True,
            ),
            metadata_columns=[
                TableColumn(name="categoryId", type="INT64", is_null=False),
                TableColumn(name="productName", type="STRING(MAX)", is_null=False),
                TableColumn(
                    name="productDescription", type="STRING(MAX)", is_null=False
                ),
                TableColumn(name="inventoryCount", type="INT64", is_null=False),
                TableColumn(name="priceInCents", type="INT64", is_null=True),
            ],
            secondary_indexes=[
                VectorSearchIndex(
                    index_name="ProductDescriptionEmbeddingIndex",
                    columns=["productDescriptionEmbedding"],
                    nullable_column=True,
                    num_branches=1000,
                    tree_depth=3,
                    index_type=distance_strategy,
                    num_leaves=100000,
                ),
            ],
        )

        raw_data = [
            (
                1,
                1,
                "Cymbal Helios Helmet",
                "Safety meets style with the Cymbal children's bike helmet. Its lightweight design, superior ventilation, and adjustable fit ensure comfort and protection on every ride. Stay bright and keep your child safe under the sun with Cymbal Helios!",
                100,
                10999,
            ),
            (
                1,
                2,
                "Cymbal Sprout",
                "Let their cycling journey begin with the Cymbal Sprout, the ideal balance bike for beginning riders ages 2-4 years. Its lightweight frame, low seat height, and puncture-proof tires promote stability and confidence as little ones learn to balance and steer. Watch them sprout into cycling enthusiasts with Cymbal Sprout!",
                10,
                13999,
            ),
            (
                1,
                3,
                "Cymbal Spark Jr.",
                "Light, vibrant, and ready for adventure, the Spark Jr. is the perfect first bike for young riders (ages 5-8). Its sturdy frame, easy-to-use brakes, and puncture-resistant tires inspire confidence and endless playtime. Let the spark of cycling ignite with Cymbal!",
                34,
                13900,
            ),
            (
                1,
                4,
                "Cymbal Summit",
                "Conquering trails is a breeze with the Summit mountain bike. Its lightweight aluminum frame, responsive suspension, and powerful disc brakes provide exceptional control and comfort for experienced bikers navigating rocky climbs or shredding downhill. Reach new heights with Cymbal Summit!",
                0,
                79999,
            ),
            (
                1,
                5,
                "Cymbal Breeze",
                "Cruise in style and embrace effortless pedaling with the Breeze electric bike. Its whisper-quiet motor and long-lasting battery let you conquer hills and distances with ease. Enjoy scenic rides, commutes, or errands with a boost of confidence from Cymbal Breeze!",
                72,
                129999,
            ),
            (
                1,
                6,
                "Cymbal Trailblazer Backpack",
                "Carry all your essentials in style with the Trailblazer backpack. Its water-resistant material, multiple compartments, and comfortable straps keep your gear organized and accessible, allowing you to focus on the adventure. Blaze new trails with Cymbal Trailblazer!",
                24,
                7999,
            ),
            (
                1,
                7,
                "Cymbal Phoenix Lights",
                "See and be seen with the Phoenix bike lights. Powerful LEDs and multiple light modes ensure superior visibility, enhancing your safety and enjoyment during day or night rides. Light up your journey with Cymbal Phoenix!",
                87,
                3999,
            ),
            (
                1,
                8,
                "Cymbal Windstar Pump",
                "Flat tires are no match for the Windstar pump. Its compact design, lightweight construction, and high-pressure capacity make inflating tires quick and effortless. Get back on the road in no time with Cymbal Windstar!",
                36,
                24999,
            ),
            (
                1,
                9,
                "Cymbal Odyssey Multi-Tool",
                "Be prepared for anything with the Odyssey multi-tool. This handy gadget features essential tools like screwdrivers, hex wrenches, and tire levers, keeping you ready for minor repairs and adjustments on the go. Conquer your journey with Cymbal Odyssey!",
                52,
                999,
            ),
            (
                1,
                10,
                "Cymbal Nomad Water Bottle",
                "Stay hydrated on every ride with the Nomad water bottle. Its sleek design, BPA-free construction, and secure lock lid make it the perfect companion for staying refreshed and motivated throughout your adventures. Hydrate and explore with Cymbal Nomad!",
                42,
                1299,
            ),
        ]

        columns = [
            "categoryId",
            "productId",
            "productName",
            "productDescription",
            "createTime",
            "inventoryCount",
            "priceInCents",
        ]

        model_ddl_statements = [
            f"""
            CREATE MODEL IF NOT EXISTS EmbeddingsModel INPUT(
                content STRING(MAX),
            ) OUTPUT(
                embeddings STRUCT<statistics STRUCT<truncated BOOL, token_count FLOAT32>, values ARRAY<FLOAT32>>,
            ) REMOTE OPTIONS (
                endpoint = '//aiplatform.googleapis.com/projects/{project_id}/locations/{zone}/publishers/google/models/text-embedding-004'
            )
            """,
            f"""
            CREATE MODEL IF NOT EXISTS LLMModel INPUT(
                prompt STRING(MAX),
            ) OUTPUT(
                content STRING(MAX),
            ) REMOTE OPTIONS (
                endpoint = '//aiplatform.googleapis.com/projects/{project_id}/locations/{zone}/publishers/google/models/gemini-pro',
                default_batch_size = 1
            )
            """,
            """
            UPDATE products p1
            SET productDescriptionEmbedding =
                (
                    SELECT embeddings.values from ML.PREDICT(
                        MODEL EmbeddingsModel,
                        (SELECT productDescription as content FROM products p2 where p2.productId=p1.productId)
                    )
                )
            WHERE categoryId=1
            """,
        ]
        database = client.instance(instance_id).database(google_database)

        def create_models():
            operation = database.update_ddl(model_ddl_statements)
            return operation.result(OPERATION_TIMEOUT_SECONDS)

        def get_embeddings(self):
            sql = """SELECT embeddings.values FROM ML.PREDICT(
              MODEL EmbeddingsModel,
               (SELECT "I'd like to buy a starter bike for my 3 year old child" as content)
            )"""

            with database.snapshot() as snapshot:
                res = snapshot.execute_sql(sql)
                return list(res)

        yield raw_data, columns, create_models, get_embeddings

        print("\nPerforming GSQL cleanup after each ANN test...")

        operation = database.update_ddl(
            [
                f"DROP TABLE IF EXISTS {table_name_ANN}",
                "DROP MODEL IF EXISTS EmbeddingsModel",
                "DROP MODEL IF EXISTS LLMModel",
                "DROP Index IF EXISTS ProductDescriptionEmbeddingIndex",
            ]
        )
        if False:  # Creating a vector index takes 30+ minutes, so avoiding this.
            operation.result(OPERATION_TIMEOUT_SECONDS)

        # Code to perform teardown after each test goes here
        print("\nGSQL Cleanup complete.")

    def test_ann_add_data1(self, setup_database):
        raw_data, columns, create_models, get_embeddings = setup_database

        # Retrieve embeddings using ML_PREDICT.
        embeddings = get_embeddings()
        print("embeddings", embeddings)

        db = SpannerVectorStore(
            instance_id=instance_id,
            database_id=google_database,
            table_name=table_name_ANN,
            id_column="categoryId",
            ignore_metadata_columns=[],
            embedding_service=embeddings,
            metadata_json_column="metadata",
        )
        _ = db


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
            "Testing the langchain integration with spanner", k=3
        )

        assert len(docs) == 3
