from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type, Union
import datetime
import os
import time
import uuid

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

project_id = 'quip-441723'
instance_id = 'contracting'
google_database = 'ann'
zone = os.environ.get("GOOGLE_DATABASE_ZONE", "us-west2")
table_name_ANN = "products"
OPERATION_TIMEOUT_SECONDS = 240

def use_case():
        # Initialize the vector store table  if necessary.
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
                TableColumn(name="createTime", type="TIMESTAMP", is_null=False),
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

        # Create the models if necessary.
        client = Client(project=project_id)
        database = client.instance(instance_id).database(google_database)

        model_ddl_statements = [
            f"""
            CREATE MODEL IF NOT EXISTS EmbeddingsModel INPUT(
                content STRING(MAX),
            ) OUTPUT(
                embeddings STRUCT<statistics STRUCT<truncated BOOL, token_count FLOAT32>, values ARRAY<FLOAT32>>,
            ) REMOTE OPTIONS (
                endpoint = '//aiplatform.googleapis.com/projects/{project_id}/locations/us-central1/publishers/google/models/text-embedding-004'
            )
            """,
            f"""
            CREATE MODEL IF NOT EXISTS LLMModel INPUT(
                prompt STRING(MAX),
            ) OUTPUT(
                content STRING(MAX),
            ) REMOTE OPTIONS (
                endpoint = '//aiplatform.googleapis.com/projects/{project_id}/locations/us-central1/publishers/google/models/gemini-pro',
                default_batch_size = 1
            )
            """,
        ]
        operation = database.update_ddl(model_ddl_statements)
        operation.result(OPERATION_TIMEOUT_SECONDS)

        def clear_and_insert_data(tx):
            tx.execute_update("DELETE FROM products WHERE 1=1")
            tx.insert(
                'products',
                columns=[
                    'categoryId', 'productId', 'productName',
                    'productDescription',
                    'createTime', 'inventoryCount', 'priceInCents',
                ],
                values=raw_data,
            )

            tx.execute_update(
            """UPDATE products p1
                SET productDescriptionEmbedding =
                (SELECT embeddings.values from ML.PREDICT(MODEL EmbeddingsModel,
                (SELECT productDescription as content FROM products p2 where p2.productId=p1.productId)))
                WHERE categoryId=1""",
            )

            embeddings = []
            rows = tx.execute_sql(
            """SELECT embeddings.values
                FROM ML.PREDICT(
                  MODEL EmbeddingsModel,
                   (SELECT "I'd like to buy a starter bike for my 3 year old child" as content)
                )""")

            for row in rows:
                for nesting in row:
                    embeddings.extend(nesting)

            return embeddings

        embeddings = database.run_in_transaction(clear_and_insert_data)

        vec_store = SpannerVectorStore(
            instance_id=instance_id,
            database_id=google_database,
            table_name=table_name_ANN,
            id_column="categoryId",
            embedding_service=embeddings,
            embedding_column="productDescriptionEmbedding",
            skip_not_nullable_columns=True,
        )
        vec_store.search_by_ANN(
            'ProductDescriptionEmbeddingIndex',
            1000,
            k=20,
            embedding_column_is_nullable=True,
            return_columns=['productName', 'productDescription', 'inventoryCount'],
        )

def main():
    use_case() 


def PENDING_COMMIT_TIMESTAMP():
    return (datetime.datetime.utcnow() + datetime.timedelta(days=1)).isoformat() + "Z"
    return 'PENDING_COMMIT_TIMESTAMP()'

raw_data = [ 
    (1, 1, "Cymbal Helios Helmet", "Safety meets style with the Cymbal children's bike helmet. Its lightweight design, superior ventilation, and adjustable fit ensure comfort and protection on every ride. Stay bright and keep your child safe under the sun with Cymbal Helios!", PENDING_COMMIT_TIMESTAMP(), 100, 10999),
    (1, 2, "Cymbal Sprout", "Let their cycling journey begin with the Cymbal Sprout, the ideal balance bike for beginning riders ages 2-4 years. Its lightweight frame, low seat height, and puncture-proof tires promote stability and confidence as little ones learn to balance and steer. Watch them sprout into cycling enthusiasts with Cymbal Sprout!", PENDING_COMMIT_TIMESTAMP(), 10, 13999),
    (1, 3, "Cymbal Spark Jr.", "Light, vibrant, and ready for adventure, the Spark Jr. is the perfect first bike for young riders (ages 5-8). Its sturdy frame, easy-to-use brakes, and puncture-resistant tires inspire confidence and endless playtime. Let the spark of cycling ignite with Cymbal!", PENDING_COMMIT_TIMESTAMP(), 34, 13900),
    (1, 4, "Cymbal Summit", "Conquering trails is a breeze with the Summit mountain bike. Its lightweight aluminum frame, responsive suspension, and powerful disc brakes provide exceptional control and comfort for experienced bikers navigating rocky climbs or shredding downhill. Reach new heights with Cymbal Summit!", PENDING_COMMIT_TIMESTAMP(), 0, 79999),
    (1, 5, "Cymbal Breeze", "Cruise in style and embrace effortless pedaling with the Breeze electric bike. Its whisper-quiet motor and long-lasting battery let you conquer hills and distances with ease. Enjoy scenic rides, commutes, or errands with a boost of confidence from Cymbal Breeze!", PENDING_COMMIT_TIMESTAMP(), 72, 129999),
    (1, 6, "Cymbal Trailblazer Backpack", "Carry all your essentials in style with the Trailblazer backpack. Its water-resistant material, multiple compartments, and comfortable straps keep your gear organized and accessible, allowing you to focus on the adventure. Blaze new trails with Cymbal Trailblazer!", PENDING_COMMIT_TIMESTAMP(), 24, 7999),
    (1, 7, "Cymbal Phoenix Lights", "See and be seen with the Phoenix bike lights. Powerful LEDs and multiple light modes ensure superior visibility, enhancing your safety and enjoyment during day or night rides. Light up your journey with Cymbal Phoenix!", PENDING_COMMIT_TIMESTAMP(), 87, 3999),
    (1, 8, "Cymbal Windstar Pump", "Flat tires are no match for the Windstar pump. Its compact design, lightweight construction, and high-pressure capacity make inflating tires quick and effortless. Get back on the road in no time with Cymbal Windstar!", PENDING_COMMIT_TIMESTAMP(), 36, 24999),
    (1, 9,"Cymbal Odyssey Multi-Tool","Be prepared for anything with the Odyssey multi-tool. This handy gadget features essential tools like screwdrivers, hex wrenches, and tire levers, keeping you ready for minor repairs and adjustments on the go. Conquer your journey with Cymbal Odyssey!", PENDING_COMMIT_TIMESTAMP(), 52, 999),
    (1, 10,"Cymbal Nomad Water Bottle","Stay hydrated on every ride with the Nomad water bottle. Its sleek design, BPA-free construction, and secure lock lid make it the perfect companion for staying refreshed and motivated throughout your adventures. Hydrate and explore with Cymbal Nomad!", PENDING_COMMIT_TIMESTAMP(), 42, 1299),
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


if __name__ == '__main__':
    main()

