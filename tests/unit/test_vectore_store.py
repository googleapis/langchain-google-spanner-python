# Copyright 2025 Google LLC
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
# limitations under the License

import unittest
from collections import namedtuple

from google.cloud.spanner_admin_database_v1.types import DatabaseDialect

from langchain_google_spanner.vector_store import (
    DistanceStrategy,
    GoogleSqlSemantics,
    PGSqlSemantics,
    SecondaryIndex,
    SpannerVectorStore,
    VectorSearchIndex,
)


class TestGoogleSqlSemantics(unittest.TestCase):
    def test_distance_function_to_string(self):
        cases = [
            (DistanceStrategy.COSINE, "COSINE_DISTANCE"),
            (DistanceStrategy.DOT_PRODUCT, "DOT_PRODUCT"),
            (DistanceStrategy.EUCLIDEAN, "EUCLIDEAN_DISTANCE"),
        ]

        sem = GoogleSqlSemantics()
        got_results = []
        want_results = []
        for strategy, want_str in cases:
            got_results.append(sem.getDistanceFunction(strategy))
            want_results.append(want_str)

        assert got_results == want_results


class TestPGSqlSemantics(unittest.TestCase):
    sem = PGSqlSemantics()

    def test_distance_function_to_string(self):
        cases = [
            (DistanceStrategy.COSINE, "spanner.cosine_distance"),
            (DistanceStrategy.DOT_PRODUCT, "spanner.dot_product"),
            (DistanceStrategy.EUCLIDEAN, "spanner.euclidean_distance"),
        ]

        got_results = []
        want_results = []
        for strategy, want_str in cases:
            got_results.append(self.sem.getDistanceFunction(strategy))
            want_results.append(want_str)

        assert got_results == want_results

    def test_distance_function_raises_exception_if_unknown(self):
        strategies = [
            100,
            -1,
        ]

        for strategy in strategies:
            with self.assertRaises(Exception):
                self.sem.getDistanceFunction(strategy)


class TestSpannerVectorStore(unittest.TestCase):
    def test_generate_create_table_sql(self):
        got = SpannerVectorStore._generate_create_table_sql(
            "users",
            "id",
            "essays",
            "science_scores",
            [],
            "id",
        )
        want = (
            "CREATE TABLE IF NOT EXISTS users (\n  id STRING(36),\n  essays STRING(MAX),"
            + "\n  science_scores ARRAY<FLOAT64>\n) PRIMARY KEY(id)"
        )
        assert got == want

    def test_generate_secondary_indices_ddl_ANN(self):
        strategies = [
            DistanceStrategy.COSINE,
            DistanceStrategy.DOT_PRODUCT,
            DistanceStrategy.EUCLIDEAN,
        ]

        nullables = [True, False]
        for distance_strategy in strategies:
            for nullable in nullables:
                got = SpannerVectorStore._generate_secondary_indices_ddl_ANN(
                    "Documents",
                    secondary_indexes=[
                        VectorSearchIndex(
                            index_name="DocEmbeddingIndex",
                            columns=["DocEmbedding"],
                            nullable_column=nullable,
                            num_branches=1000,
                            tree_depth=3,
                            index_type=distance_strategy,
                            num_leaves=100000,
                        )
                    ],
                )

                want = [
                    "CREATE VECTOR INDEX DocEmbeddingIndex\n"
                    + "  ON Documents(DocEmbedding)\n"
                    + "  WHERE DocEmbedding IS NOT NULL\n"
                    + f"  OPTIONS(distance_type='{distance_strategy}', "
                    + "tree_depth=3, num_branches=1000, num_leaves=100000)"
                ]
                if not nullable:
                    want = [
                        "CREATE VECTOR INDEX DocEmbeddingIndex\n"
                        + "  ON Documents(DocEmbedding)\n"
                        + f"  OPTIONS(distance_type='{distance_strategy}', "
                        + "tree_depth=3, num_branches=1000, num_leaves=100000)"
                    ]

                assert canonicalize(got) == canonicalize(want)

    def test_generate_ANN_indices_exception_for_non_GoogleSQL_dialect(
        self,
    ):
        strategies = [
            DistanceStrategy.COSINE,
            DistanceStrategy.DOT_PRODUCT,
            DistanceStrategy.EUCLIDEAN,
        ]

        for strategy in strategies:
            with self.assertRaises(Exception):
                SpannerVectorStore._generate_secondary_indices_ddl_ANN(
                    "Documents",
                    dialect=DatabaseDialect.POSTGRESQL,
                    secondary_indexes=[
                        VectorSearchIndex(
                            index_name="DocEmbeddingIndex",
                            columns=["DocEmbedding"],
                            num_branches=1000,
                            tree_depth=3,
                            index_type=strategy,
                            num_leaves=100000,
                        )
                    ],
                )

    def test_generate_secondary_indices_ddl_KNN_GoogleDialect(self):
        embed_column = namedtuple("Column", ["name"])
        embed_column.name = "text"
        got = SpannerVectorStore._generate_secondary_indices_ddl_KNN(
            "Documents",
            embedding_column=embed_column,
            dialect=DatabaseDialect.GOOGLE_STANDARD_SQL,
            secondary_indexes=[
                SecondaryIndex(
                    index_name="DocEmbeddingIndex",
                    columns=["DocEmbedding"],
                )
            ],
        )

        want = [
            "CREATE INDEX DocEmbeddingIndex ON "
            + "Documents(DocEmbedding)  STORING (text)"
        ]

        assert canonicalize(got) == canonicalize(want)

    def test_generate_secondary_indices_ddl_KNN_PostgresDialect(self):
        embed_column = namedtuple("Column", ["name"])
        embed_column.name = "text"
        got = SpannerVectorStore._generate_secondary_indices_ddl_KNN(
            "Documents",
            embedding_column=embed_column,
            dialect=DatabaseDialect.POSTGRESQL,
            secondary_indexes=[
                SecondaryIndex(
                    index_name="DocEmbeddingIndex",
                    columns=["DocEmbedding"],
                )
            ],
        )

        want = [
            "CREATE INDEX DocEmbeddingIndex ON "
            + "Documents(DocEmbedding)  INCLUDE (text)"
        ]

        assert canonicalize(got) == canonicalize(want)

    def test_query_ANN(self):
        got = SpannerVectorStore._query_ANN(
            "Documents",
            "DocEmbeddingIndex",
            "DocEmbedding",
            [1.0, 2.0, 3.0],
            10,
            DistanceStrategy.COSINE,
            limit=100,
            return_columns=["DocId"],
        )

        want = (
            "SELECT DocId FROM Documents@{FORCE_INDEX=DocEmbeddingIndex}\n"
            + "ORDER BY APPROX_COSINE_DISTANCE(\n"
            + "  ARRAY<FLOAT32>[1.0, 2.0, 3.0], DocEmbedding, options => JSON "
            + '\'{"num_leaves_to_search": 10})\n'
            + "LIMIT 100"
        )

        assert got == want

    def test_query_ANN_column_is_nullable(self):
        got = SpannerVectorStore._query_ANN(
            "Documents",
            "DocEmbeddingIndex",
            "DocEmbedding",
            [1.0, 2.0, 3.0],
            10,
            DistanceStrategy.COSINE,
            limit=100,
            embedding_column_is_nullable=True,
            return_columns=["DocId"],
        )

        want = (
            "SELECT DocId FROM Documents@{FORCE_INDEX=DocEmbeddingIndex}\n"
            + "WHERE DocEmbedding IS NOT NULL\n"
            + "ORDER BY APPROX_COSINE_DISTANCE(\n"
            + "  ARRAY<FLOAT32>[1.0, 2.0, 3.0], DocEmbedding, options => JSON "
            + '\'{"num_leaves_to_search": 10})\n'
            + "LIMIT 100"
        )

        assert got == want

    def test_query_ANN_column_unspecified_return_columns_star_result(self):
        got = SpannerVectorStore._query_ANN(
            "Documents",
            "DocEmbeddingIndex",
            "DocEmbedding",
            [1.0, 2.0, 3.0],
            10,
            DistanceStrategy.COSINE,
            limit=100,
            embedding_column_is_nullable=True,
        )

        want = (
            "SELECT * FROM Documents@{FORCE_INDEX=DocEmbeddingIndex}\n"
            + "WHERE DocEmbedding IS NOT NULL\n"
            + "ORDER BY APPROX_COSINE_DISTANCE(\n"
            + "  ARRAY<FLOAT32>[1.0, 2.0, 3.0], DocEmbedding, options => JSON "
            + '\'{"num_leaves_to_search": 10})\n'
            + "LIMIT 100"
        )

        assert got == want

    def test_query_ANN_order_DESC(self):
        got = SpannerVectorStore._query_ANN(
            "Documents",
            "DocEmbeddingIndex",
            "DocEmbedding",
            [1.0, 2.0, 3.0],
            10,
            DistanceStrategy.COSINE,
            limit=100,
            return_columns=["DocId"],
            ascending=False,
        )

        want = (
            "SELECT DocId FROM Documents@{FORCE_INDEX=DocEmbeddingIndex}\n"
            + "ORDER BY APPROX_COSINE_DISTANCE(\n"
            + "  ARRAY<FLOAT32>[1.0, 2.0, 3.0], DocEmbedding, options => JSON "
            + '\'{"num_leaves_to_search": 10}) DESC\n'
            + "LIMIT 100"
        )

        assert got == want

    def test_query_ANN_unspecified_limit(self):
        got = SpannerVectorStore._query_ANN(
            "Documents",
            "DocEmbeddingIndex",
            "DocEmbedding",
            [1.0, 2.0, 3.0],
            10,
            DistanceStrategy.COSINE,
            return_columns=["DocId"],
        )

        want = (
            "SELECT DocId FROM Documents@{FORCE_INDEX=DocEmbeddingIndex}\n"
            + "ORDER BY APPROX_COSINE_DISTANCE(\n"
            + "  ARRAY<FLOAT32>[1.0, 2.0, 3.0], DocEmbedding, options => JSON "
            + '\'{"num_leaves_to_search": 10})'
        )

        assert got == want


def trimSpaces(x: str) -> str:
    return x.lstrip("\n").rstrip("\n").replace("\t", "  ").strip()


def canonicalize(s):
    return list(map(trimSpaces, s))
