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
# limitations under the License

from collections import namedtuple
import unittest

from google.cloud.spanner_admin_database_v1.types import DatabaseDialect

from langchain_google_spanner.vector_store import (
    AlgoKind,
    DistanceStrategy,
    GoogleSqlSemnatics,
    PGSqlSemnatics,
    SecondaryIndex,
    SpannerVectorStore,
)


class TestGoogleSqlSemnatics(unittest.TestCase):
    def test_distance_function_to_string(self):
        cases = [
            (DistanceStrategy.COSINE, "COSINE_DISTANCE"),
            (DistanceStrategy.DOT_PRODUCT, "DOT_PRODUCT"),
            (DistanceStrategy.EUCLIDEIAN, "EUCLIDEAN_DISTANCE"),
        ]

        sem = GoogleSqlSemnatics()
        got_results = []
        want_results = []
        for strategy, want_str in cases:
            got_results.append(sem.getDistanceFunction(strategy))
            want_results.append(want_str)

        assert got_results == want_results


class TestPGSqlSemnatics(unittest.TestCase):
    def test_distance_function_to_string(self):
        cases = [
            (DistanceStrategy.COSINE, "spanner.cosine_distance"),
            (DistanceStrategy.DOT_PRODUCT, "spanner.dot_product"),
            (DistanceStrategy.EUCLIDEIAN, "spanner.euclidean_distance"),
        ]

        sem = PGSqlSemnatics()
        got_results = []
        want_results = []
        for strategy, want_str in cases:
            got_results.append(sem.getDistanceFunction(strategy))
            want_results.append(want_str)

        assert got_results == want_results

    def test_distance_function_raises_exception_if_unknown(self):
        strategies = [
            100,
            -1,
        ]

        for strategy in strategies:
            with self.assertRaises(Exception):
                sem.getDistanceFunction(strategy)


class TestSpannerVectorStore_KNN(unittest.TestCase):
    def test_generate_create_table_sql(self):
        got = SpannerVectorStore._generate_create_table_sql(
            "users",
            "id",
            "essays",
            "science_scores",
            [],
            "id",
        )
        want = "CREATE TABLE users (\n  id STRING(36),\n  essays STRING(MAX),\n  science_scores ARRAY<FLOAT64>\n) PRIMARY KEY(id)"
        assert got == want

    def test_generate_secondary_indices_ddl_ANN(self):
        strategies = [
            DistanceStrategy.COSINE,
            DistanceStrategy.DOT_PRODUCT,
            DistanceStrategy.EUCLIDEIAN,
        ]

        for distance_strategy in strategies:
            got = SpannerVectorStore._generate_secondary_indices_ddl_ANN(
                "Documents",
                secondary_indexes=[
                    SecondaryIndex(
                        index_name="DocEmbeddingIndex",
                        columns=["DocEmbedding"],
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
                + f"  OPTIONS(distance_type='{distance_strategy}', tree_depth=3, num_branches=1000, num_leaves=100000)"
            ]

            assert canonicalize(got) == canonicalize(want)

    def test_generate_secondary_indices_ddl_ANN_raises_exception_for_non_GoogleSQL_dialect(
        self,
    ):
        strategies = [
            DistanceStrategy.COSINE,
            DistanceStrategy.DOT_PRODUCT,
            DistanceStrategy.EUCLIDEIAN,
        ]

        for strategy in strategies:
            with self.assertRaises(Exception):
                SpannerVectorStore._generate_secondary_indices_ddl_ANN(
                    "Documents",
                    dialect=DatabaseDialect.POSTGRESQL,
                    secondary_indexes=[
                        SecondaryIndex(
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
                    num_branches=1000,
                    tree_depth=3,
                    index_type=DistanceStrategy.COSINE,
                    num_leaves=100000,
                )
            ],
        )

        want = [
            "CREATE INDEX DocEmbeddingIndex ON Documents(DocEmbedding)  STORING (text)"
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
                    num_branches=1000,
                    tree_depth=3,
                    index_type=DistanceStrategy.COSINE,
                    num_leaves=100000,
                )
            ],
        )

        want = [
            "CREATE INDEX DocEmbeddingIndex ON Documents(DocEmbedding)  INCLUDE (text)"
        ]

        assert canonicalize(got) == canonicalize(want)

    def test_query_ANN(self):
        got = SpannerVectorStore._query_ANN(
            "DocId",
            "Documents",
            "DocEmbeddingIndex",
            [1.0, 2.0, 3.0],
            "DocEmbedding",
            10,
            DistanceStrategy.COSINE,
            limit=100,
        )

        want = (
            "SELECT DocId FROM Documents@{FORCE_INDEX=DocEmbeddingIndex}\n"
            + " ORDER BY APPROX_COSINE_DISTANCE(\n"
            + '  ARRAY<FLOAT32>[1.0, 2.0, 3.0], DocEmbedding, options => JSON \'{"num_leaves_to_search": 10})\n'
            + "LIMIT 100"
        )

        print("got", got)
        print("want", want)
        assert got == want


def trimSpaces(x: str) -> str:
    return x.lstrip("\n").rstrip("\n").replace("\t", "  ").strip()


def canonicalize(s):
    return list(map(trimSpaces, s))
