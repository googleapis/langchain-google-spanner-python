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

import unittest

from langchain_google_spanner.vector_store import (
    DistanceStrategy,
    GoogleSqlSemnatics,
    PGSqlSemnatics,
)


class TestGoogleSqlSemnatics(unittest.TestCase):
    def test_distance_function_to_string(self):
        cases = [
            (DistanceStrategy.COSINE, "COSINE_DISTANCE"),
            (DistanceStrategy.DOT_PRODUCT, "DOT_PRODUCT"),
            (DistanceStrategy.EUCLIDEIAN, "EUCLIDEAN_DISTANCE"),
            (DistanceStrategy.APPROX_COSINE, "APPROX_COSINE_DISTANCE"),
            (DistanceStrategy.APPROX_DOT_PRODUCT, "APPROX_DOT_PRODUCT"),
            (DistanceStrategy.APPROX_EUCLIDEAN, "APPROX_EUCLIDEAN_DISTANCE"),
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
            DistanceStrategy.APPROX_COSINE,
            DistanceStrategy.APPROX_DOT_PRODUCT,
            DistanceStrategy.APPROX_EUCLIDEAN,
        ]

        for strategy in strategies:
            with self.assertRaises(Exception):
                sem.getDistanceFunction(strategy)
