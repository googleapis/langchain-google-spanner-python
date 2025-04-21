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

import base64
import datetime
import json
import os
import random
import string

import pytest
from google.cloud.spanner import Client  # type: ignore
from google.cloud.spanner_v1 import JsonObject
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship
from langchain_core.documents import Document

from langchain_google_spanner import SpannerGraphStore

project_id = os.environ["PROJECT_ID"]
instance_id = os.environ["INSTANCE_ID"]
google_database = os.environ["GOOGLE_DATABASE"]


def random_int(l=0, u=100):
    return random.randint(l, u)


def random_string(num_char=5, exclude_whitespaces=False):
    return "".join(
        random.choice(
            string.ascii_letters + ("" if exclude_whitespaces else string.whitespace)
        )
        for _ in range(num_char)
    )


def random_bytes(num_char=5):
    return base64.encodebytes(random.randbytes(num_char))


def random_bool():
    return random.choice([True, False])


def random_float():
    return random.random()


def random_timestamp():
    return datetime.datetime.fromtimestamp(random_int(0, 366 * 24 * 60 * 60 * 1000))


def random_date():
    return datetime.date.fromtimestamp(random_int(0, 366 * 24 * 60 * 60 * 1000))


def random_param():
    # None param is not supported.
    return random.choice(random_generators())() or random_param()


def random_array(g, l=0, u=2):
    length = random_int(l, u)
    return [g() for _ in range(length)]


def random_none():
    return None


def random_json():
    return JsonObject({random_string(exclude_whitespaces=True): random_int()})


def random_primitive_generators():
    return [
        random_int,
        random_string,
        random_bytes,
        random_bool,
        random_float,
        random_timestamp,
        random_date,
    ]


def random_generators():
    return (
        random_primitive_generators()
        + [lambda: random_array(g) for g in random_primitive_generators()]
        + [random_none, random_json]
    )


properties = [
    ("p{}".format(i), random_val_gen)
    for i, random_val_gen in enumerate(random_generators())
]


def random_property_names(k):
    return [k for k, _ in random.choices(properties, k=k)]


def random_property():
    k, vg = random.choice(properties)
    return k, vg()


def random_properties():
    props = {}
    for _ in range(random_int(0, len(properties))):
        k, v = random_property()
        props[k] = v
    return props


def random_node(t):
    return Node(id=random_string(), type=t, properties=random_properties())


def random_edge(e, s, t):
    return Relationship(source=s, target=t, type=e, properties=random_properties())


def random_graph_doc(suffix):
    node_types = ["Node{}{}".format(i, suffix) for i in range(2)]
    edge_types = [
        ("Edge{}{}".format(k, suffix), node_types[i], node_types[j])
        for k in range(2)
        for i in range(2)
        for j in range(2)
    ]
    nodes = {t: [random_node(t) for _ in range(random_int(10, 20))] for t in node_types}

    edges = {
        e: [
            random_edge(e, source, target)
            for source in random.sample(nodes[s], random_int(1, 5))
            for target in random.sample(nodes[t], random_int(1, 5))
        ]
        for e, s, t in edge_types
    }

    return GraphDocument(
        nodes=[n for ns in nodes.values() for n in ns],
        relationships=[e for es in edges.values() for e in es],
        source=Document(
            page_content="Hello, world!",
            metadata={"source": "https://example.com"},
        ),
    )


class TestSpannerGraphStore:
    @pytest.mark.parametrize("use_flexible_schema", [False, True])
    def test_spanner_graph_random_doc(self, use_flexible_schema):
        suffix = random_string(num_char=5, exclude_whitespaces=True)
        graph_name = "test_graph{}".format(suffix)
        graph = SpannerGraphStore(
            instance_id,
            google_database,
            graph_name,
            client=Client(project=project_id),
            use_flexible_schema=use_flexible_schema,
            static_node_properties=random_property_names(
                random_int(l=0, u=len(properties))
            ),
            static_edge_properties=random_property_names(
                random_int(l=0, u=len(properties))
            ),
        )
        graph.refresh_schema()

        try:
            node_ids = set()
            edge_ids = set()
            for _ in range(3):
                graph_doc = random_graph_doc(suffix)
                graph.add_graph_documents([graph_doc])
                node_ids.update({(n.type, n.id) for n in graph_doc.nodes})
                edge_ids.update(
                    {
                        (e.type, e.source.id, e.target.id)
                        for e in graph_doc.relationships
                    }
                )
                graph.refresh_schema()

            results = graph.query(
                """
          GRAPH {}

          MATCH ->
          RETURN "edge" AS type, COUNT(*) AS num_elements

          UNION ALL

          MATCH ()
          RETURN "node" AS type, COUNT(*) AS num_elements

          NEXT

          RETURN type, num_elements, @param AS param
          ORDER BY type
          """.format(
                    graph_name
                ),
                params={"param": random_param()},
            )
            assert len(results) == 2
            assert results[0]["type"] == "edge", "Mismatch type"
            assert results[0]["num_elements"] == len(
                edge_ids
            ), "Mismatch number of edges"
            assert results[1]["type"] == "node", "Mismatch type"
            assert results[1]["num_elements"] == len(
                node_ids
            ), "Mismatch number of nodes"

        finally:
            print("Clean up graph with name `{}`".format(graph_name))
            print(graph.get_schema)
            print(graph.get_structured_schema)
            print(graph.get_ddl())
            graph.cleanup()

    @pytest.mark.parametrize("use_flexible_schema", [False, True])
    def test_spanner_graph_doc_with_duplicate_elements(self, use_flexible_schema):
        suffix = random_string(num_char=5, exclude_whitespaces=True)
        graph_name = "test_graph{}".format(suffix)
        graph = SpannerGraphStore(
            instance_id,
            google_database,
            graph_name,
            client=Client(project=project_id),
            use_flexible_schema=use_flexible_schema,
            static_node_properties=random_property_names(
                random_int(l=0, u=len(properties))
            ),
            static_edge_properties=random_property_names(
                random_int(l=0, u=len(properties))
            ),
        )
        graph.refresh_schema()

        try:
            node0 = random_node("Node0{}".format(suffix))
            node1 = random_node("Node1{}".format(suffix))
            edge0 = random_edge("Edge01", node0, node1)
            edge1 = random_edge("Edge01", node0, node1)

            doc = GraphDocument(
                nodes=[node0, node1, node0, node1],
                relationships=[edge0, edge1],
                source=Document(
                    page_content="Hello, world!",
                    metadata={"source": "https://example.com"},
                ),
            )
            graph.add_graph_documents([doc])

            # In the case of flexible schema, `properties` is a nested json
            # field.
            results = graph.query(
                """
          GRAPH {}

          MATCH -[e]->
          LET properties = TO_JSON(e)['properties']
          RETURN COALESCE(properties.properties, JSON "{{}}") AS dynamic_properties,
                 properties AS static_properties
          """.format(
                    graph_name
                ),
                params={"param": random_param()},
            )
            assert len(results) == 1

            edge_properties = edge0.properties
            edge_properties.update(edge1.properties)
            missing_properties = set(edge_properties.keys()).difference(
                set(results[0]["dynamic_properties"].keys()).union(
                    set(results[0]["static_properties"].keys())
                )
            )
            print(edge0.properties)
            print(edge1.properties)
            print(results)
            assert (
                len(missing_properties) == 0
            ), "Missing properties of edge: {}".format(missing_properties)

        finally:
            print("Clean up graph with name `{}`".format(graph_name))
            graph.cleanup()

    @pytest.mark.parametrize("use_flexible_schema", [False, True])
    def test_spanner_graph_avoid_unnecessary_overwrite(self, use_flexible_schema):
        suffix = random_string(num_char=5, exclude_whitespaces=True)
        graph_name = "test_graph{}".format(suffix)
        graph = SpannerGraphStore(
            instance_id,
            google_database,
            graph_name,
            client=Client(project=project_id),
            use_flexible_schema=use_flexible_schema,
            static_node_properties=["a", "b"],
            static_edge_properties=["a", "b"],
        )
        graph.refresh_schema()

        try:
            node0 = Node(
                id=random_string(),
                type="Node{}".format(suffix),
                properties={"a": 1, "b": 1},
            )
            node1 = Node(
                id=random_string(),
                type="Node{}".format(suffix),
                properties={"a": 1, "b": 1},
            )
            edge0 = Relationship(
                source=node0,
                target=node1,
                type="Edge{}".format(suffix),
                properties={"a": 1, "b": 1},
            )
            doc = GraphDocument(
                nodes=[node0, node1],
                relationships=[edge0],
                source=Document(
                    page_content="Hello, world!",
                    metadata={"source": "https://example.com"},
                ),
            )

            query = """GRAPH {}
                   MATCH (n {{id: @nodeId}})
                   LET properties = TO_JSON(n)['properties']
                   RETURN int64(properties.a) AS a, int64(properties.b) AS b
                   UNION ALL
                   MATCH -[e {{id: @nodeId}}]->
                   LET properties = TO_JSON(e)['properties']
                   RETURN int64(properties.a) AS a, int64(properties.b) AS b
                """.format(
                graph_name
            )
            graph.add_graph_documents([doc])

            # Test initial value: a=1, b=1
            results = graph.query(query, {"nodeId": node0.id})
            assert len(results) == 2, "Actual results: {}".format(results)
            assert all((r["a"] == 1 for r in results)), "Actual results: {}".format(
                results
            )
            assert all((r["b"] == 1 for r in results)), "Actual results: {}".format(
                results
            )

            node0.properties["a"] = 2
            edge0.properties["a"] = 2
            graph.add_graph_documents([doc])

            # Test value after first overwrite: a=2, b=1
            results = graph.query(query, {"nodeId": node0.id})
            assert len(results) == 2, "Actual results: {}".format(results)
            assert all((r["a"] == 2 for r in results)), "Actual results: {}".format(
                results
            )
            assert all((r["b"] == 1 for r in results)), "Actual results: {}".format(
                results
            )

            node0.properties = {}
            edge0.properties = {}
            graph.add_graph_documents([doc])

            # Test value after second overwrite: a=2, b=1
            results = graph.query(query, {"nodeId": node0.id})
            assert len(results) == 2, "Actual results: {}".format(results)
            assert all((r["a"] == 2 for r in results)), "Actual results: {}".format(
                results
            )
            assert all((r["b"] == 1 for r in results)), "Actual results: {}".format(
                results
            )
        finally:
            print("Clean up graph with name `{}`".format(graph_name))
            graph.cleanup()

    @pytest.mark.parametrize(
        "graph_name, raises_exception",
        [
            ("test_graph", False),
            ("123test_graph", True),
            ("test_graph2", False),
            ("test-graph", True),
        ],
    )
    def test_spanner_graph_invalid_graph_name(self, graph_name, raises_exception):
        suffix = random_string(num_char=5, exclude_whitespaces=True)
        graph_name += suffix
        if raises_exception:
            with pytest.raises(Exception) as excinfo:
                SpannerGraphStore(
                    instance_id,
                    google_database,
                    graph_name,
                    client=Client(project=project_id),
                    static_node_properties=["a", "b"],
                    static_edge_properties=["a", "b"],
                )
            assert "not a valid identifier" in str(excinfo.value)
        else:
            SpannerGraphStore(
                instance_id,
                google_database,
                graph_name,
                client=Client(project=project_id),
                static_node_properties=["a", "b"],
                static_edge_properties=["a", "b"],
            )

    @pytest.mark.parametrize("use_flexible_schema", [False, True])
    def test_spanner_graph_with_existing_graph(self, use_flexible_schema):
        suffix = random_string(num_char=5, exclude_whitespaces=True)
        graph_name = "test_graph{}".format(suffix)
        node_table_name = "{}_node".format(graph_name)
        edge_table_name = "{}_edge".format(graph_name)
        graph = SpannerGraphStore(
            instance_id,
            google_database,
            graph_name,
            client=Client(project=project_id),
            use_flexible_schema=use_flexible_schema,
        )
        graph.refresh_schema()
        try:
            graph.impl.apply_ddls(
                [
                    f"""
                  CREATE TABLE IF NOT EXISTS {node_table_name} (
                    id INT64 NOT NULL,
                    str STRING(MAX),
                    token TOKENLIST AS (TOKENIZE_FULLTEXT(str)) HIDDEN,
                  ) PRIMARY KEY (id)
                """,
                    f"""
                  CREATE TABLE IF NOT EXISTS {edge_table_name} (
                    id INT64 NOT NULL,
                    target_id INT64 NOT NULL,
                  ) PRIMARY KEY (id, target_id)
                """,
                    f"""
                  CREATE PROPERTY GRAPH IF NOT EXISTS {graph_name}
                  NODE TABLES (
                    {node_table_name} AS NodeA
                      LABEL Node
                      LABEL NodeA PROPERTIES(id, id AS node_a_id),
                    {node_table_name} AS NodeB
                      LABEL Node
                      LABEL NodeB PROPERTIES(id, id AS node_b_id)
                  )
                  EDGE TABLES (
                    {edge_table_name} AS EdgeAB
                      SOURCE KEY(id) REFERENCES NodeA
                      DESTINATION KEY(target_id) REFERENCES NodeB
                      LABEL Edge PROPERTIES(id AS source_id, target_id AS dest_id)
                      LABEL EdgeAB PROPERTIES(id AS node_a_id, target_id AS node_b_id),
                    {edge_table_name} AS EdgeBA
                      SOURCE KEY(id) REFERENCES NodeB
                      DESTINATION KEY(target_id) REFERENCES NodeA
                      LABEL Edge PROPERTIES(id AS source_id, target_id AS dest_id)
                      LABEL EdgeBA PROPERTIES(target_id AS node_a_id, id AS node_b_id),
                  )
                """,
                ]
            )
            graph.refresh_schema()
            schema = json.loads(graph.get_schema)
            edgeab = graph.schema.get_edge_schema("EdgeAB")
            edgeba = graph.schema.get_edge_schema("EdgeBA")
            assert (edgeab.source.node_name, edgeab.target.node_name) == (
                "NodeA",
                "NodeB",
            )
            assert (edgeba.source.node_name, edgeba.target.node_name) == (
                "NodeB",
                "NodeA",
            )
            # TOKENLIST-typed properties are ignored.
            assert len(schema["Node properties per node label"]["Node"]) == 4, schema[
                "Node properties per node label"
            ]["Node"]
            assert len(schema["Node properties per node label"]["NodeA"]) == 3, schema[
                "Node properties per node label"
            ]["NodeA"]
            assert len(schema["Node properties per node label"]["NodeB"]) == 3, schema[
                "Node properties per node label"
            ]["NodB"]
            assert len(schema["Possible edges per label"]["EdgeAB"]) == 4, schema[
                "Possible edges per label"
            ]["EdgeAB"]
            assert len(schema["Possible edges per label"]["EdgeBA"]) == 4, schema[
                "Possible edges per label"
            ]["EdgeBA"]
            assert len(schema["Possible edges per label"]["Edge"]) == 8, schema[
                "Possible edges per label"
            ]["Edge"]
        finally:
            print("Clean up graph with name `{}`".format(graph_name))
            graph.cleanup()
