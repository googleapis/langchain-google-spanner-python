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
import os
import random
import string
from google.cloud.spanner import Client
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
          string.ascii_letters
          + ("" if exclude_whitespaces else string.whitespace)
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
  return datetime.datetime.fromtimestamp(
      random_int(0, 366 * 24 * 60 * 60 * 1000)
  )


def random_param():
  # None param is not supported.
  return random.choice(random_generators())() or random_param()


def random_array(g, l=0, u=2):
  length = random_int(l, u)
  return [g() for _ in range(length)]


def random_none():
  return None


def random_primitive_generators():
  return [
      random_int,
      random_string,
      random_bytes,
      random_bool,
      random_float,
      random_timestamp,
  ]


def random_generators():
  return (
      random_primitive_generators()
      + [lambda: random_array(g) for g in random_primitive_generators()]
      + [random_none]
  )


properties = [
    ("p{}".format(i), random_val_gen)
    for i, random_val_gen in enumerate(random_generators())
]


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
  return Relationship(
      source=s, target=t, type=e, properties=random_properties()
  )


def random_graph_doc(suffix):

  node_types = ["Node{}{}".format(i, suffix) for i in range(2)]
  edge_types = [
      ("Edge{}{}".format(k, suffix), node_types[i], node_types[j])
      for k in range(2)
      for i in range(2)
      for j in range(2)
  ]
  nodes = {
      t: [random_node(t) for _ in range(random_int(10, 20))] for t in node_types
  }

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

  def test_spanner_graph_random_doc(self):
    suffix = random_string(num_char=5, exclude_whitespaces=True)
    graph_name = "test_graph{}".format(suffix)
    graph = SpannerGraphStore(
        instance_id,
        google_database,
        graph_name,
        client=Client(project=project_id),
    )
    graph.refresh_schema()

    try:
      node_ids = set()
      edge_ids = set()
      for _ in range(3):
        graph_doc = random_graph_doc(suffix)
        graph.add_graph_documents([graph_doc])
        node_ids.update({(n.type, n.id) for n in graph_doc.nodes})
        edge_ids.update({
            (e.type, e.source.id, e.target.id) for e in graph_doc.relationships
        })
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
          """.format(graph_name),
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

  def test_spanner_graph_doc_with_duplicate_elements(self):
    suffix = random_string(num_char=5, exclude_whitespaces=True)
    graph_name = "test_graph{}".format(suffix)
    graph = SpannerGraphStore(
        instance_id,
        google_database,
        graph_name,
        client=Client(project=project_id),
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

      results = graph.query(
          """
          GRAPH {}

          MATCH -[e]->
          RETURN TO_JSON(e)['properties'] AS properties
          """.format(graph_name),
          params={"param": random_param()},
      )
      assert len(results) == 1

      print(set(edge0.properties.keys()))
      print(set(edge1.properties.keys()))
      print(set(results[0]["properties"].keys()))

      edge_properties = edge0.properties
      edge_properties.update(edge1.properties)
      missing_properties = set(edge_properties.keys()).difference(
          set(results[0]["properties"].keys())
      )
      assert (
          len(missing_properties) == 0
      ), "Missing properties of edge: {}".format(missing_properties)

    finally:
      print("Clean up graph with name `{}`".format(graph_name))
      graph.cleanup()
