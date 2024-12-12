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

import os
import random
import string

import pytest
from google.cloud import spanner
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship
from langchain_core.documents import Document
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings

from langchain_google_spanner.graph_retriever import (
    SpannerGraphGQLRetriever,
    SpannerGraphNodeVectorRetriever,
    SpannerGraphSemanticGQLRetriever,
)
from langchain_google_spanner.graph_store import SpannerGraphStore

project_id = os.environ["PROJECT_ID"]
instance_id = os.environ["INSTANCE_ID"]
database_id = os.environ["GOOGLE_DATABASE"]


def random_string(num_char=3):
    return "".join(random.choice(string.ascii_letters) for _ in range(num_char))


def get_llm():
    llm = ChatVertexAI(
        model="gemini-1.5-flash-002",
        temperature=0,
    )
    return llm


def get_embedding():
    embeddings = VertexAIEmbeddings(model_name="text-embedding-004")
    return embeddings


def get_spanner_graph():
    suffix = random_string(num_char=3)
    graph_name = "test_graph{}".format(suffix)
    graph = SpannerGraphStore(
        instance_id=instance_id,
        database_id=database_id,
        graph_name=graph_name,
        client=spanner.Client(project=project_id),
    )
    return graph, suffix


def load_data(graph: SpannerGraphStore, suffix: str):
    type_suffix = "_" + suffix
    graph_documents = [
        GraphDocument(
            nodes=[
                Node(
                    id="Elias Thorne",
                    type="Person" + type_suffix,
                    properties={
                        "name": "Elias Thorne",
                        "description": "lived in the desert",
                    },
                ),
                Node(
                    id="Zephyr",
                    type="Animal" + type_suffix,
                    properties={"name": "Zephyr", "description": "pet falcon"},
                ),
                Node(
                    id="Elara",
                    type="Person" + type_suffix,
                    properties={
                        "name": "Elara",
                        "description": "resided in the capital city",
                    },
                ),
                Node(id="Desert", type="Location" + type_suffix, properties={}),
                Node(id="Capital City", type="Location" + type_suffix, properties={}),
            ],
            relationships=[
                Relationship(
                    source=Node(
                        id="Elias Thorne", type="Person" + type_suffix, properties={}
                    ),
                    target=Node(
                        id="Desert", type="Location" + type_suffix, properties={}
                    ),
                    type="LivesIn",
                    properties={},
                ),
                Relationship(
                    source=Node(
                        id="Elias Thorne", type="Person" + type_suffix, properties={}
                    ),
                    target=Node(
                        id="Zephyr", type="Animal" + type_suffix, properties={}
                    ),
                    type="Owns",
                    properties={},
                ),
                Relationship(
                    source=Node(id="Elara", type="Person" + type_suffix, properties={}),
                    target=Node(
                        id="Capital City", type="Location" + type_suffix, properties={}
                    ),
                    type="LivesIn",
                    properties={},
                ),
                Relationship(
                    source=Node(
                        id="Elias Thorne", type="Person" + type_suffix, properties={}
                    ),
                    target=Node(id="Elara", type="Person" + type_suffix, properties={}),
                    type="Sibling",
                    properties={},
                ),
            ],
            source=Document(
                metadata={},
                page_content=(
                    "Elias Thorne lived in the desert. He was a skilled craftsman"
                    " who worked with sandstone. Elias had a pet falcon named"
                    " Zephyr. His sister, Elara, resided in the capital city and"
                    " ran a spice shop. They rarely met due to the distance."
                ),
            ),
        )
    ]

    # Add embeddings to the graph documents for Person nodes
    embedding_service = get_embedding()
    for graph_document in graph_documents:
        for node in graph_document.nodes:
            if node.type == "Person{}".format(type_suffix):
                if "description" in node.properties:
                    node.properties["desc_embedding"] = embedding_service.embed_query(
                        node.properties["description"]
                    )
    graph.add_graph_documents(graph_documents)
    graph.refresh_schema()


class TestRetriever:

    @pytest.fixture
    def setup_db_load_data(self):
        graph, suffix = get_spanner_graph()
        load_data(graph, suffix)
        yield graph, suffix
        # teardown
        graph.cleanup()

    def test_spanner_graph_gql_retriever(self, setup_db_load_data):
        graph, suffix = setup_db_load_data
        retriever = SpannerGraphGQLRetriever(
            graph_store=graph,
            llm=get_llm(),
        )
        response = retriever.invoke("Where does Elias Thorne's sibling live?")

        assert len(response) == 1
        assert "Capital City" in response[0].page_content

    def test_spanner_graph_semantic_gql_retriever(self, setup_db_load_data):
        graph, suffix = setup_db_load_data
        suffix = "_" + suffix
        retriever = SpannerGraphSemanticGQLRetriever.from_llm(
            graph_store=graph,
            llm=get_llm(),
            embedding_service=get_embedding(),
        )
        retriever.add_example(
            "Where does Sam Smith live?",
            """
        GRAPH QAGraph
        MATCH (n:Person{suffix} {{name: "Sam Smith"}})-[:LivesIn]->(l:Location{suffix})
        RETURN l.id AS location_id
    """.format(
                suffix=suffix
            ),
        )
        retriever.add_example(
            "Where does Sam Smith's sibling live?",
            """
        GRAPH QAGraph
        MATCH (n:Person{suffix} {{name: "Sam Smith"}})-[:Sibling]->(m:Person{suffix})-[:LivesIn]->(l:Location{suffix})
        RETURN l.id AS location_id
    """.format(
                suffix=suffix
            ),
        )
        response = retriever.invoke("Where does Elias Thorne's sibling live?")
        assert response == [
            Document(metadata={}, page_content='{\n  "location_id": "Capital City"\n}')
        ]

    def test_spanner_graph_vector_node_retriever_error(self, setup_db_load_data):
        with pytest.raises(ValueError):
            graph, suffix = setup_db_load_data
            suffix = "_" + suffix
            SpannerGraphNodeVectorRetriever(
                graph_store=graph,
                embedding_service=get_embedding(),
                label_expr="Person{}".format(suffix),
                embeddings_column="desc_embedding",
                k=1,
            )

    def test_spanner_graph_vector_node_retriever(self, setup_db_load_data):
        graph, suffix = setup_db_load_data
        suffix = "_" + suffix
        retriever = SpannerGraphNodeVectorRetriever(
            graph_store=graph,
            embedding_service=get_embedding(),
            label_expr="Person{}".format(suffix),
            return_properties_list=["name"],
            embeddings_column="desc_embedding",
            k=1,
        )
        response = retriever.invoke("Who lives in desert?")
        assert len(response) == 1
        assert "Elias Thorne" in response[0].page_content
