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

import pytest
from google.cloud import spanner
from langchain.evaluation import load_evaluator
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship
from langchain_core.documents import Document
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings

from langchain_google_spanner.graph_qa import SpannerGraphQAChain
from langchain_google_spanner.graph_store import SpannerGraphStore


project_id = os.environ["PROJECT_ID"]
instance_id = os.environ["INSTANCE_ID"]
database_id = os.environ["GOOGLE_DATABASE"]


def get_llm():
    llm = ChatVertexAI(
        model="gemini-1.5-flash-002",
        temperature=0,
    )
    return llm


def get_evaluator():
    return load_evaluator(
        "embedding_distance",
        embeddings=VertexAIEmbeddings(model_name="text-embedding-004"),
    )


def get_spanner_graph():
    suffix = random_string(num_char=5, exclude_whitespaces=True)
    graph_name = "test_graph{}".format(suffix)
    graph = SpannerGraphStore(
        instance_id=instance_id,
        database_id=database_id,
        graph_name=graph_name,
        client=spanner.Client(project=project_id),
    )
    return graph


def load_data(graph: SpannerGraphStore):
    graph_documents = [
        GraphDocument(
            nodes=[
                Node(
                    id="Elias Thorne",
                    type="Person",
                    properties={
                        "name": "Elias Thorne",
                        "description": "lived in the desert",
                    },
                ),
                Node(
                    id="Zephyr",
                    type="Animal",
                    properties={"name": "Zephyr", "description": "pet falcon"},
                ),
                Node(
                    id="Elara",
                    type="Person",
                    properties={
                        "name": "Elara",
                        "description": "resided in the capital city",
                    },
                ),
                Node(id="Desert", type="Location", properties={}),
                Node(id="Capital City", type="Location", properties={}),
            ],
            relationships=[
                Relationship(
                    source=Node(id="Elias Thorne", type="Person", properties={}),
                    target=Node(id="Desert", type="Location", properties={}),
                    type="LivesIn",
                    properties={},
                ),
                Relationship(
                    source=Node(id="Elias Thorne", type="Person", properties={}),
                    target=Node(id="Zephyr", type="Animal", properties={}),
                    type="Owns",
                    properties={},
                ),
                Relationship(
                    source=Node(id="Elara", type="Person", properties={}),
                    target=Node(id="Capital City", type="Location", properties={}),
                    type="LivesIn",
                    properties={},
                ),
                Relationship(
                    source=Node(id="Elias Thorne", type="Person", properties={}),
                    target=Node(id="Elara", type="Person", properties={}),
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
    graph.add_graph_documents(graph_documents)
    graph.refresh_schema()


class TestSpannerGraphQAChain:

    @pytest.fixture
    def setup_db_load_data(self):
        graph = get_spanner_graph()
        load_data(graph)
        yield graph
        # teardown
        print(graph.get_schema)
        graph.cleanup()

    @pytest.fixture
    def chain(self, setup_db_load_data):
        graph = setup_db_load_data
        return SpannerGraphQAChain.from_llm(
            get_llm(),
            graph=graph,
            verbose=True,
            return_intermediate_steps=True,
            allow_dangerous_requests=True,
        )

    @pytest.fixture
    def chain_without_opt_in(self, setup_db_load_data):
        graph = setup_db_load_data
        return SpannerGraphQAChain.from_llm(
            get_llm(),
            graph=graph,
            verbose=True,
            return_intermediate_steps=True,
        )

    def test_spanner_graph_qa_chain_1(self, chain):
        question = "Where does Elias Thorne's sibling live?"
        response = chain.invoke("query=" + question)
        print(response)

        answer = response["result"]
        assert (
            get_evaluator().evaluate_strings(
                prediction=answer,
                reference="Elias Thorne's sibling lives in Capital City.\n",
            )["score"]
            < 0.1
        )

    def test_spanner_graph_qa_chain_no_answer(self, chain):
        question = "Where does Sarah's sibling live?"
        response = chain.invoke("query=" + question)
        print(response)

        answer = response["result"]
        assert (
            get_evaluator().evaluate_strings(
                prediction=answer,
                reference="I don't know the answer.\n",
            )["score"]
            < 0.1
        )

    def test_spanner_graph_qa_chain_without_opt_in(self, setup_db_load_data):
        with pytest.raises(ValueError):
            graph = setup_db_load_data
            SpannerGraphQAChain.from_llm(
                get_llm(),
                graph=graph,
                verbose=True,
                return_intermediate_steps=True,
            )
