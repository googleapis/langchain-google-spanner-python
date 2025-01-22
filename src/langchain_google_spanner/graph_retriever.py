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
# limitations under the License.

import json
from typing import Any, List, Optional

from langchain.schema.retriever import BaseRetriever
from langchain_community.graphs.graph_document import GraphDocument
from langchain_core.callbacks import (
    CallbackManagerForChainRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.language_models import BaseLanguageModel
from langchain_core.load import dumps
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import FewShotPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_core.vectorstores import InMemoryVectorStore
from pydantic import Field

from langchain_google_spanner.graph_store import SpannerGraphStore
from langchain_google_spanner.vector_store import DistanceStrategy, QueryParameters

from .graph_utils import extract_gql
from .prompts import DEFAULT_GQL_TEMPLATE, DEFAULT_GQL_TEMPLATE_PART1

GQL_GENERATION_PROMPT = PromptTemplate(
    template=DEFAULT_GQL_TEMPLATE,
    input_variables=["question", "schema"],
)


def convert_to_doc(data: dict[str, Any]) -> Document:
    """Converts data to a Document."""
    content = dumps(data)
    return Document(page_content=content, metadata={})


def get_graph_name_from_schema(schema: str) -> str:
    return "`" + json.loads(schema)["Name of graph"] + "`"


class SpannerGraphTextToGQLRetriever(BaseRetriever):
    """A Retriever that translates natural language queries to GQL and
    queries SpannerGraphStore using the GQL.
    Returns the documents retrieved as result.
    If examples are provided, it uses a semantic similarity model to compare the
    input question to a set of examples to generate the GQL query.
    """

    graph_store: SpannerGraphStore = Field(exclude=True)
    k: int = 10
    """Number of top results to return"""
    llm: Optional[BaseLanguageModel] = None
    selector: Optional[SemanticSimilarityExampleSelector] = None

    @classmethod
    def from_params(
        cls,
        llm: Optional[BaseLanguageModel] = None,
        embedding_service: Optional[Embeddings] = None,
        **kwargs: Any,
    ) -> "SpannerGraphTextToGQLRetriever":
        if llm is None:
            raise ValueError("`llm` cannot be none")
        selector = None
        if embedding_service is not None:
            selector = SemanticSimilarityExampleSelector.from_examples(
                [], embedding_service, InMemoryVectorStore, k=2
            )
        return cls(
            llm=llm,
            selector=selector,
            **kwargs,
        )

    @staticmethod
    def _duplicate_braces_in_string(text: str) -> str:
        """Replaces single curly braces with double curly braces in a string.

        Args:
          text: The input string.

        Returns:
          The modified string with double curly braces.
        """
        text = text.replace("{", "{{")
        text = text.replace("}", "}}")
        return text

    def add_example(self, question: str, gql: str):
        if self.selector is None:
            raise ValueError("`selector` cannot be None")
        self.selector.add_example(
            {
                "input": question,
                "query": SpannerGraphTextToGQLRetriever._duplicate_braces_in_string(
                    gql
                ),
            }
        )

    def _get_relevant_documents(
        self, question: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Translate the natural language query to GQL using examples looked up
        by a semantic similarity model, execute it, and return the results as
        Documents.
        """

        if self.llm is None:
            raise ValueError("`llm` cannot be None")

        gql_chain: RunnableSequence
        if self.selector is None:
            generic_prompt = GQL_GENERATION_PROMPT
            gql_chain = RunnableSequence(
                generic_prompt | self.llm | StrOutputParser()
            )
        else:
            # Define the prompt template
            few_shot_prompt = FewShotPromptTemplate(
                example_selector=self.selector,
                example_prompt=PromptTemplate.from_template(
                    "Question: {input}\nGQL Query: {query}"
                ),
                prefix="""
                Create an ISO GQL query for the question using the schema.""",
                suffix=DEFAULT_GQL_TEMPLATE_PART1,
                input_variables=["question", "schema"],
            )
            gql_chain = RunnableSequence (
                few_shot_prompt | self.llm | StrOutputParser()
            )

        # 1. Generate gql query from natural language query using LLM
        gql_query = extract_gql(
            gql_chain.invoke(
                {
                    "question": question,
                    "schema": self.graph_store.get_schema,
                }
            )
        )
        print(gql_query)

        # 2. Execute the gql query against spanner graph
        responses = self.graph_store.query(gql_query)[: self.k]

        # 3. Transform the results into a list of Documents
        documents = []
        for response in responses:
            documents.append(convert_to_doc(response))
        return documents


class SpannerGraphVectorContextRetriever(BaseRetriever):
    """Retriever that does a vector search on nodes in a SpannerGraphStore.
    If expand_by_hops is provided , the nodes (and edges) at a distance upto
    the expand_by hops will also be returned.
    """

    graph_store: SpannerGraphStore = Field(exclude=True)
    embedding_service: Optional[Embeddings] = None
    label_expr: str = "%"
    """A label expression for the nodes to search"""
    return_properties_list: List[str] = []
    """The list of properties to return"""
    embeddings_column: str = "embedding"
    """The name of the column that stores embedding"""
    query_parameters: QueryParameters = QueryParameters()
    top_k: int = 3
    """Number of vector similarity matches to return"""
    expand_by_hops: int = -1
    """Number of hops to traverse to expand graph results"""
    k: int = 10
    """Number of graph results to return"""

    @classmethod
    def from_params(
        cls, embedding_service: Optional[Embeddings] = None, **kwargs: Any
    ) -> "SpannerGraphVectorContextRetriever":
        if embedding_service is None:
            raise ValueError("`embedding_service` cannot be None")
        return cls(
            embedding_service=embedding_service,
            **kwargs,
        )

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if self.embedding_service is None:
            raise ValueError("`embedding_service` cannot be None")

        sum = 0
        if self.return_properties_list:
            sum += 1
        if self.expand_by_hops != -1:
            sum += 1
        if sum != 1:
            raise ValueError(
                "One and only one of `return_properties` or `expand_by_hops` must be provided."
            )

    @staticmethod
    def _clean_element(element: dict[str, Any], embedding_column: str) -> None:
        """Removes specified keys and embedding from properties in graph element.

        Args:
          element: A dictionary representing element
        """

        keys_to_remove = [
            "source_node_identifier",
            "destination_node_identifier",
            "identifier",
        ]
        for key in keys_to_remove:
            if key in element:
                del element[key]

        if "properties" in element and embedding_column in element["properties"]:
            del element["properties"][embedding_column]

    @staticmethod
    def _get_distance_function(distance_strategy=DistanceStrategy.EUCLIDEIAN) -> str:
        """Gets the vector distance function."""
        if distance_strategy == DistanceStrategy.COSINE:
            return "COSINE_DISTANCE"

        return "EUCLIDEAN_DISTANCE"

    def _get_relevant_documents(
        self, question: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Translate the natural language query to GQL, execute it,
        and return the results as Documents."""

        schema = self.graph_store.get_schema
        graph_name = get_graph_name_from_schema(schema)
        node_variable = "node"

        if self.embedding_service is None:
            raise ValueError("`embedding_service` cannot be None")
        query_embeddings = self.embedding_service.embed_query(question)

        distance_fn = SpannerGraphVectorContextRetriever._get_distance_function(
            self.query_parameters.distance_strategy
        )

        VECTOR_QUERY = """
            GRAPH {graph_name}
            MATCH ({node_var}:{label_expr})
            WHERE {node_var}.{embeddings_column} IS NOT NULL
            ORDER BY {distance_fn}({node_var}.{embeddings_column},
                    ARRAY[{query_embeddings}])
            LIMIT {top_k}
        """
        gql_query = VECTOR_QUERY.format(
            graph_name=graph_name,
            node_var=node_variable,
            label_expr=self.label_expr,
            embeddings_column=self.embeddings_column,
            distance_fn=distance_fn,
            query_embeddings=",".join(map(str, query_embeddings)),
            top_k=self.top_k,
        )

        if self.expand_by_hops >= 0:
            gql_query += """
          RETURN node
          NEXT
          MATCH p = TRAIL (node) -[]-{{0,{}}} ()
          RETURN SAFE_TO_JSON(p) as path
          """.format(
                self.expand_by_hops
            )
        elif self.return_properties_list:
            return_properties = ",".join(
                map(
                    lambda x: node_variable + ".`" + x + "`",
                    self.return_properties_list,
                )
            )
            gql_query += """
      RETURN {}
      """.format(
                return_properties
            )
        else:
            raise ValueError(
                "Either `return_properties` or `expand_by_hops` must be provided."
            )

        print(gql_query)

        # 2. Execute the gql query against spanner graph
        responses = self.graph_store.query(gql_query)[: self.k]

        # 3. Transform the results into a list of Documents
        documents = []
        if self.expand_by_hops >= 0:
            for response in responses:
                elements = json.loads((response["path"]).serialize())
                for element in elements:
                    SpannerGraphVectorContextRetriever._clean_element(
                        element, self.embeddings_column
                    )
                response["path"] = elements
                content = dumps(response["path"])
                documents.append(Document(page_content=content, metadata={}))

        else:
            for response in responses:
                documents.append(convert_to_doc(response))

        return documents
