import json
from typing import Any, List

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


def graph_doc_to_doc(graph_doc: GraphDocument) -> Document:
    """Converts a GraphDocument to a Document."""
    content = dumps(graph_doc, pretty=True)
    return Document(page_content=content, metadata={})


def get_distance_function(distance_strategy=DistanceStrategy.EUCLIDEIAN) -> str:
    """Gets the vector distance function."""
    if distance_strategy == DistanceStrategy.COSINE:
        return "COSINE_DISTANCE"

    return "EUCLIDEAN_DISTANCE"


def get_graph_name_from_schema(schema: str):
    return json.loads(schema)["Name of graph"]


def duplicate_braces_in_string(text):
    """Replaces single curly braces with double curly braces in a string.

    Args:
      text: The input string.

    Returns:
      The modified string with double curly braces.
    """
    text = text.replace("{", "{{")
    text = text.replace("}", "}}")
    return text


class SpannerGraphGQLRetriever(BaseRetriever):
    """A Retriever that translates natural language queries to GQL and
    queries SpannerGraphStore using the GQL.
    Returns the documents retrieved as result.
    """

    graph_store: SpannerGraphStore = Field(exclude=True)
    llm: BaseLanguageModel = None
    k: int = 10
    """Number of top results to return"""

    def _get_relevant_documents(
        self, question: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Translate the natural language query to GQL, execute it,
        and return the results as Documents.
        """

        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        # Initialize the chain
        gql_chain = GQL_GENERATION_PROMPT | self.llm | StrOutputParser()
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
        try:
            graph_documents = self.graph_store.query(gql_query)[: self.k]
        except Exception as e:
            raise ValueError(str(e))

        # 3. Transform the results into a list of Documents
        documents = []
        for graph_document in graph_documents:
            documents.append(graph_doc_to_doc(graph_document))
        return documents


class SpannerGraphSemanticGQLRetriever(BaseRetriever):
    """A Retriever that translates natural language queries to GQL and
    and queries SpannerGraphStore to retrieve documents. It uses a semantic
    similarity model to compare the input question to a set of examples to
    generate the GQL query.
    """

    graph_store: SpannerGraphStore = Field(exclude=True)
    llm: BaseLanguageModel = None
    k: int = 10
    """Number of top results to return"""
    selector: SemanticSimilarityExampleSelector = None

    @classmethod
    def from_llm(
        cls, embedding_service: Embeddings = None, **kwargs: Any
    ) -> "SpannerGraphSemanticGQLRetriever":
        selector = SemanticSimilarityExampleSelector.from_examples(
            [], embedding_service, InMemoryVectorStore(embedding_service), k=2
        )
        return cls(
            selector=selector,
            **kwargs,
        )

    def add_example(self, question: str, gql: str):
        self.selector.add_example(
            {"input": question, "query": duplicate_braces_in_string(gql)}
        )

    def _get_relevant_documents(
        self, question: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Translate the natural language query to GQL using examples looked up
        by a semantic similarity model, execute it, and return the results as
        Documents.
        """

        # Define the prompt template
        prompt = FewShotPromptTemplate(
            example_selector=self.selector,
            example_prompt=PromptTemplate.from_template(
                "Question: {input}\nGQL Query: {query}"
            ),
            prefix="""
            Create an ISO GQL query for the question using the schema.""",
            suffix=DEFAULT_GQL_TEMPLATE_PART1,
            input_variables=["question", "schema"],
        )

        # Initialize the chain
        gql_chain = prompt | self.llm | StrOutputParser()
        # 1. Generate gql query from natural language query using LLM
        gql_query = extract_gql(
            gql_chain.invoke(
                {
                    "question": question,
                    "schema": self.graph_store.get_schema,
                }
            )
        )
        print(gql_query)  # TODO(amullick): REMOVE

        # 2. Execute the gql query against spanner graph
        try:
            graph_documents = self.graph_store.query(gql_query)[: self.k]
        except Exception as e:
            raise ValueError(str(e))

        # 3. Transform the results into a list of Documents
        documents = []
        for graph_document in graph_documents:
            documents.append(graph_doc_to_doc(graph_document))
        return documents


class SpannerGraphNodeVectorRetriever(BaseRetriever):
    """Retriever that does a vector search on nodes in a SpannerGraphStore.
    If a graph expansion query is provided, it will be executed after the
    initial vector search to expand the returned context.
    """

    graph_store: SpannerGraphStore = Field(exclude=True)
    embedding_service: Embeddings
    label_expr: str = "%"
    return_properties_list: List[str] = []
    embeddings_column: str = "embedding"
    query_parameters: QueryParameters = QueryParameters()
    k: int = 10
    """Number of top results to return"""
    graph_expansion_query: str = None
    """GQL query to expand the returned context"""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        print(self.return_properties_list)
        print(self.graph_expansion_query)
        if not self.return_properties_list and self.graph_expansion_query is None:
            raise ValueError(
                "Either `return_properties` or `graph_expansion_query` must be provided."
            )

    def _get_relevant_documents(
        self, question: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Translate the natural language query to GQL, execute it,
        and return the results as Documents."""

        schema = self.graph_store.get_schema
        graph_name = get_graph_name_from_schema(schema)
        node_variable = "node"
        query_embeddings = self.embedding_service.embed_query(question)

        distance_fn = get_distance_function(self.query_parameters.distance_strategy)

        VECTOR_QUERY = """
            GRAPH {graph_name}
            MATCH ({node_var}:{label_expr})
            ORDER BY {distance_fn}({node_var}.{embeddings_column},
                    ARRAY[{query_embeddings}])
            LIMIT {k}
        """
        gql_query = VECTOR_QUERY.format(
            graph_name=graph_name,
            node_var=node_variable,
            label_expr=self.label_expr,
            embeddings_column=self.embeddings_column,
            distance_fn=distance_fn,
            query_embeddings=",".join(map(str, query_embeddings)),
            k=self.k,
        )

        if self.return_properties_list:
            return_properties = ",".join(
                map(lambda x: node_variable + "." + x, self.return_properties_list)
            )
            gql_query += """
      RETURN {}
      """.format(
                return_properties
            )
        elif self.graph_expansion_query is not None:
            gql_query += """
      RETURN node
      NEXT
      {}
      """.format(
                self.graph_expansion_query
            )
        else:
            raise ValueError(
                "Either `return_properties` or `graph_expansion_query` must be provided."
            )

        print(gql_query)

        # 2. Execute the gql query against spanner graph
        try:
            graph_documents = self.graph_store.query(gql_query)[: self.k]
        except Exception as e:
            raise ValueError(str(e))

        # 3. Transform the results into a list of Documents
        documents = []
        for graph_document in graph_documents:
            documents.append(graph_doc_to_doc(graph_document))
        return documents
