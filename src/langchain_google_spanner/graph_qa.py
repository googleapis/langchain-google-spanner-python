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

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from langchain.chains.base import Chain
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import BasePromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.runnables import RunnableSequence
from pydantic.v1 import BaseModel, Field

from langchain_google_spanner.graph_store import SpannerGraphStore

from .prompts import (
    DEFAULT_GQL_FIX_TEMPLATE,
    DEFAULT_GQL_TEMPLATE,
    DEFAULT_GQL_VERIFY_TEMPLATE,
    SPANNERGRAPH_QA_TEMPLATE,
)

GQL_GENERATION_PROMPT = PromptTemplate(
    template=DEFAULT_GQL_TEMPLATE,
    input_variables=["question", "schema"],
)


class VerifyGqlOutput(BaseModel):
    input_gql: str
    made_change: bool
    explanation: str
    verified_gql: str


verify_gql_output_parser = JsonOutputParser(pydantic_object=VerifyGqlOutput)

GQL_VERIFY_PROMPT = PromptTemplate(
    template=DEFAULT_GQL_VERIFY_TEMPLATE,
    input_variables=["question", "generated_gql", "graph_schema"],
    partial_variables={
        "format_instructions": verify_gql_output_parser.get_format_instructions()
    },
)

GQL_FIX_PROMPT = PromptTemplate(
    template=DEFAULT_GQL_FIX_TEMPLATE,
    input_variables=["question", "generated_gql", "err_msg", "schema"],
)

SPANNERGRAPH_QA_PROMPT = PromptTemplate(
    template=SPANNERGRAPH_QA_TEMPLATE,
    input_variables=["question", "graph_schema", "graph_query", "context"],
)

INTERMEDIATE_STEPS_KEY = "intermediate_steps"


def fix_gql_syntax(query: str) -> str:
    """Fixes the syntax of a GQL query.
    Example 1:
        Input:
            MATCH (p:paper {id: 0})-[c:cites*8]->(p2:paper)
        Output:
            MATCH (p:paper {id: 0})-[c:cites]->{8}(p2:paper)
    Example 2:
        Input:
            MATCH (p:paper {id: 0})-[c:cites*1..8]->(p2:paper)
        Output:
            MATCH (p:paper {id: 0})-[c:cites]->{1:8}(p2:paper)

    Args:
        query: The input GQL query.

    Returns:
        Possibly modified GQL query.
    """

    query = re.sub(r"-\[(.*?):(\w+)\*(\d+)\.\.(\d+)\]->", r"-[\1:\2]->{\3,\4}", query)
    query = re.sub(r"-\[(.*?):(\w+)\*(\d+)\]->", r"-[\1:\2]->{\3}", query)
    query = re.sub(r"<-\[(.*?):(\w+)\*(\d+)\.\.(\d+)\]-", r"<-[\1:\2]-{\3,\4}", query)
    query = re.sub(r"<-\[(.*?):(\w+)\*(\d+)\]-", r"<-[\1:\2]-{\3}", query)
    query = re.sub(r"-\[(.*?):(\w+)\*(\d+)\.\.(\d+)\]-", r"-[\1:\2]-{\3,\4}", query)
    query = re.sub(r"-\[(.*?):(\w+)\*(\d+)\]-", r"-[\1:\2]-{\3}", query)
    return query


def extract_gql(text: str) -> str:
    """Extract GQL query from a text.

    Args:
        text: Text to extract GQL query from.

    Returns:
        GQL query extracted from the text.
    """
    pattern = r"```(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    query = matches[0] if matches else text
    return fix_gql_syntax(query)


class SpannerGraphQAChain(Chain):
    """Chain for question-answering against a Spanner Graph database by
        generating GQL statements from natural language questions.

    *Security note*: Make sure that the database connection uses credentials
        that are narrowly-scoped to only include necessary permissions.
        Failure to do so may result in data corruption or loss, since the calling
        code may attempt commands that would result in deletion, mutation
        of data if appropriately prompted or reading sensitive data if such
        data is present in the database.
        The best way to guard against such negative outcomes is to (as
        appropriate)
        limit the permissions granted to the credentials used with this tool.

        See https://python.langchain.com/docs/security for more information.
    """

    graph: SpannerGraphStore = Field(exclude=True)
    gql_generation_chain: RunnableSequence
    gql_fix_chain: RunnableSequence
    gql_verify_chain: RunnableSequence
    qa_chain: RunnableSequence
    max_gql_fix_retries: int = 1
    """ Number of retries to fix an errornous generated graph query."""
    top_k: int = 10
    """Restricts the number of results returned in the graph query."""
    return_intermediate_steps: bool = False
    """Whether to return the intermediate steps along with the final answer."""
    verify_gql: bool = True
    """Whether to have a stage in the chain to verify and fix the generated GQL."""
    input_key: str = "query"  #: :meta private:
    output_key: str = "result"  #: :meta private:
    allow_dangerous_requests: bool = False
    """Forced user opt-in to acknowledge that the chain can make dangerous requests.

    *Security note*: Make sure that the database connection uses credentials
        that are narrowly-scoped to only include necessary permissions.
        Failure to do so may result in data corruption or loss, since the calling
        code may attempt commands that would result in deletion, mutation
        of data if appropriately prompted or reading sensitive data if such
        data is present in the database.
        The best way to guard against such negative outcomes is to (as appropriate)
        limit the permissions granted to the credentials used with this tool.

        See https://python.langchain.com/docs/security for more information.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the chain."""
        super().__init__(**kwargs)
        if not self.allow_dangerous_requests:
            raise ValueError(
                "In order to use this chain, you must acknowledge that it can make "
                "dangerous requests by setting `allow_dangerous_requests` to `True`."
                "You must narrowly scope the permissions of the database connection "
                "to only include necessary permissions. Failure to do so may result "
                "in data corruption or loss or reading sensitive data if such data is "
                "present in the database. "
                "Only use this chain if you understand the risks and have taken the "
                "necessary precautions. "
                "See https://python.langchain.com/docs/security for more information."
            )

    @property
    def input_keys(self) -> List[str]:
        """Input keys.

        :meta private:
        """
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        """Output keys.

        :meta private:
        """
        return [self.output_key]

    @classmethod
    def from_llm(
        cls,
        llm: Optional[BaseLanguageModel] = None,
        *,
        qa_prompt: Optional[BasePromptTemplate] = None,
        gql_prompt: Optional[BasePromptTemplate] = None,
        gql_verify_prompt: Optional[BasePromptTemplate] = None,
        gql_fix_prompt: Optional[BasePromptTemplate] = None,
        qa_llm_kwargs: Optional[Dict[str, Any]] = None,
        gql_llm_kwargs: Optional[Dict[str, Any]] = None,
        gql_verify_llm_kwargs: Optional[Dict[str, Any]] = None,
        gql_fix_llm_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> SpannerGraphQAChain:
        """Initialize from LLM."""
        if not llm:
            raise ValueError("`llm` parameter must be provided")
        if gql_prompt and gql_llm_kwargs:
            raise ValueError(
                "Specifying gql_prompt and gql_llm_kwargs together is"
                " not allowed. Please pass prompt via gql_llm_kwargs."
            )
        if gql_fix_prompt and gql_fix_llm_kwargs:
            raise ValueError(
                "Specifying gql_fix_prompt and gql_fix_llm_kwargs together is"
                " not allowed. Please pass prompt via gql_fix_llm_kwargs."
            )
        if qa_prompt and qa_llm_kwargs:
            raise ValueError(
                "Specifying qa_prompt and qa_llm_kwargs together is"
                " not allowed. Please pass prompt via qa_llm_kwargs."
            )

        use_qa_llm_kwargs = qa_llm_kwargs if qa_llm_kwargs is not None else {}
        use_gql_llm_kwargs = gql_llm_kwargs if gql_llm_kwargs is not None else {}
        use_gql_verify_llm_kwargs = (
            gql_verify_llm_kwargs if gql_verify_llm_kwargs is not None else {}
        )
        use_gql_fix_llm_kwargs = (
            gql_fix_llm_kwargs if gql_fix_llm_kwargs is not None else {}
        )

        if "prompt" not in use_qa_llm_kwargs:
            use_qa_llm_kwargs["prompt"] = (
                qa_prompt if qa_prompt is not None else SPANNERGRAPH_QA_PROMPT
            )
        if "prompt" not in use_gql_llm_kwargs:
            use_gql_llm_kwargs["prompt"] = (
                gql_prompt if gql_prompt is not None else GQL_GENERATION_PROMPT
            )
        if "prompt" not in use_gql_verify_llm_kwargs:
            use_gql_verify_llm_kwargs["prompt"] = (
                gql_verify_prompt
                if gql_verify_prompt is not None
                else GQL_VERIFY_PROMPT
            )
        if "prompt" not in use_gql_fix_llm_kwargs:
            use_gql_fix_llm_kwargs["prompt"] = (
                gql_fix_prompt if gql_fix_prompt is not None else GQL_FIX_PROMPT
            )

        gql_generation_chain = use_gql_llm_kwargs["prompt"] | llm | StrOutputParser()
        gql_fix_chain = use_gql_fix_llm_kwargs["prompt"] | llm | StrOutputParser()
        gql_verify_chain = (
            use_gql_verify_llm_kwargs["prompt"] | llm | verify_gql_output_parser
        )
        qa_chain = use_qa_llm_kwargs["prompt"] | llm | StrOutputParser()

        return cls(
            gql_generation_chain=gql_generation_chain,
            gql_fix_chain=gql_fix_chain,
            gql_verify_chain=gql_verify_chain,
            qa_chain=qa_chain,
            **kwargs,
        )

    def execute_query(
        self, _run_manager: CallbackManagerForChainRun, gql_query: str
    ) -> List[Any]:
        try:
            _run_manager.on_text("Executing gql:", end="\n", verbose=self.verbose)
            _run_manager.on_text(
                gql_query, color="green", end="\n", verbose=self.verbose
            )
            return self.graph.query(gql_query)[: self.top_k]
        except Exception as e:
            raise ValueError(str(e))

    def execute_with_retry(
        self,
        _run_manager: CallbackManagerForChainRun,
        intermediate_steps: List,
        question: str,
        gql_query: str,
    ) -> tuple[str, List[Any]]:
        retries = 0
        while retries <= self.max_gql_fix_retries:
            try:
                intermediate_steps.append({"generated_query": gql_query})
                return gql_query, self.execute_query(_run_manager, gql_query)
            except Exception as e:
                err_msg = str(e)
                self.log_invalid_query(_run_manager, gql_query, err_msg)
                intermediate_steps.pop()
                intermediate_steps.append({"query_failed_" + str(retries): gql_query})
                fix_chain_result = self.gql_fix_chain.invoke(
                    {
                        "question": question,
                        "err_msg": err_msg,
                        "generated_gql": gql_query,
                        "schema": self.graph.get_schema,
                    }
                )
                gql_query = extract_gql(fix_chain_result)
            finally:
                retries += 1

        raise ValueError("The generated gql query is invalid")

    def log_invalid_query(
        self,
        _run_manager: CallbackManagerForChainRun,
        generated_gql: str,
        err_msg: str,
    ) -> None:
        _run_manager.on_text("Invalid generated gql:", end="\n", verbose=self.verbose)
        _run_manager.on_text(generated_gql, color="red", end="\n", verbose=self.verbose)
        _run_manager.on_text(
            "Query error: ", color="red", end="\n", verbose=self.verbose
        )
        _run_manager.on_text(err_msg, color="red", end="\n", verbose=self.verbose)

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:

        intermediate_steps: List = []

        """Generate gql statement, uses it to look up in db and answer question."""

        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        question = inputs[self.input_key]
        gen_response = self.gql_generation_chain.invoke(
            {"question": question, "schema": self.graph.get_schema},
        )
        generated_gql = extract_gql(gen_response)

        if self.verify_gql:
            verify_response = self.gql_verify_chain.invoke(
                {
                    "question": question,
                    "generated_gql": generated_gql,
                    "graph_schema": self.graph.get_schema,
                }
            )
            verified_gql = fix_gql_syntax(verify_response["verified_gql"])
            intermediate_steps.append({"verified_gql": verified_gql})
        else:
            verified_gql = generated_gql

        final_gql = ""
        if verified_gql:
            (final_gql, context) = self.execute_with_retry(
                _run_manager, intermediate_steps, question, verified_gql
            )
            if not final_gql:
                raise ValueError("No GQL was generated.")
            _run_manager.on_text("Full Context:", end="\n", verbose=self.verbose)
            _run_manager.on_text(
                str(context), color="green", end="\n", verbose=self.verbose
            )
            intermediate_steps.append({"context": context})
        else:
            context = []

        qa_result = self.qa_chain.invoke(
            {
                "question": question,
                "graph_schema": self.graph.get_schema,
                "graph_query": final_gql,
                "context": str(context),
            }
        )
        chain_result: Dict[str, Any] = {self.output_key: qa_result}
        if self.return_intermediate_steps:
            chain_result[INTERMEDIATE_STEPS_KEY] = intermediate_steps

        return chain_result
