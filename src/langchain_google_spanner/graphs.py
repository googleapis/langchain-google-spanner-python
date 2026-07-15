# Copyright 2026 Google LLC
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

from typing import Any, Dict, List, Optional, Protocol, Union, runtime_checkable

from langchain_core.documents import Document
from langchain_core.load.serializable import Serializable
from pydantic import Field


class Node(Serializable):
    """Represents a node in a graph with associated properties.

    Attributes:
        id (Union[str, int]): A unique identifier for the node.
        type (str): The type or label of the node, default is "Node".
        properties (dict): Additional properties and metadata associated with the node.
    """

    id: Union[str, int]
    type: str = "Node"
    properties: dict = Field(default_factory=dict)


class Relationship(Serializable):
    """Represents a directed relationship between two nodes in a graph.

    Attributes:
        source (Node): The source node of the relationship.
        target (Node): The target node of the relationship.
        type (str): The type of the relationship.
        properties (dict): Additional properties associated with the relationship.
    """

    source: Node
    target: Node
    type: str
    properties: dict = Field(default_factory=dict)


class GraphDocument(Serializable):
    """Represents a graph document consisting of nodes and relationships.

    Attributes:
        nodes (List[Node]): A list of nodes in the graph.
        relationships (List[Relationship]): A list of relationships in the graph.
        source (Optional[Document]): The document from which the graph information is
            derived.
    """

    nodes: List[Node]
    relationships: List[Relationship]
    source: Optional[Document] = None


@runtime_checkable
class GraphStore(Protocol):
    """Abstract class for graph operations."""

    @property
    def get_schema(self) -> str:
        """Return the schema of the Graph database"""
        ...

    @property
    def get_structured_schema(self) -> Dict[str, Any]:
        """Return the schema of the Graph database"""
        ...

    def query(self, query: str, params: Optional[dict] = None) -> List[Dict[str, Any]]:
        """Query the graph."""
        ...

    def refresh_schema(self) -> None:
        """Refresh the graph schema information."""
        ...

    def add_graph_documents(
        self, graph_documents: List[GraphDocument], include_source: bool = False
    ) -> None:
        """Take GraphDocument as input as uses it to construct a graph."""
        ...
