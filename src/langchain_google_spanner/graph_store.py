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

import json
import re
import string
from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, Iterable, List, Mapping, Optional, Tuple, Union

from google.cloud import spanner
from google.cloud.spanner_v1 import JsonObject, param_types
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship
from langchain_community.graphs.graph_store import GraphStore
from requests.structures import CaseInsensitiveDict

from .type_utils import TypeUtility
from .version import __version__

MUTATION_BATCH_SIZE = 1000
DEFAULT_DDL_TIMEOUT = 300
NODE_KIND = "NODE"
EDGE_KIND = "EDGE"
USER_AGENT_GRAPH_STORE = "langchain-google-spanner-python:graphstore/" + __version__


class NodeWrapper(object):
    """Wrapper around Node to support set operations using node id"""

    def __init__(self, node: Node):
        self.node = node

    def __hash__(self):
        return hash(self.node.id)

    def __eq__(self, other: Any):
        if isinstance(other, NodeWrapper):
            return self.node.id == other.node.id
        return False


class EdgeWrapper(object):
    """Wrapper around Relationship to support set operations using edge source and target id"""

    def __init__(self, edge: Relationship):
        self.edge = edge

    def __hash__(self):
        return hash(
            (
                self.edge.source.id,
                self.edge.target.id,
                self.edge.type,
            )
        )

    def __eq__(self, other: Any):
        if isinstance(other, EdgeWrapper):
            return (
                self.edge.source.id == other.edge.source.id
                and self.edge.target.id == other.edge.target.id
                and self.edge.type == other.edge.type
            )
        return False


def partition_graph_docs(
    graph_documents: List[GraphDocument],
) -> Tuple[dict, dict]:
    """Returns nodes and edges grouped by the type.

    Args:
      graph_documents: List[GraphDocument].

    Returns:
      A tuple of two dictionaries. The first is nodes grouped by node types,
      the second is edges grouped by edge types.
    """
    nodes: CaseInsensitiveDict[dict[NodeWrapper, Node]] = CaseInsensitiveDict()
    edges: CaseInsensitiveDict[dict[EdgeWrapper, Relationship]] = CaseInsensitiveDict()
    for doc in graph_documents:
        for node in doc.nodes:
            ns = nodes.setdefault(node.type, dict())
            nw = NodeWrapper(node)
            if nw in ns:
                # Combine the properties for nodes with the same id.
                ns[nw].properties.update(node.properties)
            else:
                ns[nw] = node

        for edge in doc.relationships:
            # Partition edges by the triplet because there could be edges with the
            # same type but between different types of nodes.
            edge_name = "{}_{}_{}".format(edge.source.type, edge.type, edge.target.type)
            es = edges.setdefault(edge_name, dict())
            ew = EdgeWrapper(edge)
            if ew in es:
                # Combine the properties for edges with the same id.
                es[ew].properties.update(edge.properties)
            else:
                es[ew] = edge
    return {name: [n for _, n in ns.items()] for name, ns in nodes.items()}, {
        name: [e for _, e in es.items()] for name, es in edges.items()
    }


def client_with_user_agent(
    client: Optional[spanner.Client], user_agent: str
) -> spanner.Client:
    if not client:
        client = spanner.Client()
    client_agent = client._client_info.user_agent
    if not client_agent:
        client._client_info.user_agent = user_agent
    elif user_agent not in client_agent:
        client._client_info.user_agent = " ".join([client_agent, user_agent])
    return client


class GraphDocumentUtility:
    """Utilities to process graph documents."""

    @staticmethod
    def is_valid_identifier(s: str) -> bool:
        return re.match(r"^[a-z][a-z0-9_]{0,127}$", s, re.IGNORECASE) is not None

    @staticmethod
    def to_identifier(s: str) -> str:
        return "`" + s + "`"

    @staticmethod
    def to_identifiers(s: List[str]) -> Iterable[str]:
        return map(GraphDocumentUtility.to_identifier, s)

    @staticmethod
    def fixup_identifier(s: str) -> str:
        return re.sub("[{}]".format(string.whitespace + string.punctuation), "_", s)

    @staticmethod
    def fixup_graph_documents(graph_documents: List[GraphDocument]) -> None:
        """Preprocess graph documents.

        Args:
          graph_documents: List[GraphDocument].
        """
        for graph_document in graph_documents:
            for node in graph_document.nodes:
                GraphDocumentUtility.fixup_element(node)
            for edge in graph_document.relationships:
                GraphDocumentUtility.fixup_element(edge)

    @staticmethod
    def fixup_element(element: Union[Node, Relationship]) -> None:
        """Preprocess graph element.

        Args:
          element: Node or Relationship.
        """
        element.type = GraphDocumentUtility.fixup_identifier(element.type)
        should_ignore = lambda v: v is None or (isinstance(v, list) and len(v) == 0)
        element.properties = {
            k: v for k, v in element.properties.items() if not should_ignore(v)
        }

        if isinstance(element, Relationship):
            element.source.type = GraphDocumentUtility.fixup_identifier(
                element.source.type
            )
            element.target.type = GraphDocumentUtility.fixup_identifier(
                element.target.type
            )


class ElementSchema(object):
    """Schema representation of a node or an edge."""

    NODE_KEY_COLUMN_NAME: str = "id"
    TARGET_NODE_KEY_COLUMN_NAME: str = "target_id"

    # Reserved column names when `use_flexible_schema` is true.
    # Properties are stored in a JSON column named `properties`;
    # Edge types are stored in a string column named `label`.
    DYNAMIC_PROPERTY_COLUMN_NAME: str = "properties"
    DYNAMIC_LABEL_COLUMN_NAME: str = "label"

    name: str
    kind: str
    key_columns: List[str]
    base_table_name: str
    labels: List[str]
    properties: CaseInsensitiveDict[str]
    types: CaseInsensitiveDict[param_types.Type]
    source: NodeReference
    target: NodeReference

    def is_dynamic_schema(self) -> bool:
        return (
            self.types.get(ElementSchema.DYNAMIC_PROPERTY_COLUMN_NAME, None)
            == param_types.JSON
        )

    @staticmethod
    def make_node_schema(
        node_type: str,
        node_label: str,
        graph_name: str,
        property_types: CaseInsensitiveDict,
    ) -> ElementSchema:
        node = ElementSchema()
        node.types = property_types
        node.properties = CaseInsensitiveDict({prop: prop for prop in node.types})
        node.labels = [node_label]
        node.base_table_name = "%s_%s" % (graph_name, node_label)
        node.name = node_type
        node.kind = NODE_KIND
        node.key_columns = [ElementSchema.NODE_KEY_COLUMN_NAME]
        return node

    @staticmethod
    def make_edge_schema(
        edge_type: str,
        edge_label: str,
        graph_schema: SpannerGraphSchema,
        key_columns: List[str],
        property_types: CaseInsensitiveDict,
        source_node_type: str,
        target_node_type: str,
    ) -> ElementSchema:
        edge = ElementSchema()
        edge.types = property_types
        edge.properties = CaseInsensitiveDict({prop: prop for prop in edge.types})

        edge.labels = [edge_label]
        edge.base_table_name = "%s_%s" % (graph_schema.graph_name, edge_label)
        edge.key_columns = key_columns
        edge.name = edge_type
        edge.kind = EDGE_KIND

        source_node_schema = graph_schema.get_node_schema(
            graph_schema.node_type_name(source_node_type)
        )
        if source_node_schema is None:
            raise ValueError("No source node schema `%s` found" % source_node_type)

        target_node_schema = graph_schema.get_node_schema(
            graph_schema.node_type_name(target_node_type)
        )
        if target_node_schema is None:
            raise ValueError("No target node schema `%s` found" % target_node_type)

        edge.source = NodeReference(
            source_node_schema.name,
            [ElementSchema.NODE_KEY_COLUMN_NAME],
            [ElementSchema.NODE_KEY_COLUMN_NAME],
        )
        edge.target = NodeReference(
            target_node_schema.name,
            [ElementSchema.NODE_KEY_COLUMN_NAME],
            [ElementSchema.TARGET_NODE_KEY_COLUMN_NAME],
        )
        return edge

    @staticmethod
    def from_static_nodes(
        name: str, nodes: List[Node], graph_schema: SpannerGraphSchema
    ) -> ElementSchema:
        """Builds ElementSchema from a list of nodes.

        Args:
          name: name of the schema.
          nodes: a non-empty list of nodes.
          graph_schema: schema of the graph.

        Returns:
          ElementSchema: schema representation of the nodes.

        Raises:
          ValueError: An error occured building element schema.
        """
        if len(nodes) == 0:
            raise ValueError("The list of nodes should not be empty")

        types = CaseInsensitiveDict(
            {
                k: TypeUtility.value_to_param_type(v)
                for n in nodes
                for k, v in n.properties.items()
            }
        )
        if ElementSchema.NODE_KEY_COLUMN_NAME in types:
            raise ValueError(
                "Node properties should not contain property with name: `%s`"
                % ElementSchema.NODE_KEY_COLUMN_NAME
            )
        types[ElementSchema.NODE_KEY_COLUMN_NAME] = TypeUtility.value_to_param_type(
            nodes[0].id
        )
        return ElementSchema.make_node_schema(
            name, name, graph_schema.graph_name, types
        )

    @staticmethod
    def from_dynamic_nodes(
        name: str, nodes: List[Node], graph_schema: SpannerGraphSchema
    ) -> ElementSchema:
        """Builds ElementSchema from a list of nodes.

        Args:
          name: name of the schema;
          nodes: a non-empty list of nodes.
          graph_schema: schema of the graph;

        Returns:
          ElementSchema: schema representation of the nodes.

        Raises:
          ValueError: An error occured building element schema.
        """
        if len(nodes) == 0:
            raise ValueError("The list of nodes should not be empty")

        types = CaseInsensitiveDict(
            {
                ElementSchema.DYNAMIC_PROPERTY_COLUMN_NAME: param_types.JSON,
                ElementSchema.DYNAMIC_LABEL_COLUMN_NAME: param_types.STRING,
                ElementSchema.NODE_KEY_COLUMN_NAME: TypeUtility.value_to_param_type(
                    nodes[0].id
                ),
            }
        )
        types.update(
            CaseInsensitiveDict(
                {
                    k: TypeUtility.value_to_param_type(v)
                    for n in nodes
                    for k, v in n.properties.items()
                    if k in graph_schema.static_node_properties
                }
            )
        )
        return ElementSchema.make_node_schema(
            NODE_KIND, NODE_KIND, graph_schema.graph_name, types
        )

    @staticmethod
    def from_static_edges(
        name: str,
        edges: List[Relationship],
        graph_schema: SpannerGraphSchema,
    ) -> ElementSchema:
        """Builds ElementSchema from a list of edges.

        Args:
          name: name of the schema;
          edges: a non-empty list of edges.
          graph_schema: schema of the graph;

        Returns:
          ElementSchema: schema representation of the edges.

        Raises:
          ValueError: An error occured building element schema.
        """
        if len(edges) == 0:
            raise ValueError("The list of edges should not be empty")

        types = CaseInsensitiveDict(
            {
                k: TypeUtility.value_to_param_type(v)
                for e in edges
                for k, v in e.properties.items()
            }
        )

        for col_name in [
            ElementSchema.NODE_KEY_COLUMN_NAME,
            ElementSchema.TARGET_NODE_KEY_COLUMN_NAME,
        ]:
            if col_name in types:
                raise ValueError(
                    "Edge properties should not contain property with name: `%s`"
                    % col_name
                )
        types[ElementSchema.NODE_KEY_COLUMN_NAME] = TypeUtility.value_to_param_type(
            edges[0].source.id
        )
        types[ElementSchema.TARGET_NODE_KEY_COLUMN_NAME] = (
            TypeUtility.value_to_param_type(edges[0].target.id)
        )
        return ElementSchema.make_edge_schema(
            name,
            name,
            graph_schema,
            [
                ElementSchema.NODE_KEY_COLUMN_NAME,
                ElementSchema.TARGET_NODE_KEY_COLUMN_NAME,
            ],
            types,
            edges[0].source.type,
            edges[0].target.type,
        )

    @staticmethod
    def from_dynamic_edges(
        name: str,
        edges: List[Relationship],
        graph_schema: SpannerGraphSchema,
    ) -> ElementSchema:
        """Builds ElementSchema from a list of edges.

        Args:
          name: name of the schema.
          edges: a non-empty list of edges.
          graph_schema: schema of the graph.

        Returns:
          ElementSchema: schema representation of the edges.

        Raises:
          ValueError: An error occured building element schema.
        """
        if len(edges) == 0:
            raise ValueError("The list of edges should not be empty")

        types = CaseInsensitiveDict(
            {
                ElementSchema.DYNAMIC_PROPERTY_COLUMN_NAME: param_types.JSON,
                ElementSchema.DYNAMIC_LABEL_COLUMN_NAME: param_types.STRING,
                ElementSchema.NODE_KEY_COLUMN_NAME: TypeUtility.value_to_param_type(
                    edges[0].source.id
                ),
                ElementSchema.TARGET_NODE_KEY_COLUMN_NAME: TypeUtility.value_to_param_type(
                    edges[0].target.id
                ),
            }
        )
        types.update(
            CaseInsensitiveDict(
                {
                    k: TypeUtility.value_to_param_type(v)
                    for e in edges
                    for k, v in e.properties.items()
                    if k in graph_schema.static_edge_properties
                }
            )
        )
        return ElementSchema.make_edge_schema(
            EDGE_KIND,
            EDGE_KIND,
            graph_schema,
            [
                ElementSchema.NODE_KEY_COLUMN_NAME,
                ElementSchema.TARGET_NODE_KEY_COLUMN_NAME,
                ElementSchema.DYNAMIC_LABEL_COLUMN_NAME,
            ],
            types,
            edges[0].source.type,
            edges[0].target.type,
        )

    def add_nodes(
        self, name: str, nodes: List[Node]
    ) -> Generator[Tuple[str, Tuple[str], List[List[Any]]], None, None]:
        """Builds the data required to add a list of nodes to Spanner.

        Args:
          name: type of name;
          nodes: a list of Nodes.

        Returns:
          An iterator that yields a tuple consists of the following:
            str: a table name;
            List[str]: a list of column names;
            List[List[Any]]: a list of rows.

        Raises:
          ValueError: An error occured adding nodes.
        """
        if len(nodes) == 0:
            raise ValueError("Empty list of nodes")

        # Group changes by columns: this avoids overwriting columns that aren't
        # specified.
        rows_by_columns: Dict[Tuple[str], List[List[Any]]] = {}
        for node in nodes:
            properties = node.properties.copy()
            properties[ElementSchema.NODE_KEY_COLUMN_NAME] = node.id

            if self.is_dynamic_schema():
                dynamic_properties = {
                    k: TypeUtility.value_for_json(v)
                    for k, v in node.properties.items()
                    if k not in self.types
                }
                if dynamic_properties:
                    # Json loads and dumps handles some invalid characters
                    # that the JsonDecoder doesn't accept.
                    properties[ElementSchema.DYNAMIC_PROPERTY_COLUMN_NAME] = JsonObject(
                        json.loads(json.dumps(dynamic_properties))
                    )
                properties[ElementSchema.DYNAMIC_LABEL_COLUMN_NAME] = node.type

            columns = tuple(sorted((k for k in properties if k in self.types)))
            row = [properties[k] for k in columns]
            rows_by_columns.setdefault(columns, []).append(row)

        for columns, rows in rows_by_columns.items():
            yield self.base_table_name, columns, rows

    def add_edges(
        self, name: str, edges: List[Relationship]
    ) -> Generator[Tuple[str, Tuple[str], List[List[Any]]], None, None]:
        """Builds the data required to add a list of edges to Spanner.

        Args:
          name: type of edge;
          edges: a list of Relationships.

        Returns:
          An iterator that yields a tuple consists of the following:
            str: a table name;
            List[str]: a list of column names;
            List[List[Any]]: a list of rows.

        Raises:
          ValueError: An error occured adding edges.
        """
        if len(edges) == 0:
            raise ValueError("Empty list of edges")

        # Group changes by columns: this avoids overwriting columns that aren't
        # specified.
        rows_by_columns: Dict[Tuple[str], List[List[Any]]] = {}
        for edge in edges:
            properties = edge.properties.copy()
            properties[ElementSchema.NODE_KEY_COLUMN_NAME] = edge.source.id
            properties[ElementSchema.TARGET_NODE_KEY_COLUMN_NAME] = edge.target.id

            if self.is_dynamic_schema():
                dynamic_properties = {
                    k: TypeUtility.value_for_json(v)
                    for k, v in edge.properties.items()
                    if k not in self.types
                }
                if dynamic_properties:
                    # Json loads and dumps handles some invalid characters
                    # that the JsonDecoder doesn't accept.
                    properties[ElementSchema.DYNAMIC_PROPERTY_COLUMN_NAME] = JsonObject(
                        json.loads(json.dumps(dynamic_properties))
                    )
                properties[ElementSchema.DYNAMIC_LABEL_COLUMN_NAME] = edge.type

            columns = tuple(sorted((k for k in properties if k in self.types)))
            row = [properties[k] for k in columns]
            rows_by_columns.setdefault(columns, []).append(row)

        for columns, rows in rows_by_columns.items():
            yield self.base_table_name, columns, rows

    @staticmethod
    def from_info_schema(
        element_schema: Dict[str, Any],
        decl_by_types: CaseInsensitiveDict,
    ) -> ElementSchema:
        """Builds ElementSchema from information schema represenation of an element.

        Args:
          element_schema: the information schema represenation of an element;
          decl_by_types: type information of property declarations.

        Returns:
          ElementSchema

        Raises:
          ValueError: An error occured building graph schema.
        """
        element = ElementSchema()
        element.name = element_schema["name"]
        element.kind = element_schema["kind"]
        if element.kind not in [NODE_KIND, EDGE_KIND]:
            raise ValueError("Invalid element kind `{}`".format(element.kind))

        element.key_columns = element_schema["keyColumns"]
        element.base_table_name = element_schema["baseTableName"]
        element.labels = element_schema["labelNames"]

        element.properties = CaseInsensitiveDict(
            {
                prop_def["propertyDeclarationName"]: prop_def["valueExpressionSql"]
                for prop_def in element_schema.get("propertyDefinitions", [])
                if prop_def["propertyDeclarationName"] in decl_by_types
            }
        )
        element.types = CaseInsensitiveDict(
            {decl: decl_by_types[decl] for decl in element.properties.keys()}
        )

        if element.kind == EDGE_KIND:
            element.source = NodeReference(
                element_schema["sourceNodeTable"]["nodeTableName"],
                element_schema["sourceNodeTable"]["nodeTableColumns"],
                element_schema["sourceNodeTable"]["edgeTableColumns"],
            )
            element.target = NodeReference(
                element_schema["destinationNodeTable"]["nodeTableName"],
                element_schema["destinationNodeTable"]["nodeTableColumns"],
                element_schema["destinationNodeTable"]["edgeTableColumns"],
            )
        return element

    def to_ddl(self, graph_schema: SpannerGraphSchema) -> str:
        """Returns a CREATE TABLE ddl that represents the element schema.

        Args:
          graph_schema: Spanner Graph schema.

        Returns:
          str: a string of CREATE TABLE ddl statement.

        Raises:
          ValueError: An error occured building graph ddl.
        """

        to_identifier = GraphDocumentUtility.to_identifier
        to_identifiers = GraphDocumentUtility.to_identifiers

        def get_reference_node_table(name: str) -> str:
            node_schema = graph_schema.nodes.get(name, None)
            if node_schema is None:
                raise ValueError("No node schema `%s` found" % name)
            return node_schema.base_table_name

        return """CREATE TABLE {} (
          {}{}
        ) PRIMARY KEY ({}){}
      """.format(
            to_identifier(self.base_table_name),
            ",\n                ".join(
                (
                    "{} {}".format(
                        to_identifier(n),
                        TypeUtility.spanner_type_to_schema_str(
                            t, include_type_annotations=True
                        ),
                    )
                    for n, t in self.types.items()
                )
            ),
            (
                ",\n                FOREIGN KEY ({}) REFERENCES {}({})".format(
                    ", ".join(to_identifiers(self.target.edge_keys)),
                    to_identifier(get_reference_node_table(self.target.node_name)),
                    ", ".join(to_identifiers(self.target.node_keys)),
                )
                if self.kind == EDGE_KIND
                else ""
            ),
            ",".join(to_identifiers(self.key_columns)),
            (
                ", INTERLEAVE IN PARENT {}".format(
                    to_identifier(get_reference_node_table(self.source.node_name))
                )
                if self.kind == EDGE_KIND
                else ""
            ),
        )

    def evolve(self, new_schema: ElementSchema) -> List[str]:
        """Evolves current schema from the new schema.

        Args:
          new_schema: an ElementSchema representing new nodes/edges.

        Returns:
          List[str]: a list of DDL statements.

        Raises:
          ValueError: An error occured evolving graph schema.
        """
        if self.kind != new_schema.kind:
            raise ValueError(
                "Schema with name `{}` should have the same kind, got {}, expected {}".format(
                    self.name, new_schema.kind, self.kind
                )
            )
        if self.key_columns != new_schema.key_columns:
            raise ValueError(
                "Schema with name `{}` should have the same keys, got {}, expected {}".format(
                    self.name, new_schema.key_columns, self.key_columns
                )
            )
        if self.base_table_name.casefold() != new_schema.base_table_name.casefold():
            raise ValueError(
                "Schema with name `{}` should have the same base table name, got {},"
                " expected {}".format(
                    self.name, new_schema.base_table_name, self.base_table_name
                )
            )

        # Only validate property definition when they're the same definition,
        # don't validate when two different definitions are based on the same
        # underlying table.
        if self.name == new_schema.name:
            for k, v in new_schema.properties.items():
                if k in self.properties:
                    if self.properties[k].casefold() != v.casefold():
                        raise ValueError(
                            "Property with name `{}` should have the same definition, got {},"
                            " expected {}".format(k, v, self.properties[k])
                        )

        for k, v in new_schema.types.items():
            if k in self.types:
                if self.types[k] != v:
                    raise ValueError(
                        "Property with name `{}` should have the same type, got {},"
                        " expected {}".format(k, v, self.types[k])
                    )

        to_identifier = GraphDocumentUtility.to_identifier
        ddls = [
            "ALTER TABLE {} ADD COLUMN {} {}".format(
                to_identifier(self.base_table_name),
                to_identifier(n),
                TypeUtility.spanner_type_to_schema_str(
                    t, include_type_annotations=True
                ),
            )
            for n, t in new_schema.types.items()
            if n not in self.properties
        ]
        self.properties.update(new_schema.properties)
        self.types.update(new_schema.types)
        return ddls


class Label(object):
    """Schema representation of a label."""

    def __init__(self, name: str, prop_names: set[str]):
        self.name = name
        self.prop_names = prop_names


class NodeReference(object):
    """Schema representation of a source or destination node reference."""

    def __init__(self, node_name: str, node_keys: List[str], edge_keys: List[str]):
        self.node_name = node_name
        self.node_keys = node_keys
        self.edge_keys = edge_keys


class SpannerGraphSchema(object):
    """Schema representation of a property graph."""

    GRAPH_INFORMATION_SCHEMA_QUERY_TEMPLATE = """
  SELECT property_graph_metadata_json
  FROM INFORMATION_SCHEMA.PROPERTY_GRAPHS
  WHERE property_graph_name = '{}'
  """

    def __init__(
        self,
        graph_name: str,
        use_flexible_schema: bool,
        static_node_properties: List[str] = [],
        static_edge_properties: List[str] = [],
    ):
        """Initializes the graph schema.

        Args:
          graph_name: the name of the graph;
          use_flexible_schema: whether to use the flexible schema which uses a
            JSON blob to store node and edge properties;
          static_node_properties: in flexible schema, treat these node
            properties as static;
          static_edge_properties: in flexible schema, treat these edge
            properties as static.

        Raises:
          ValueError: An error occured initializing graph schema.
        """
        if not GraphDocumentUtility.is_valid_identifier(graph_name):
            raise ValueError(
                "Graph name `{}` is not a valid identifier".format(graph_name)
            )

        self.graph_name: str = graph_name
        self.nodes: CaseInsensitiveDict[ElementSchema] = CaseInsensitiveDict({})
        self.edges: CaseInsensitiveDict[ElementSchema] = CaseInsensitiveDict({})
        self.node_tables: CaseInsensitiveDict[ElementSchema] = CaseInsensitiveDict({})
        self.edge_tables: CaseInsensitiveDict[ElementSchema] = CaseInsensitiveDict({})
        self.labels: CaseInsensitiveDict[Label] = CaseInsensitiveDict({})
        self.properties: CaseInsensitiveDict[param_types.Type] = CaseInsensitiveDict({})
        self.use_flexible_schema = use_flexible_schema
        self.static_node_properties = set(static_node_properties)
        self.static_edge_properties = set(static_edge_properties)

    def evolve(self, graph_documents: List[GraphDocument]) -> List[str]:
        """Evolves current schema into a schema representing the input documents.

        Args:
          graph_documents: a list of GraphDocument.

        Returns:
          List[str]: a list of DDL statements.
        """
        nodes, edges = partition_graph_docs(graph_documents)

        ddls = []
        for k, ns in nodes.items():
            node_schema = (
                ElementSchema.from_static_nodes(k, ns, self)
                if not self.use_flexible_schema
                else ElementSchema.from_dynamic_nodes(k, ns, self)
            )
            ddls.extend(self._update_node_schema(node_schema))
            self._update_labels_and_properties(node_schema)

        for k, es in edges.items():
            edge_schema = (
                ElementSchema.from_static_edges(k, es, self)
                if not self.use_flexible_schema
                else ElementSchema.from_dynamic_edges(k, es, self)
            )
            ddls.extend(self._update_edge_schema(edge_schema))
            self._update_labels_and_properties(edge_schema)

        if len(ddls) != 0:
            ddls += [self.to_ddl()]
        return ddls

    def from_information_schema(self, info_schema: Dict[str, Any]) -> None:
        """Builds the schema from information schema represenation.

        Args:
          info_schema: the information schema represenation of a graph;
        """
        property_decls = info_schema.get("propertyDeclarations", [])
        decl_by_types = CaseInsensitiveDict(
            {
                decl["name"]: TypeUtility.schema_str_to_spanner_type(decl["type"])
                for decl in property_decls
                if TypeUtility.schema_str_to_spanner_type(decl["type"]) is not None
            }
        )
        for node in info_schema["nodeTables"]:
            node_schema = ElementSchema.from_info_schema(node, decl_by_types)
            self._update_node_schema(node_schema)
            self._update_labels_and_properties(node_schema)

        for edge in info_schema.get("edgeTables", []):
            edge_schema = ElementSchema.from_info_schema(edge, decl_by_types)
            self._update_edge_schema(edge_schema)
            self._update_labels_and_properties(edge_schema)

    def node_type_name(self, name: str) -> str:
        return NODE_KIND if self.use_flexible_schema else name

    def edge_type_name(self, name: str) -> str:
        return EDGE_KIND if self.use_flexible_schema else name

    def get_node_schema(self, name: str) -> Optional[ElementSchema]:
        """Gets the node schema by name.

        Args:
          name: the node schema name.

        Returns:
          Optional[ElementSchema]: returns None if there is no such node schema.
        """
        return self.nodes.get(name, None)

    def get_edge_schema(self, name: str) -> Optional[ElementSchema]:
        """Gets the edge schema by name.

        Args:
          name: the edge schema name.

        Returns:
          Optional[ElementSchema]: returns None if there is no such edge schema.
        """
        return self.edges.get(name, None)

    def get_property_type(self, name: str) -> Optional[param_types.Type]:
        """Gets the property type by name.

        Args:
          name: the property name.

        Returns:
          Optional[param_types.Type]: returns None if there is no such property.
        """
        return self.properties.get(name, None)

    def get_properties_as_struct_type(self, properties: List[str]) -> param_types.Type:
        """Gets the struct type with properties as fields.

        Args:
          properties: a list of property names.

        Returns:
          param_types.Struct: a struct type with properties as fields.
        """
        struct_fields = []
        for p in properties:
            field_type = self.get_property_type(p)
            if field_type is None:
                raise ValueError("No property of given name `%s` found" % p)
            field = param_types.StructField(p, field_type)
            struct_fields.append(field)

        return param_types.Struct(struct_fields)

    def __repr__(self) -> str:
        """Builds a string representation of the graph schema.

        Returns:
          str: a string representation of the graph schema.
        """
        properties = CaseInsensitiveDict(
            {
                k: TypeUtility.spanner_type_to_schema_str(v)
                for k, v in self.properties.items()
            }
        )
        node_labels = {label for node in self.nodes.values() for label in node.labels}
        edge_labels = {label for edge in self.edges.values() for label in edge.labels}
        Triplet = Tuple[ElementSchema, ElementSchema, ElementSchema]
        triplets_per_label: CaseInsensitiveDict[List[Triplet]] = CaseInsensitiveDict({})
        for edge in self.edges.values():
            for label in edge.labels:
                source_node = self.get_node_schema(edge.source.node_name)
                target_node = self.get_node_schema(edge.target.node_name)
                if source_node is None:
                    raise ValueError(f"Source node {edge.source.node_name} not found")
                if target_node is None:
                    raise ValueError(f"Tource node {edge.target.node_name} not found")
                triplets_per_label.setdefault(label, []).append(
                    (source_node, edge, target_node)
                )
        return json.dumps(
            {
                "Name of graph": self.graph_name,
                "Node properties per node label": {
                    label: [
                        {
                            "property name": name,
                            "property type": properties[name],
                        }
                        for name in self.labels[label].prop_names
                    ]
                    for label in node_labels
                },
                "Edge properties per edge label": {
                    label: [
                        {
                            "property name": name,
                            "property type": properties[name],
                        }
                        for name in self.labels[label].prop_names
                    ]
                    for label in edge_labels
                },
                "Possible edges per label": {
                    label: [
                        "(:{}) -[:{}]-> (:{})".format(
                            source_node_label, label, target_node_label
                        )
                        for (source, edge, target) in triplets
                        for source_node_label in source.labels
                        for target_node_label in target.labels
                    ]
                    for label, triplets in triplets_per_label.items()
                },
            },
            indent=2,
        )

    def to_ddl(self) -> str:
        """Returns a CREATE PROPERTY GRAPH ddl that represents the graph schema.

        Returns:
          str: a string of CREATE PROPERTY GRAPH ddl statement.
        """
        to_identifier = GraphDocumentUtility.to_identifier
        to_identifiers = GraphDocumentUtility.to_identifiers

        def construct_label_and_properties(
            target_label: str,
            labels: CaseInsensitiveDict[Label],
            element: ElementSchema,
        ) -> str:
            props = labels[target_label].prop_names
            defs = [
                "{} AS {}".format(v if k != v else to_identifier(k), to_identifier(k))
                for k, v in element.properties.items()
                if k in props
            ]
            return """LABEL {} PROPERTIES({})""".format(
                to_identifier(target_label), ", ".join(defs)
            )

        def construct_label_and_properties_list(
            target_labels: List[str],
            labels: CaseInsensitiveDict[Label],
            element: ElementSchema,
        ) -> str:
            return "\n".join(
                (
                    construct_label_and_properties(target_label, labels, element)
                    for target_label in target_labels
                )
            )

        def construct_columns(cols: List[str]) -> str:
            return ", ".join(to_identifiers(cols))

        def construct_key(keys: List[str]) -> str:
            return "KEY({})".format(construct_columns(keys))

        def construct_node_reference(
            endpoint_type: str, endpoint: NodeReference
        ) -> str:
            return "{} KEY({}) REFERENCES {}({})".format(
                endpoint_type,
                construct_columns(endpoint.edge_keys),
                to_identifier(endpoint.node_name),
                construct_columns(endpoint.node_keys),
            )

        def construct_element_table(
            element: ElementSchema, labels: CaseInsensitiveDict[Label]
        ) -> str:
            definition = [
                "{} AS {}".format(
                    to_identifier(element.base_table_name),
                    to_identifier(element.name),
                ),
                construct_key(element.key_columns),
            ]
            if element.kind == EDGE_KIND:
                definition += [
                    construct_node_reference("SOURCE", element.source),
                    construct_node_reference("DESTINATION", element.target),
                ]
            definition += [
                construct_label_and_properties_list(element.labels, labels, element)
            ]
            return "\n    ".join(definition)

        ddl = "CREATE OR REPLACE PROPERTY GRAPH {}".format(
            to_identifier(self.graph_name)
        )
        ddl += "\nNODE TABLES(\n  "
        ddl += ",\n  ".join(
            (construct_element_table(node, self.labels) for node in self.nodes.values())
        )
        ddl += "\n)"
        if len(self.edges) > 0:
            ddl += "\nEDGE TABLES(\n  "
            ddl += ",\n  ".join(
                (
                    construct_element_table(edge, self.labels)
                    for edge in self.edges.values()
                )
            )
            ddl += "\n)"
        return ddl

    def _update_node_schema(self, node_schema: ElementSchema) -> List[str]:
        """Evolves node schema.

        Args:
          node_schema: a node ElementSchema.

        Returns:
          List[str]: a list of DDL statements that requires to evolve the schema.
        """

        old_schema = self.nodes.get(node_schema.name, None)
        if old_schema is not None:
            ddls = old_schema.evolve(node_schema)
        elif node_schema.base_table_name in self.node_tables:
            ddls = self.node_tables[node_schema.base_table_name].evolve(node_schema)
        else:
            ddls = [node_schema.to_ddl(self)]
            self.node_tables[node_schema.base_table_name] = node_schema

        self.nodes[node_schema.name] = old_schema or node_schema
        return ddls

    def _update_edge_schema(self, edge_schema: ElementSchema) -> List[str]:
        """Evolves edge schema.

        Args:
          edge_schema: an edge ElementSchema.

        Returns:
          List[str]: a list of DDL statements that requires to evolve the schema.
        """
        old_schema = self.edges.get(edge_schema.name, None)
        if old_schema is not None:
            ddls = old_schema.evolve(edge_schema)
        elif edge_schema.base_table_name in self.edge_tables:
            ddls = self.edge_tables[edge_schema.base_table_name].evolve(edge_schema)
        else:
            ddls = [edge_schema.to_ddl(self)]
            self.edge_tables[edge_schema.base_table_name] = edge_schema

        self.edges[edge_schema.name] = old_schema or edge_schema
        return ddls

    def _update_labels_and_properties(self, element_schema: ElementSchema) -> None:
        """Updates labels and properties based on an element schema.

        Args:
          element_schema: an ElementSchema.
        """
        for l in element_schema.labels:
            if l in self.labels:
                self.labels[l].prop_names.update(element_schema.properties.keys())
            else:
                self.labels[l] = Label(l, set(element_schema.properties.keys()))

        self.properties.update(element_schema.types)

    def add_nodes(
        self, name: str, nodes: List[Node]
    ) -> Generator[Tuple[str, Tuple[str], List[List[Any]]], None, None]:
        """Builds the data required to add a list of nodes to Spanner.

        Args:
          name: type of name;
          nodes: a list of Nodes.

        Returns:
          An iterator that yields a tuple consists of the following:
            str: a table name;
            List[str]: a list of column names;
            List[List[Any]]: a list of rows.
        """
        node_schema = self.get_node_schema(self.node_type_name(name))
        if node_schema is None:
            raise ValueError("Unknown node schema: `%s`" % name)
        for v in node_schema.add_nodes(name, nodes):
            yield v

    def add_edges(
        self, name: str, edges: List[Relationship]
    ) -> Generator[Tuple[str, Tuple[str], List[List[Any]]], None, None]:
        """Builds the data required to add a list of edges to Spanner.

        Args:
          name: type of edge;
          edges: a list of Relationships.

        Returns:
          An iterator that yields a tuple consists of the following:
            str: a table name;
            List[str]: a list of column names;
            List[List[Any]]: a list of rows.
        """
        edge_schema = self.get_edge_schema(self.edge_type_name(name))
        if edge_schema is None:
            print(list(self.edges.keys()))
            raise ValueError("Unknown edge schema `%s`" % name)
        for v in edge_schema.add_edges(name, edges):
            yield v


class SpannerInterface(ABC):
    """Wrapper of Spanner APIs."""

    @abstractmethod
    def query(self, query: str, params: dict = {}) -> List[Dict[str, Any]]:
        """Runs a Spanner query.

        Args:
          query: query string;
          params: Spanner query params;
          param_types: Spanner param types.

        Returns:
          List[Dict[str, Any]]: query results.
        """
        raise NotImplementedError

    @abstractmethod
    def apply_ddls(self, ddls: List[str], options: Dict[str, Any] = {}) -> None:
        """Applies a list of schema modifications.

        Args:
          ddls: Spanner Schema DDLs.
        """
        raise NotImplementedError

    @abstractmethod
    def insert_or_update(
        self, table: str, columns: Tuple[str], values: List[List[Any]]
    ) -> None:
        """Insert or update the table.

        Args:
          table: Spanner table name;
          columns: a tuple of column names;
          values: list of values.
        """
        raise NotImplementedError


class SpannerImpl(SpannerInterface):
    """Implementation of SpannerInterface."""

    def __init__(
        self,
        instance_id: str,
        database_id: str,
        client: Optional[spanner.Client] = None,
    ):
        """Initializes the Spanner implementation.

        Args:
          instance_id: Google Cloud Spanner instance id;
          database_id: Google Cloud Spanner database id;
          client: an optional instance of Spanner client.
        """
        self.client = client or spanner.Client()
        self.instance = self.client.instance(instance_id)
        self.database = self.instance.database(database_id)

    def query(self, query: str, params: dict = {}) -> List[Dict[str, Any]]:
        param_types = {k: TypeUtility.value_to_param_type(v) for k, v in params.items()}
        with self.database.snapshot() as snapshot:
            rows = snapshot.execute_sql(query, params=params, param_types=param_types)
            return [
                {
                    column: value
                    for column, value in zip(
                        [column.name for column in rows.fields], row
                    )
                }
                for row in rows
            ]

    def apply_ddls(self, ddls: List[str], options: Dict[str, Any] = {}) -> None:
        if not ddls:
            return

        op = self.database.update_ddl(ddl_statements=ddls)
        print("Waiting for DDL operations to complete...")
        return op.result(options.get("timeout", DEFAULT_DDL_TIMEOUT))

    def insert_or_update(
        self, table: str, columns: Tuple[str], values: List[List[Any]]
    ) -> None:
        for i in range(0, len(values), MUTATION_BATCH_SIZE):
            value_batch = values[i : i + MUTATION_BATCH_SIZE]
            with self.database.batch() as batch:
                batch.insert_or_update(table=table, columns=columns, values=value_batch)


class SpannerGraphStore(GraphStore):
    """A Spanner Graph implementation of GraphStore interface"""

    def __init__(
        self,
        instance_id: str,
        database_id: str,
        graph_name: str,
        client: Optional[spanner.Client] = None,
        use_flexible_schema: bool = False,
        static_node_properties: List[str] = [],
        static_edge_properties: List[str] = [],
        impl: Optional[SpannerInterface] = None,
    ):
        """Initializes SpannerGraphStore.

        Args:
          instance_id: Google Cloud Spanner instance id;
          database_id: Google Cloud Spanner database id;
          graph_name: Graph name;
          client: an optional instance of Spanner client.
          use_flexible_schema: whether to use the flexible schema which uses a
          JSON blob to store node and edge properties;
          static_node_properties: in flexible schema, treat these node
          properties as static;
          static_edge_properties: in flexible schema, treat these edge
          properties as static.
        """
        self.impl = impl or SpannerImpl(
            instance_id,
            database_id,
            client_with_user_agent(client, USER_AGENT_GRAPH_STORE),
        )
        self.schema = SpannerGraphSchema(
            graph_name,
            use_flexible_schema,
            static_node_properties,
            static_edge_properties,
        )

        self.refresh_schema()

    def add_graph_documents(
        self,
        graph_documents: List[GraphDocument],
        include_source: bool = False,
        baseEntityLabel: bool = False,
    ) -> None:
        """Constructs nodes and relationships in the graph based on the provided docs.

        Args:
          graph_documents (List[GraphDocument]): A list of GraphDocument objects
            that contain the nodes and relationships to be added to the graph. Each
            GraphDocument should encapsulate the structure of part of the graph,
        """
        if include_source:
            raise NotImplementedError("include_source is not supported yet")
        if baseEntityLabel:
            raise NotImplementedError("baseEntityLabel is not supported yet")

        GraphDocumentUtility.fixup_graph_documents(graph_documents)

        ddls = self.schema.evolve(graph_documents)
        if ddls:
            self.impl.apply_ddls(ddls)
            self.refresh_schema()
        else:
            print("No schema change required...")

        nodes, edges = partition_graph_docs(graph_documents)
        for name, elements in nodes.items():
            if len(elements) == 0:
                continue
            for table, columns, rows in self.schema.add_nodes(name, elements):
                print("Insert nodes of type `{}`...".format(name))
                self.impl.insert_or_update(table, columns, rows)

        for name, elements in edges.items():
            if len(elements) == 0:
                continue
            for table, columns, rows in self.schema.add_edges(name, elements):
                print("Insert edges of type `{}`...".format(name))
                self.impl.insert_or_update(table, columns, rows)

    def query(self, query: str, params: dict = {}) -> List[Dict[str, Any]]:
        """Query Spanner database.

        Args:
          query (str): The query to execute.
          params (dict): The parameters to pass to the query.

        Returns:
          List[Dict[str, Any]]: The list of dictionaries containing the query
          results.
        """
        return self.impl.query(query, params)

    @property
    def get_schema(self) -> str:
        return str(self.schema)

    @property
    def get_structured_schema(self) -> Dict[str, Any]:
        return json.loads(repr(self.schema))

    def get_ddl(self) -> str:
        return self.schema.to_ddl()

    def refresh_schema(self) -> None:
        """Refreshes the Neo4j graph schema information."""
        results = self.query(
            SpannerGraphSchema.GRAPH_INFORMATION_SCHEMA_QUERY_TEMPLATE.format(
                self.schema.graph_name
            )
        )
        if len(results) == 0:
            return

        if len(results) != 1:
            raise Exception(
                "Unexpected number of rows from information schema: {}".format(
                    len(results)
                )
            )

        self.schema.from_information_schema(results[0]["property_graph_metadata_json"])

    def cleanup(self):
        """Removes all data from your Spanner Graph.

        USE IT WITH CAUTION!

        The graph, tables and the associated data will all be removed.
        """
        to_identifier = GraphDocumentUtility.to_identifier
        self.impl.apply_ddls(
            [
                "DROP PROPERTY GRAPH IF EXISTS {}".format(
                    to_identifier(self.schema.graph_name)
                )
            ]
        )
        self.impl.apply_ddls(
            [
                "DROP TABLE IF EXISTS {}".format(to_identifier(edge.base_table_name))
                for edge in self.schema.edges.values()
            ]
        )
        self.impl.apply_ddls(
            [
                "DROP TABLE IF EXISTS {}".format(to_identifier(node.base_table_name))
                for node in self.schema.nodes.values()
            ]
        )
        self.schema = SpannerGraphSchema(
            self.schema.graph_name, self.schema.use_flexible_schema
        )
