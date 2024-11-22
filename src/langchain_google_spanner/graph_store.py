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
import string
from abc import abstractmethod
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union

from google.cloud import spanner
from google.cloud.spanner_v1 import param_types
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship
from langchain_community.graphs.graph_store import GraphStore
from requests.structures import CaseInsensitiveDict

from .type_utils import TypeUtility

MUTATION_BATCH_SIZE = 1000


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
            )
        )

    def __eq__(self, other: Any):
        if isinstance(other, EdgeWrapper):
            return (
                self.edge.source.id == other.edge.source.id
                and self.edge.target.id == other.edge.target.id
            )
        return False


def partition_graph_docs(
    graph_documents: List[GraphDocument],
) -> Tuple[dict, dict]:
    """Returns nodes and edges grouped by the type.

    Parameters:
    - graph_documents: List[GraphDocument]

    Returns:
    - A tuple of two dictionaries. The first is nodes grouped by node types,
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


class GraphDocumentUtility:
    """Utilities to process graph documents."""

    @staticmethod
    def to_identifier(s: str) -> str:
        return "`" + s + "`"

    @staticmethod
    def to_identifiers(s: List[str]) -> Iterable[str]:
        return map(GraphDocumentUtility.to_identifier, s)

    @staticmethod
    def fixup_identifier(s: str) -> str:
        return re.sub("[{}]".format(string.whitespace), "_", s)

    @staticmethod
    def fixup_graph_documents(graph_documents: List[GraphDocument]) -> None:
        """Preprocess graph documents

        Parameters:
        - graph_documents: List[GraphDocument]
        """
        for graph_document in graph_documents:
            for node in graph_document.nodes:
                GraphDocumentUtility.fixup_element(node)
            for edge in graph_document.relationships:
                GraphDocumentUtility.fixup_element(edge)

    @staticmethod
    def fixup_element(element: Union[Node, Relationship]) -> None:
        """Preprocess graph element

        Parameters:
        - element: Node or Relationship
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

    name: str
    kind: str
    key_columns: List[str]
    base_table_name: str
    labels: List[str]
    properties: CaseInsensitiveDict[str]
    types: CaseInsensitiveDict[param_types.Type]
    source: NodeReference
    target: NodeReference

    @staticmethod
    def from_nodes(name: str, nodes: List[Node]) -> ElementSchema:
        """Builds ElementSchema from a list of nodes.

        Parameters:
        - name: name of the schema;
        - nodes: a non-empty list of nodes.

        Returns:
        - ElementSchema: schema representation of the nodes.
        """
        if len(nodes) == 0:
            raise ValueError("The list of nodes should not be empty")

        props = set((key.casefold() for n in nodes for key in n.properties.keys()))
        if ElementSchema.NODE_KEY_COLUMN_NAME in props:
            raise ValueError(
                "Node properties should not contain property with name: `%s`"
                % ElementSchema.NODE_KEY_COLUMN_NAME
            )
        props.add(ElementSchema.NODE_KEY_COLUMN_NAME)

        node = ElementSchema()
        node.name = name
        node.kind = "NODE"
        node.key_columns = [ElementSchema.NODE_KEY_COLUMN_NAME]
        node.base_table_name = name
        node.labels = [name]
        node.properties = CaseInsensitiveDict({prop: prop for prop in props})
        node.types = CaseInsensitiveDict(
            {
                k: TypeUtility.value_to_param_type(v)
                for n in nodes
                for k, v in n.properties.items()
            }
        )
        node.types[ElementSchema.NODE_KEY_COLUMN_NAME] = (
            TypeUtility.value_to_param_type(nodes[0].id)
        )
        return node

    @staticmethod
    def from_edges(name: str, edges: List[Relationship]) -> ElementSchema:
        """Builds ElementSchema from a list of edges.

        Parameters:
        - name: name of the schema;
        - nodes: a non-empty list of edges.

        Returns:
        - ElementSchema: schema representation of the edges.
        """
        if len(edges) == 0:
            raise ValueError("The list of edges should not be empty")

        props = set((key.casefold() for e in edges for key in e.properties.keys()))
        if ElementSchema.NODE_KEY_COLUMN_NAME in props:
            raise ValueError(
                "Edge properties should not contain property with name: `%s`"
                % ElementSchema.NODE_KEY_COLUMN_NAME
            )
        if ElementSchema.TARGET_NODE_KEY_COLUMN_NAME in props:
            raise ValueError(
                "Edge properties should not contain property with name: `%s`"
                % ElementSchema.TARGET_NODE_KEY_COLUMN_NAME
            )
        props.add(ElementSchema.NODE_KEY_COLUMN_NAME)
        props.add(ElementSchema.TARGET_NODE_KEY_COLUMN_NAME)

        edge = ElementSchema()
        edge.name = name
        edge.kind = "EDGE"
        edge.key_columns = [
            ElementSchema.NODE_KEY_COLUMN_NAME,
            ElementSchema.TARGET_NODE_KEY_COLUMN_NAME,
        ]
        edge.base_table_name = name

        # Uses the type as label because the label can be shared by multiple base
        # tables.
        edge.labels = [edges[0].type]
        edge.properties = CaseInsensitiveDict({prop: prop for prop in props})
        edge.types = CaseInsensitiveDict(
            {
                k: TypeUtility.value_to_param_type(v)
                for e in edges
                for k, v in e.properties.items()
            }
        )
        edge.types[ElementSchema.NODE_KEY_COLUMN_NAME] = (
            TypeUtility.value_to_param_type(edges[0].source.id)
        )
        edge.types[ElementSchema.TARGET_NODE_KEY_COLUMN_NAME] = (
            TypeUtility.value_to_param_type(edges[0].target.id)
        )

        edge.source = NodeReference(
            edges[0].source.type,
            [ElementSchema.NODE_KEY_COLUMN_NAME],
            [ElementSchema.NODE_KEY_COLUMN_NAME],
        )
        edge.target = NodeReference(
            edges[0].target.type,
            [ElementSchema.NODE_KEY_COLUMN_NAME],
            [ElementSchema.TARGET_NODE_KEY_COLUMN_NAME],
        )
        return edge

    @staticmethod
    def from_info_schema(
        element_schema: Dict[str, Any],
        property_decls: List[Any],
    ) -> ElementSchema:
        """Builds ElementSchema from information schema represenation of an element.

        Parameters:
        - element_schema: the information schema represenation of an element;
        - property_decls: the information schema represenation of property
          declarations.

        Returns:
        - ElementSchema
        """
        element = ElementSchema()
        element.name = element_schema["name"]
        element.kind = element_schema["kind"]
        element.key_columns = element_schema["keyColumns"]
        element.base_table_name = element_schema["baseTableName"]
        element.labels = element_schema["labelNames"]
        element.properties = CaseInsensitiveDict(
            {
                prop_def["propertyDeclarationName"]: prop_def["valueExpressionSql"]
                for prop_def in element_schema.get("propertyDefinitions", [])
            }
        )
        element.types = CaseInsensitiveDict(
            {
                decl["name"]: TypeUtility.schema_str_to_spanner_type(decl["type"])
                for decl in property_decls
            }
        )

        if element.kind == "EDGE":
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

    def to_ddl(self) -> str:
        """Returns a CREATE TABLE ddl that represents the element schema.

        Returns:
        - str: a string of CREATE TABLE ddl statement.
        """

        to_identifier = GraphDocumentUtility.to_identifier
        to_identifiers = GraphDocumentUtility.to_identifiers
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
                    to_identifier(self.target.node_name),
                    ", ".join(to_identifiers(self.target.node_keys)),
                )
                if self.kind == "EDGE"
                else ""
            ),
            ",".join(to_identifiers(self.key_columns)),
            (
                ", INTERLEAVE IN PARENT {}".format(to_identifier(self.source.node_name))
                if self.kind == "EDGE"
                else ""
            ),
        )

    def evolve(self, new_schema: ElementSchema) -> List[str]:
        """Evolves current schema from the new schema.

        Parameters:
        - new_schema: an ElementSchema representing new nodes/edges.

        Returns:
        - List[str]: a list of DDL statements.
        """
        if self.name.casefold() != new_schema.name.casefold():
            raise ValueError(
                "Schema should have the same kind, got {}, expected {}".format(
                    new_schema.name, self.name
                )
            )
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

    def __init__(self, graph_name: str):
        """Initializes the graph schema.

        Parameters:
        - graph_name: the name of the graph.
        """
        self.graph_name: str = graph_name
        self.nodes: CaseInsensitiveDict[ElementSchema] = CaseInsensitiveDict({})
        self.edges: CaseInsensitiveDict[ElementSchema] = CaseInsensitiveDict({})
        self.labels: CaseInsensitiveDict[Label] = CaseInsensitiveDict({})
        self.properties: CaseInsensitiveDict[param_types.Type] = CaseInsensitiveDict({})

    def evolve(self, graph_documents: List[GraphDocument]) -> List[str]:
        """Evolves current schema into a schema representing the input documents.

        Parameters:
        - graph_documents: a list of GraphDocument.

        Returns:
        - List[str]: a list of DDL statements.
        """
        nodes, edges = partition_graph_docs(graph_documents)

        ddls = []
        for k, ns in nodes.items():
            node_schema = ElementSchema.from_nodes(k, ns)
            ddls.extend(self._update_node_schema(node_schema))
            self._update_labels_and_properties(node_schema)

        for k, es in edges.items():
            edge_schema = ElementSchema.from_edges(k, es)
            ddls.extend(self._update_edge_schema(edge_schema))
            self._update_labels_and_properties(edge_schema)

        if len(ddls) != 0:
            ddls += [self.to_ddl()]
        return ddls

    def from_information_schema(self, info_schema: Dict[str, Any]) -> None:
        """Builds the schema from information schema represenation.

        Parameters:
        - info_schema: the information schema represenation of a graph;
        """
        property_decls = info_schema.get("propertyDeclarations", [])
        self.nodes = CaseInsensitiveDict(
            {
                node["name"]: ElementSchema.from_info_schema(node, property_decls)
                for node in info_schema["nodeTables"]
            }
        )
        self.edges = CaseInsensitiveDict(
            {
                edge["name"]: ElementSchema.from_info_schema(edge, property_decls)
                for edge in info_schema.get("edgeTables", [])
            }
        )
        self.labels = CaseInsensitiveDict(
            {
                label["name"]: Label(
                    label["name"], set(label.get("propertyDeclarationNames", []))
                )
                for label in info_schema["labels"]
            }
        )
        self.properties = CaseInsensitiveDict(
            {
                decl["name"]: TypeUtility.schema_str_to_spanner_type(decl["type"])
                for decl in property_decls
            }
        )

    def get_node_schema(self, name: str) -> Optional[ElementSchema]:
        """Gets the node schema by name.

        Parameters:
        - name: the node schema name.

        Returns:
        - Optional[ElementSchema]: returns None if there is no such node schema.
        """
        return self.nodes.get(name, None)

    def get_edge_schema(self, name: str) -> Optional[ElementSchema]:
        """Gets the edge schema by name.

        Parameters:
        - name: the edge schema name.

        Returns:
        - Optional[ElementSchema]: returns None if there is no such edge schema.
        """
        return self.edges.get(name, None)

    def get_property_type(self, name: str) -> Optional[param_types.Type]:
        """Gets the property type by name.

        Parameters:
        - name: the property name.

        Returns:
        - Optional[param_types.Type]: returns None if there is no such property.
        """
        return self.properties.get(name, None)

    def get_properties_as_struct_type(self, properties: List[str]) -> param_types.Type:
        """Gets the struct type with properties as fields.

        Parameters:
        - properties: a list of property names.

        Returns:
        - param_types.Struct: a struct type with properties as fields.
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
        - str: a string representation of the graph schema.
        """
        properties = {
            k: TypeUtility.spanner_type_to_schema_str(v)
            for k, v in self.properties.items()
        }
        import json

        return json.dumps(
            {
                "Name of graph": self.graph_name,
                "Node properties per node type": {
                    node.name: [
                        {
                            "property name": name,
                            "property type": properties[name],
                        }
                        for name in node.properties.keys()
                    ]
                    for node in self.nodes.values()
                },
                "Edge properties per edge type": {
                    edge.name: [
                        {
                            "property name": name,
                            "property type": properties[name],
                        }
                        for name in edge.properties.keys()
                    ]
                    for edge in self.edges.values()
                },
                "Node labels per node type": {
                    node.name: node.labels for node in self.nodes.values()
                },
                "Edge labels per edge type": {
                    edge.name: edge.labels for edge in self.edges.values()
                },
                "Edges": {
                    edge.name: "From {} nodes to {} nodes".format(
                        edge.source.node_name, edge.target.node_name
                    )
                    for edge in self.edges.values()
                },
            },
            indent=2,
        )

    def to_ddl(self) -> str:
        """Returns a CREATE PROPERTY GRAPH ddl that represents the graph schema.

        Returns:
        - str: a string of CREATE PROPERTY GRAPH ddl statement.
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

        def constuct_element_table(
            element: ElementSchema, labels: CaseInsensitiveDict[Label]
        ) -> str:
            definition = [
                "{} AS {}".format(
                    to_identifier(element.base_table_name),
                    to_identifier(element.name),
                ),
                construct_key(element.key_columns),
            ]
            if element.kind == "EDGE":
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
            (constuct_element_table(node, self.labels) for node in self.nodes.values())
        )
        ddl += "\n)"
        if len(self.edges) > 0:
            ddl += "\nEDGE TABLES(\n  "
            ddl += ",\n  ".join(
                (
                    constuct_element_table(edge, self.labels)
                    for edge in self.edges.values()
                )
            )
            ddl += "\n)"
        return ddl

    def _update_node_schema(self, node_schema: ElementSchema) -> List[str]:
        """Evolves node schema.

        Parameters:
        - node_schema: a node ElementSchema.

        Returns:
        - List[str]: a list of DDL statements that requires to evolve the schema.
        """
        if node_schema.name not in self.nodes:
            ddls = [node_schema.to_ddl()]
            self.nodes[node_schema.name] = node_schema
            return ddls
        ddls = self.nodes[node_schema.name].evolve(node_schema)
        return ddls

    def _update_edge_schema(self, edge_schema: ElementSchema) -> List[str]:
        """Evolves edge schema.

        Parameters:
        - edge_schema: an edge ElementSchema.

        Returns:
        - List[str]: a list of DDL statements that requires to evolve the schema.
        """
        if edge_schema.name not in self.edges:
            ddls = [edge_schema.to_ddl()]
            self.edges[edge_schema.name] = edge_schema
            return ddls
        ddls = self.edges[edge_schema.name].evolve(edge_schema)
        return ddls

    def _update_labels_and_properties(self, element_schema: ElementSchema) -> None:
        """Updates labels and properties based on an element schema.

        Parameters:
        - element_schema: an ElementSchema.
        """
        for l in element_schema.labels:
            if l in self.labels:
                self.labels[l].prop_names.update(element_schema.properties.keys())
            else:
                self.labels[l] = Label(l, set(element_schema.properties.keys()))

        self.properties.update(element_schema.types)


class SpannerImpl(object):
    """Wrapper of Spanner APIs."""

    def __init__(
        self,
        instance_id: str,
        database_id: str,
        client: Optional[spanner.Client] = None,
    ):
        """Initializes the Spanner implementation.

        Parameters:
        - instance_id: Google Cloud Spanner instance id;
        - database_id: Google Cloud Spanner database id;
        - client: an optional instance of Spanner client.
        """
        self.client = client or spanner.Client()
        self.instance = self.client.instance(instance_id)
        self.database = self.instance.database(database_id)

    def query(self, query: str, params: dict = {}) -> List[Dict[str, Any]]:
        """Runs a Spanner query.

        Parameters:
        - query: query string;
        - params: Spanner query params;
        - param_types: Spanner param types.

        Returns:
        - List[Dict[str, Any]]: query results.
        """

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
        """Applies a list of schema modifications.

        Parameters:
        - ddls: Spanner Schema DDLs.
        """
        if not ddls:
            return

        op = self.database.update_ddl(ddl_statements=ddls)
        print("Waiting for DDL operations to complete...")
        return op.result(options.get("timeout", 60))

    def insert_or_update(
        self, table: str, columns: List[str], values: List[List[Any]]
    ) -> None:
        """Insert or update the table.

        Parameters:
        - table: Spanner table name;
        - columns: list of column names;
        - values: list of values.
        """

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
    ):
        """Parameters:

        - instance_id: Google Cloud Spanner instance id;
        - database_id: Google Cloud Spanner database id;
        - graph_name: Graph name;
        - client: an optional instance of Spanner client.
        """
        self.impl = SpannerImpl(instance_id, database_id, client)
        self.schema = SpannerGraphSchema(graph_name)

        self.refresh_schema()

    def add_graph_documents(
        self,
        graph_documents: List[GraphDocument],
        include_source: bool = False,
        baseEntityLabel: bool = False,
    ) -> None:
        """Constructs nodes and relationships in the graph based on the provided docs.

        Parameters:
        - graph_documents (List[GraphDocument]): A list of GraphDocument objects
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
            table, columns, rows = self._add_nodes(name, elements)

            print("Insert nodes of type `{}`...".format(name))
            self.impl.insert_or_update(table, columns, rows)

        for name, elements in edges.items():
            if len(elements) == 0:
                continue
            table, columns, rows = self._add_edges(name, elements)

            print("Insert edges of type `{}`...".format(name))
            self.impl.insert_or_update(table, columns, rows)

    def query(self, query: str, params: dict = {}) -> List[Dict[str, Any]]:
        """Query Spanner database.

        Parameters:
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
        return {
            "nodes": self.schema.nodes,
            "edges": self.schema.edges,
            "labels": self.schema.labels,
            "properties": self.schema.properties,
        }

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

    def _add_nodes(
        self, name: str, nodes: List[Node]
    ) -> Tuple[str, List[str], List[List[Any]]]:
        """Builds the data required to add a list of nodes to Spanner.

        Parameters:
          - name: type of name;
          - nodes: a list of Nodes.

        Returns:
          - str: a table name;
          - List[str]: a list of column names;
          - List[List[Any]]: a list of rows.
        """
        if len(nodes) == 0:
            raise ValueError("Empty list of nodes")

        columns = list(set({k for node in nodes for k, v in node.properties.items()}))

        rows = []
        for node in nodes:
            row = [node.properties.get(k, None) for k in columns]
            row.append(node.id)
            rows.append(row)

        columns.append(ElementSchema.NODE_KEY_COLUMN_NAME)
        return name, columns, rows

    def _add_edges(
        self, name: str, edges: List[Relationship]
    ) -> Tuple[str, List[str], List[List[Any]]]:
        """Builds the data required to add a list of edges to Spanner.

        Parameters:
          - name: type of edge;
          - edges: a list of Relationships.

        Returns:
          - str: a table name;
          - List[str]: a list of column names;
          - List[List[Any]]: a list of rows.
        """
        if len(edges) == 0:
            raise ValueError("Empty list of edges")

        columns = list(set({k for edge in edges for k, v in edge.properties.items()}))

        rows = []
        for edge in edges:
            row = [edge.properties.get(k, None) for k in columns]
            row.append(edge.source.id)
            row.append(edge.target.id)
            rows.append(row)

        columns.append(ElementSchema.NODE_KEY_COLUMN_NAME)
        columns.append(ElementSchema.TARGET_NODE_KEY_COLUMN_NAME)
        return name, columns, rows

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
        self.schema = SpannerGraphSchema(self.schema.graph_name)
