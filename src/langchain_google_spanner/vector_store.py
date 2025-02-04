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

import datetime
import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type, Union

import numpy as np
from google.cloud import spanner  # type: ignore
from google.cloud.spanner_admin_database_v1.types import DatabaseDialect
from google.cloud.spanner_v1 import JsonObject, param_types
from langchain_community.vectorstores.utils import maximal_marginal_relevance
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from .version import __version__

logger = logging.getLogger(__name__)

ID_COLUMN_NAME = "langchain_id"
CONTENT_COLUMN_NAME = "content"
EMBEDDING_COLUMN_NAME = "embedding"
ADDITIONAL_METADATA_COLUMN_NAME = "metadata"

USER_AGENT_VECTOR_STORE = "langchain-google-spanner-python:vector_store/" + __version__

KNN_DISTANCE_SEARCH_QUERY_ALIAS = "distance"

from dataclasses import dataclass


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


@dataclass
class TableColumn:
    """
    Represents column configuration, to be used as part of create DDL statement for table creation.

    Attributes:
        column_name (str): The name of the column.
        type (str): The type of the column.
        is_null (bool): Indicates whether the column allows null values.
        vector_length Optional(int): for ANN, mandatory and must be >=1 for the embedding column.
    """

    name: str
    type: str
    is_null: bool = True
    vector_length: Optional[int] = None

    def __post_init__(self):
        # Check if column_name is None after initialization
        if self.name is None:
            raise ValueError("column_name is mandatory and cannot be None.")

        if self.type is None:
            raise ValueError("type is mandatory and cannot be None.")

        if (self.vector_length is not None) and (self.vector_length <= 0):
            raise ValueError("vector_length must be >=1")


class SecondaryIndex:
    def __init__(
        self,
        index_name: str,
        columns: list[str],
        storing_columns: Optional[list[str]] = None,
    ):
        self.index_name = index_name
        self.columns = columns
        self.storing_columns = storing_columns

    def __post_init__(self):
        # Check if column_name is None after initialization
        if self.index_name is None:
            raise ValueError("Index Name can't be None")

        if self.columns is None:
            raise ValueError("Index Columns can't be None")


class VectorSearchIndex(SecondaryIndex):
    """
    The index for use with Approximate Nearest Neighbor (ANN) vector search.
    """

    def __init__(
        self,
        num_leaves: int,
        num_branches: int,
        tree_depth: int,
        distance_type: DistanceStrategy,
        nullable_column: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.num_leaves = num_leaves
        self.num_branches = num_branches
        self.tree_depth = tree_depth
        self.distance_type = distance_type
        self.nullable_column = nullable_column

    def __post_init__(self):
        if self.index_name is None:
            raise ValueError("index_name must be set")

        if len(self.columns) == 0:
            raise ValueError("columns must be set")

        ok_tree_depth = self.tree_depth in (2, 3)
        if not ok_tree_depth:
            raise ValueError("tree_depth must be either 2 or 3")


class DistanceStrategy(Enum):
    """
    Enum for distance calculation strategies.
    """

    COSINE = 1
    EUCLIDEAN = 2
    DOT_PRODUCT = 3

    def __str__(self):
        return self.name


class DialectSemantics(ABC):
    """
    Abstract base class for dialect semantics.
    """

    @abstractmethod
    def getDistanceFunction(self, distance_strategy=DistanceStrategy.EUCLIDEAN) -> str:
        """
        Abstract method to get the distance function based on the provided distance strategy.

        Parameters:
        - distance_strategy (DistanceStrategy): The distance calculation strategy. Defaults to DistanceStrategy.EUCLIDEAN.

        Returns:
        - str: The name of the distance function.
        """
        raise NotImplementedError(
            "getDistanceFunction method must be implemented by subclass."
        )

    @abstractmethod
    def getDeleteDocumentsParameters(self, columns) -> Tuple[str, Any]:
        raise NotImplementedError(
            "getDeleteDocumentsParameters method must be implemented by subclass."
        )

    @abstractmethod
    def getDeleteDocumentsValueParameters(self, columns, values) -> Dict[str, Any]:
        raise NotImplementedError(
            "getDeleteDocumentsValueParameters method must be implemented by subclass."
        )


# Maps between distance strategy enums and the appropriate vector search index name.
GOOGLE_DIALECT_TO_KNN_DISTANCE_FUNCTIONS = {
    DistanceStrategy.COSINE: "COSINE_DISTANCE",
    DistanceStrategy.DOT_PRODUCT: "DOT_PRODUCT",
    DistanceStrategy.EUCLIDEAN: "EUCLIDEAN_DISTANCE",
}

# Maps between distance strategy and the appropriate ANN search function name.
GOOGLE_DIALECT_TO_ANN_DISTANCE_FUNCTIONS = {
    DistanceStrategy.COSINE: "APPROX_COSINE_DISTANCE",
    DistanceStrategy.DOT_PRODUCT: "APPROX_DOT_PRODUCT",
    DistanceStrategy.EUCLIDEAN: "APPROX_EUCLIDEAN_DISTANCE",
}


class GoogleSqlSemantics(DialectSemantics):
    """
    Implementation of dialect semantics for Google SQL.
    """

    def getDistanceFunction(self, distance_strategy=DistanceStrategy.EUCLIDEAN) -> str:
        return GOOGLE_DIALECT_TO_KNN_DISTANCE_FUNCTIONS.get(
            distance_strategy, "EUCLIDEAN"
        )

    def getDeleteDocumentsParameters(self, columns) -> Tuple[str, Any]:
        where_clause_condition = " AND ".join(
            ["{} = @{}".format(column, column) for column in columns]
        )

        param_types_dict = {column: param_types.STRING for column in columns}

        return where_clause_condition, param_types_dict

    def getDeleteDocumentsValueParameters(self, columns, values) -> Dict[str, Any]:
        return dict(zip(columns, values))

    def getIndexDistanceType(self, distance_strategy) -> str:
        value = GOOGLE_DIALECT_TO_ANN_DISTANCE_FUNCTIONS.get(distance_strategy, None)
        if value is None:
            raise Exception(f"{distance_strategy} is unsupported for distance_type")
        return value


# Maps between DistanceStrategy and the expected PostgreSQL distance equivalent.
PG_DIALECT_TO_KNN_DISTANCE_FUNCTIONS = {
    DistanceStrategy.COSINE: "spanner.cosine_distance",
    DistanceStrategy.DOT_PRODUCT: "spanner.dot_product",
    DistanceStrategy.EUCLIDEAN: "spanner.euclidean_distance",
}


class PGSqlSemantics(DialectSemantics):
    """
    Implementation of dialect semantics for PostgreSQL.
    """

    def getDistanceFunction(self, distance_strategy=DistanceStrategy.EUCLIDEAN) -> str:
        name = PG_DIALECT_TO_KNN_DISTANCE_FUNCTIONS.get(distance_strategy, None)
        if name is None:
            raise Exception(
                "Unsupported PostgreSQL distance strategy: {}".format(distance_strategy)
            )
        return name

    def getDeleteDocumentsParameters(self, columns) -> Tuple[str, Any]:
        where_clause_condition = " AND ".join(
            [
                "{} = ${}".format(column, index + 1)
                for index, column in enumerate(columns)
            ]
        )

        value_placeholder_list = [
            "p{}".format(index + 1) for index in range(len(columns))
        ]

        param_types_dict = {
            value_placeholder: param_types.STRING
            for value_placeholder in value_placeholder_list
        }

        return where_clause_condition, param_types_dict

    def getDeleteDocumentsValueParameters(self, columns, values) -> Dict[str, Any]:
        value_placeholder_list = [
            "p{}".format(index + 1) for index in range(len(columns))
        ]
        return dict(zip(value_placeholder_list, values))


class QueryParameters:
    """
    Class representing query parameters for nearest neighbors search.
    """

    class NearestNeighborsAlgorithm(Enum):
        """
        Enum for nearest neighbors search algorithms.
        """

        EXACT_NEAREST_NEIGHBOR = 1
        APPROXIMATE_NEAREST_NEIGHBOR = 2

    def __init__(
        self,
        algorithm=NearestNeighborsAlgorithm.EXACT_NEAREST_NEIGHBOR,
        distance_strategy=DistanceStrategy.EUCLIDEAN,
        read_timestamp: Optional[datetime.datetime] = None,
        min_read_timestamp: Optional[datetime.datetime] = None,
        max_staleness: Optional[datetime.timedelta] = None,
        exact_staleness: Optional[datetime.timedelta] = None,
    ):
        """
        Initialize query parameters.

        Parameters:
        - algorithm (NearestNeighborsAlgorithm): The nearest neighbors search algorithm. Defaults to NearestNeighborsAlgorithm.BRUTE_FORCE.
        - distance_strategy (DistanceStrategy): The distance calculation strategy. Defaults to DistanceStrategy.EUCLIDEAN.
        - staleness (int): The staleness value. Defaults to 0.
        """
        self.algorithm = algorithm
        self.distance_strategy = distance_strategy

        key: Optional[str]
        value: Any

        self.staleness = None
        key = None

        if read_timestamp:
            key = "read_timestamp"
            value = read_timestamp
        elif min_read_timestamp:
            key = "min_read_timestamp"
            value = min_read_timestamp
        elif max_staleness:
            key = "max_staleness"
            value = max_staleness
        elif exact_staleness:
            key = "exact_staleness"
            value = exact_staleness

        if key is not None:
            self.staleness = {key: value}


class SpannerVectorStore(VectorStore):
    GSQL_TYPES = {
        CONTENT_COLUMN_NAME: ["STRING"],
        EMBEDDING_COLUMN_NAME: ["ARRAY<FLOAT64>", "ARRAY<FLOAT32>"],
        "metadata_json_column": ["JSON"],
    }

    PGSQL_TYPES = {
        CONTENT_COLUMN_NAME: ["character varying"],
        EMBEDDING_COLUMN_NAME: ["double precision[]"],
        "metadata_json_column": ["jsonb"],
    }

    """
    A class for managing vector stores in Google Cloud Spanner.
    """

    @staticmethod
    def init_vector_store_table(
        instance_id: str,
        database_id: str,
        table_name: str,
        client: Optional[spanner.Client] = None,
        id_column: Union[str, TableColumn] = ID_COLUMN_NAME,
        content_column: str = CONTENT_COLUMN_NAME,
        embedding_column: str = EMBEDDING_COLUMN_NAME,
        metadata_columns: Optional[List[TableColumn]] = None,
        primary_key: Optional[str] = None,
        vector_size: Optional[int] = None,
        secondary_indexes: Optional[List[SecondaryIndex | VectorSearchIndex]] = None,
    ) -> bool:
        """
        Initialize the vector store new table in Google Cloud Spanner.

        Parameters:
        - instance_id (str): The ID of the Spanner instance.
        - database_id (str): The ID of the Spanner database.
        - table_name (str): The name of the table to initialize.
        - client (Client): The Spanner client. Defaults to Client(project="span-cloud-testing").
        - id_column (str): The name of the row ID column. Defaults to ID_COLUMN_NAME.
        - content_column (str): The name of the content column. Defaults to CONTENT_COLUMN_NAME.
        - embedding_column (str): The name of the embedding column. Defaults to EMBEDDING_COLUMN_NAME.
        - metadata_columns (Optional[List[Tuple]]): List of tuples containing metadata column information. Defaults to None.
        - vector_size (Optional[int]): The size of the vector for KNN or ANN. Defaults to None. It is presumed that exactly ONLY 1 field will have the vector.
        """

        client = client_with_user_agent(client, USER_AGENT_VECTOR_STORE)
        instance = client.instance(instance_id)

        if not instance.exists():
            raise Exception("Instance with id:  {} doesn't exist.".format(instance_id))

        database = instance.database(database_id)

        if not database.exists():
            raise Exception("Database with id: {} doesn't exist.".format(database_id))

        database.reload()

        ddl = SpannerVectorStore._generate_sql(
            database.database_dialect,
            table_name,
            id_column,
            content_column,
            embedding_column,
            metadata_columns,
            primary_key,
            secondary_indexes,
            vector_size,
        )

        operation = database.update_ddl(ddl)

        print("Waiting for operation to complete...")
        operation.result(100000)

        return True

    @staticmethod
    def _generate_sql(
        dialect,
        table_name,
        id_column,
        content_column,
        embedding_column,
        column_configs,
        primary_key,
        secondary_indexes: Optional[List[SecondaryIndex | VectorSearchIndex]] = None,
        vector_size: Optional[int] = None,
    ):
        """
        Generate SQL for creating the vector store table.

        Parameters:
        - dialect: The database dialect.
        - table_name: The name of the table.
        - id_column: The name of the row ID column.
        - content_column: The name of the content column.
        - embedding_column: The name of the embedding column.
        - column_names: List of tuples containing metadata column information.
        - vector_size: The vector length to be used by default. It is presumed by proxy of the langchain usage patterns, that exactly ONE column will be used as the embedding.

        Returns:
        - str: The generated SQL.
        """

        embedding_config = list(
            filter(lambda x: x.name == embedding_column, column_configs)
        )
        if embedding_column and len(embedding_config) > 0:
            config = embedding_config[0]
            if config.vector_length is None or config.vector_length <= 0:
                raise ValueError("vector_length is mandatory and must be >=1")

        ddl_statements = [
            SpannerVectorStore._generate_create_table_sql(
                table_name,
                id_column,
                content_column,
                embedding_column,
                column_configs,
                primary_key,
                dialect,
                vector_length=vector_size,
            )
        ]

        if not secondary_indexes:
            secondary_indexes = []

        ann_indices = list(
            filter(lambda index: type(index) is VectorSearchIndex, secondary_indexes)
        )
        ddl_statements += SpannerVectorStore._generate_secondary_indices_ddl_ANN(
            table_name,
            dialect,
            secondary_indexes=list(ann_indices),
        )

        knn_indices = list(
            filter(lambda index: type(index) is SecondaryIndex, secondary_indexes)
        )
        ddl_statements += SpannerVectorStore._generate_secondary_indices_ddl_KNN(
            table_name,
            embedding_column,
            dialect,
            secondary_indexes=list(knn_indices),
        )

        return ddl_statements

    @staticmethod
    def _generate_create_table_sql(
        table_name,
        id_column,
        content_column,
        embedding_column,
        column_configs,
        primary_key,
        dialect=DatabaseDialect.GOOGLE_STANDARD_SQL,
        vector_length=None,
    ):
        create_table_statement = f"CREATE TABLE IF NOT EXISTS {table_name} (\n"

        if not isinstance(id_column, TableColumn):
            if dialect == DatabaseDialect.POSTGRESQL:
                id_column = TableColumn(id_column, "varchar(36)", is_null=True)
            else:
                id_column = TableColumn(id_column, "STRING(36)", is_null=True)

        if not isinstance(content_column, TableColumn):
            if dialect == DatabaseDialect.POSTGRESQL:
                content_column = TableColumn(content_column, "text", is_null=True)
            else:
                content_column = TableColumn(
                    content_column, "STRING(MAX)", is_null=True
                )

        if not isinstance(embedding_column, TableColumn):
            if dialect == DatabaseDialect.POSTGRESQL:
                embedding_column = TableColumn(
                    embedding_column, "float8[]", is_null=True
                )
            else:
                embedding_column = TableColumn(
                    embedding_column, "ARRAY<FLOAT64>", is_null=True
                )

        if not embedding_column.vector_length:
            ok_vector_length = vector_length and vector_length > 0
            if ok_vector_length:
                embedding_column.vector_length = vector_length

        configs = [id_column, content_column, embedding_column]

        if column_configs is not None:
            configs.extend(column_configs)

        column_configs = configs

        if primary_key is None:
            primary_key = id_column.name + "," + content_column.name

        if column_configs is not None:
            for column_config in column_configs:
                # Append column name and data type
                column_sql = f"  {column_config.name} {column_config.type}"

                if column_config.vector_length and column_config.vector_length >= 1:
                    column_sql += f"(vector_length=>{column_config.vector_length})"

                # Add nullable constraint if specified
                if not column_config.is_null:
                    column_sql += " NOT NULL"

                # Add a comma and a newline for the next column
                column_sql += ",\n"
                create_table_statement += column_sql

        # Remove the last comma and newline, add closing parenthesis
        if dialect == DatabaseDialect.POSTGRESQL:
            create_table_statement += "  PRIMARY KEY(" + primary_key + ")\n)"
        else:
            create_table_statement = (
                create_table_statement.rstrip(",\n")
                + "\n) PRIMARY KEY("
                + primary_key
                + ")"
            )

        return create_table_statement

    @staticmethod
    def _generate_secondary_indices_ddl_KNN(
        table_name, embedding_column, dialect, secondary_indexes=None
    ):
        if not secondary_indexes:
            return []

        secondary_index_ddl_statements = []
        for secondary_index in secondary_indexes:
            statement = f"CREATE INDEX {secondary_index.index_name} ON {table_name}("
            statement = statement + ",".join(secondary_index.columns) + ")  "

            if dialect == DatabaseDialect.POSTGRESQL:
                statement = statement + "INCLUDE ("
            else:
                statement = statement + "STORING ("

            if secondary_index.storing_columns is None:
                secondary_index.storing_columns = [embedding_column.name]
            elif embedding_column not in secondary_index.storing_columns:
                secondary_index.storing_columns.append(embedding_column.name)

            statement = statement + ",".join(secondary_index.storing_columns) + ")"
            secondary_index_ddl_statements.append(statement)
        return secondary_index_ddl_statements

    @staticmethod
    def _generate_secondary_indices_ddl_ANN(
        table_name, dialect=DatabaseDialect.GOOGLE_STANDARD_SQL, secondary_indexes=[]
    ):
        if dialect != DatabaseDialect.GOOGLE_STANDARD_SQL:
            raise Exception(
                f"ANN is only supported for the GoogleSQL dialect not {dialect}. File an issue on Github?"
            )

        if not secondary_indexes:
            return []

        secondary_index_ddl_statements = []

        for secondary_index in secondary_indexes:
            column_name = secondary_index.columns[0]
            statement = f"CREATE VECTOR INDEX IF NOT EXISTS {secondary_index.index_name}\n\tON {table_name}({column_name})"
            if getattr(secondary_index, "nullable_column", False):
                statement += f"\n\tWHERE {column_name} IS NOT NULL"
            options_segments = [f"distance_type='{secondary_index.distance_type}'"]
            if getattr(secondary_index, "tree_depth", 0) > 0:
                tree_depth = secondary_index.tree_depth
                if tree_depth not in (2, 3):
                    raise Exception(f"tree_depth: {tree_depth} must be either 2 or 3")
                options_segments.append(f"tree_depth={secondary_index.tree_depth}")

            if secondary_index.num_branches > 0:
                options_segments.append(f"num_branches={secondary_index.num_branches}")

            if secondary_index.num_leaves > 0:
                options_segments.append(f"num_leaves={secondary_index.num_leaves}")

            statement += "\n\tOPTIONS(" + ", ".join(options_segments) + ")"
            secondary_index_ddl_statements.append(statement.strip())

        return secondary_index_ddl_statements

    def __init__(
        self,
        instance_id: str,
        database_id: str,
        table_name: str,
        embedding_service: Embeddings,
        id_column: str = ID_COLUMN_NAME,
        content_column: str = CONTENT_COLUMN_NAME,
        embedding_column: str = EMBEDDING_COLUMN_NAME,
        client: Optional[spanner.Client] = None,
        metadata_columns: Optional[List[str]] = None,
        ignore_metadata_columns: Optional[List[str]] = None,
        metadata_json_column: Optional[str] = None,
        query_parameters: QueryParameters = QueryParameters(),
    ):
        """
        Initialize the SpannerVectorStore.

        Parameters:
        - instance_id (str): The ID of the Spanner instance.
        - database_id (str): The ID of the Spanner database.
        - table_name (str): The name of the table.
        - embedding_service (Embeddings): The embedding service.
        - id_column (str): The name of the row ID column. Defaults to ID_COLUMN_NAME.
        - content_column (str): The name of the content column. Defaults to CONTENT_COLUMN_NAME.
        - embedding_column (str): The name of the embedding column. Defaults to EMBEDDING_COLUMN_NAME.
        - client (Client): The Spanner client. Defaults to Client().
        - metadata_columns (Optional[List[str]]): List of metadata columns. Defaults to None.
        - ignore_metadata_columns (Optional[List[str]]): List of metadata columns to ignore. Defaults to None.
        - metadata_json_column (Optional[str]): The generic metadata column. Defaults to None.
        - query_parameters (QueryParameters): The query parameters. Defaults to QueryParameters().
        """
        self._instance_id = instance_id
        self._database_id = database_id
        self._table_name = table_name
        self._client = client_with_user_agent(client, USER_AGENT_VECTOR_STORE)
        self._id_column = id_column
        self._content_column = content_column
        self._embedding_column = embedding_column
        self._metadata_json_column = metadata_json_column

        self._query_parameters = query_parameters
        self._embedding_service = embedding_service

        if metadata_columns is not None and ignore_metadata_columns is not None:
            raise Exception(
                "Either opt-In and pass metadata_column or opt-out and pass ignore_metadata_columns."
            )

        instance = self._client.instance(instance_id)

        if not instance.exists():
            raise Exception("Instance with id-{} doesn't exist.".format(instance_id))

        self._database = instance.database(database_id)

        self._database.reload()

        self._dialect_semantics: DialectSemantics = GoogleSqlSemantics()
        types = self.GSQL_TYPES

        if self._database.database_dialect == DatabaseDialect.POSTGRESQL:
            self._dialect_semantics = PGSqlSemantics()
            types = self.PGSQL_TYPES

        if not self._database.exists():
            raise Exception("Database with id-{} doesn't exist.".format(database_id))

        table = self._database.table(table_name)

        if not table.exists():
            raise Exception("Table with name-{} doesn't exist.".format(table_name))

        column_type_map = self._get_column_type_map(self._database, table_name)

        default_columns = [id_column, content_column, embedding_column]

        columns_to_insert = [] + default_columns

        if ignore_metadata_columns is not None:
            columns_to_insert = [
                element
                for element in column_type_map.keys()
                if element not in ignore_metadata_columns
            ]

            self._metadata_columns = [
                item for item in columns_to_insert if item not in default_columns
            ]
        else:
            self._metadata_columns = []

            if metadata_columns is not None:
                columns_to_insert.extend(metadata_columns)
                self._metadata_columns.extend(metadata_columns)

            if (
                metadata_json_column is not None
                and metadata_json_column not in columns_to_insert
            ):
                columns_to_insert.append(metadata_json_column)
                self._metadata_columns.append(metadata_json_column)

        self._columns_to_insert = columns_to_insert

        self._validate_table_schema(column_type_map, types, default_columns)

    def _get_column_type_map(self, database, table_name):
        query = """
            SELECT column_name, spanner_type, is_nullable
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_NAME = {table_name}
        """.format(
            table_name="'" + table_name + "'"
        )

        with database.snapshot() as snapshot:
            results = snapshot.execute_sql(query)

        column_type_map = {}
        for row in results:
            column_type_map[row[0]] = row

        return column_type_map

    def _validate_table_schema(self, column_type_map, types, default_columns):
        if not all(key in column_type_map for key in self._columns_to_insert):
            raise Exception(
                "One or more columns from list not present in table: {} ",
                self._columns_to_insert,
            )

        if not all(key in column_type_map for key in default_columns):
            raise Exception(
                "One or more columns from the {}, {}, {} are not present in table. Please validate schema.",
                self._id_column,
                self._content_column,
                self._embedding_column,
            )

        content_column_type = column_type_map[self._content_column][1]
        if not any(
            substring.lower() in content_column_type.lower()
            for substring in types[CONTENT_COLUMN_NAME]
        ):
            raise Exception(
                "Content Column is not of correct type. Expected one of: {} but found: {}",
                types[CONTENT_COLUMN_NAME],
                content_column_type,
            )

        embedding_column_type = column_type_map[self._embedding_column][1]
        if not any(
            substring.lower() in embedding_column_type.lower()
            for substring in types[EMBEDDING_COLUMN_NAME]
        ):
            raise Exception(
                "Embedding Column is not of correct type. Expected one of: {} but found: {}".format(
                    types[EMBEDDING_COLUMN_NAME], embedding_column_type
                )
            )

        if self._metadata_json_column is not None:
            metadata_json_column_type = column_type_map[self._metadata_json_column][1]
            allowed_types = types["metadata_json_column"]
            if not any(
                substring.lower() in metadata_json_column_type.lower()
                for substring in allowed_types
            ):
                raise Exception(
                    "Embedding Column is not of correct type. Expected one of: {} but found: {}",
                    allowed_types,
                    embedding_column_type,
                )

        for column_name, column_config in column_type_map.items():
            if column_name not in self._columns_to_insert:
                if "NO" == column_config[2].upper():
                    raise Exception(
                        "Found not nullable constraint on column: {}.",
                        column_name,
                    )

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        if self._query_parameters.distance_strategy == DistanceStrategy.COSINE:
            return self._cosine_relevance_score_fn
        elif self._query_parameters.distance_strategy == DistanceStrategy.EUCLIDEAN:
            return self._euclidean_relevance_score_fn
        else:
            raise Exception(
                "Unknown distance strategy: {}, must be cosine or euclidean.",
                self._query_parameters.distance_strategy,
            )

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        batch_size: int = 5000,
        **kwargs: Any,
    ) -> List[str]:
        """
        Add texts to the vector store index.

        Args:
            texts (Iterable[str]): Iterable of strings to add to the vector store.
            metadatas (Optional[List[dict]]): Optional list of metadatas associated with the texts.
            ids (Optional[List[str]]): Optional list of IDs for the texts.
            batch_size (int): The batch size for inserting data. Defaults to 5000.

        Returns:
            List[str]: List of IDs of the added texts.
        """
        texts_list = list(texts)
        number_of_records = len(texts_list)

        if number_of_records == 0:
            return []

        if ids is not None and len(ids) != number_of_records:
            raise ValueError(
                f"size of list of IDs should be equals to number of documents. Expected: {number_of_records}  but found {len(ids)}"
            )

        if metadatas is not None and len(metadatas) != number_of_records:
            raise ValueError(
                f"size of list of metadatas should be equals to number of documents. Expected: {number_of_records}  but found {len(metadatas)}"
            )

        embeds = self._embedding_service.embed_documents(texts_list)

        if metadatas is None:
            metadatas = [{} for _ in texts]

        values_dict: dict = {key: [] for key in self._metadata_columns}

        if metadatas:
            for row_metadata in metadatas:
                if self._metadata_json_column is not None:
                    row_metadata[self._metadata_json_column] = JsonObject(row_metadata)

                for column_name in self._metadata_columns:
                    if row_metadata.get(column_name) is not None:
                        values_dict[column_name].append(row_metadata[column_name])
                    else:
                        values_dict[column_name].append(None)

        if ids is not None:
            values_dict[self._id_column] = ids

        values_dict[self._content_column] = texts
        values_dict[self._embedding_column] = embeds

        columns_to_insert = values_dict.keys()

        rows_to_insert = [
            [values_dict[key][i] for key in values_dict]
            for i in range(number_of_records)
        ]

        for i in range(0, len(rows_to_insert), batch_size):
            batch = rows_to_insert[i : i + batch_size]
            self._insert_data(batch, columns_to_insert)

        return ids if ids is not None else []

    def _insert_data(self, records, columns_to_insert):
        with self._database.batch() as batch:
            batch.insert_or_update(
                table=self._table_name,
                columns=columns_to_insert,
                values=records,
            )

    def add_documents(
        self,
        documents: List[Document],
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """
        Add documents to the vector store.

        Args:
            documents (List[Document]): Documents to add to the vector store.
            ids (Optional[List[str]]): Optional list of IDs for the documents.

        Returns:
            List[str]: List of IDs of the added texts.
        """
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        return self.add_texts(texts=texts, metadatas=metadatas, ids=ids, **kwargs)

    def delete(
        self,
        ids: Optional[List[str]] = None,
        documents: Optional[List[Document]] = None,
        **kwargs: Any,
    ) -> Optional[bool]:
        """
        Delete records from the vector store.

        Args:
            ids (Optional[List[str]]): List of IDs to delete.
            documents (Optional[List[Document]]): List of documents to delete.

        Returns:
            Optional[bool]: True if deletion is successful, False otherwise, None if not implemented.
        """
        if ids is None and documents is None:
            raise Exception("Pass id/documents to delete")

        columns = []
        values: List[Any] = []

        if ids is not None:
            columns = [self._id_column]
            values = ["('" + value + "')" for value in ids]
        elif documents is not None:
            columns = [self._content_column] + self._metadata_columns

            if self._metadata_json_column is not None:
                columns.remove(self._metadata_json_column)

            for doc in documents:
                value: List[Any] = []
                value.append(doc.page_content)

                for column_name in columns:
                    if column_name != self._content_column:
                        value.append(doc.metadata.get(column_name))

                values.append(value)

        delete_row_count: int = 0

        def delete_records(transaction):
            nonlocal delete_row_count
            base_delete_statement = "DELETE FROM {} WHERE ".format(self._table_name)

            (
                where_clause,
                param_types_map,
            ) = self._dialect_semantics.getDeleteDocumentsParameters(columns)

            # Concatenate the conditions with the base DELETE statement
            sql_delete = base_delete_statement + where_clause

            # Iterate over the list of lists of values
            for value_tuple in values:
                # Construct the params dictionary
                values_tuple_param = (
                    self._dialect_semantics.getDeleteDocumentsValueParameters(
                        columns, value_tuple
                    )
                )

                count = transaction.execute_update(
                    dml=sql_delete,
                    params=values_tuple_param,
                    param_types=param_types_map,
                )

                delete_row_count = delete_row_count + count

        self._database.run_in_transaction(delete_records)

        if delete_row_count > 0:
            return True
        return None

    def similarity_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        pre_filter: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """
        Perform similarity search for a given query.

        Args:
            query (str): The query string.
            k (int): The number of nearest neighbors to retrieve. Defaults to 4.
            pre_filter (Optional[str]): Pre-filter condition for the query. Defaults to None.

        Returns:
            List[Document]: List of documents most similar to the query.
        """
        if self.__using_ANN:
            results, column_order_map = self._get_rows_by_similarity_search_ann(
                embedding,
                k,
                pre_filter,
                **kwargs,
            )
        else:
            results, column_order_map = self._get_rows_by_similarity_search_knn(
                embedding, k, pre_filter
            )

        documents = self._get_documents_from_query_results(
            list(results), column_order_map
        )
        return documents

    def _get_rows_by_similarity_search_ann(
        self,
        embedding: List[float],
        k: int,
        pre_filter: Optional[str] = None,
        index_name: str = "",
        num_leaves: int = 1000,
        embedding_column_is_nullable: bool = False,
        ascending: bool = True,
        return_columns: Optional[List[str]] = None,
    ):
        sql = SpannerVectorStore._generate_sql_for_ANN(
            self._table_name,
            index_name,
            self._embedding_column,
            embedding,
            num_leaves,
            k,
            self._query_parameters.distance_strategy,
            pre_filter=pre_filter,
            embedding_column_is_nullable=embedding_column_is_nullable,
            ascending=ascending,
            return_columns=return_columns or self._columns_to_insert,
        )
        staleness = self._query_parameters.staleness
        with self._database.snapshot(
            **staleness if staleness is not None else {}
        ) as snapshot:
            results = snapshot.execute_sql(sql=sql)
            column_order_map = {
                value: index for index, value in enumerate(self._columns_to_insert)
            }
            return results, column_order_map

    @staticmethod
    def _generate_sql_for_ANN(
        table_name: str,
        index_name: str,
        embedding_column_name: str,
        embedding: List[float],
        num_leaves: int,
        k: int,
        strategy: DistanceStrategy = DistanceStrategy.COSINE,
        pre_filter: Optional[str] = None,
        embedding_column_is_nullable: bool = False,
        ascending: bool = True,
        return_columns: Optional[List[str]] = None,
    ) -> str:
        if not embedding_column_name:
            raise Exception("embedding_column_name must be set")

        ann_strategy_name = GOOGLE_DIALECT_TO_ANN_DISTANCE_FUNCTIONS.get(strategy, None)
        if not ann_strategy_name:
            raise Exception(f"{strategy} is not supported for ANN")

        column_names = None
        if return_columns:
            column_names = ",".join(return_columns)

        if not column_names:
            column_names = "*"

        sql = (
            f"SELECT {column_names} FROM {table_name}"
            + "@{FORCE_INDEX="
            + f"{index_name}"
            + (
                ("}\nWHERE " + ("1=1" if not pre_filter else f"{pre_filter}") + "\n")
                if (not embedding_column_is_nullable)
                else "}\nWHERE "
                + f"{embedding_column_name} IS NOT NULL"
                + ("" if not pre_filter else f" AND {pre_filter}")
                + "\n"
            )
            + f"ORDER BY {ann_strategy_name}(\n"
            + f"  ARRAY<FLOAT32>{embedding}, {embedding_column_name}, options => JSON '"
            + '{"num_leaves_to_search": %s}\')%s\n'
            % (num_leaves, "" if ascending else " DESC")
        )

        if k:
            sql += f"LIMIT {k}"

        return sql.strip()

    def _get_rows_by_similarity_search_knn(
        self,
        embedding: List[float],
        k: int,
        pre_filter: Optional[str] = None,
        **kwargs: Any,
    ):
        staleness = self._query_parameters.staleness

        distance_function = self._dialect_semantics.getDistanceFunction(
            self._query_parameters.distance_strategy
        )

        parameter = ("@vector_embedding", "vector_embedding")

        if self._database.database_dialect == DatabaseDialect.POSTGRESQL:
            parameter = ("$1", "p1")

        select_column_names = ",".join(self._columns_to_insert) + ","
        column_order_map = {
            value: index for index, value in enumerate(self._columns_to_insert)
        }
        column_order_map[KNN_DISTANCE_SEARCH_QUERY_ALIAS] = len(self._columns_to_insert)

        sql_query = """
            SELECT {select_column_names} {distance_function}({embedding_column}, {vector_embedding_placeholder}) AS {distance_alias}
            FROM {table_name}
            WHERE {filter}
            ORDER BY distance
            LIMIT {k_count};
        """.format(
            table_name=self._table_name,
            embedding_column=self._embedding_column,
            select_column_names=select_column_names,
            vector_embedding_placeholder=parameter[0],
            filter=pre_filter if pre_filter is not None else "1 = 1",
            k_count=k,
            distance_function=distance_function,
            distance_alias=KNN_DISTANCE_SEARCH_QUERY_ALIAS,
        )

        with self._database.snapshot(
            **staleness if staleness is not None else {}
        ) as snapshot:
            results = snapshot.execute_sql(
                sql=sql_query,
                params={parameter[1]: embedding},
                param_types={parameter[1]: param_types.Array(param_types.FLOAT64)},
            )

            return list(results), column_order_map

    def _get_documents_from_query_results(
        self, results: List[List], column_order_map: Dict[str, int]
    ) -> List[Tuple[Document, float]]:
        documents = []

        for row in results:
            page_content = row[column_order_map[self._content_column]]

            if (
                self._metadata_json_column is not None
                and row[column_order_map[self._metadata_json_column]]
            ):
                metadata = row[column_order_map[self._metadata_json_column]]
            else:
                metadata = {
                    key: row[column_order_map[key]]
                    for key in self._metadata_columns
                    if row[column_order_map[key]] is not None
                }

            doc = Document(page_content=page_content, metadata=metadata)
            documents.append(
                (doc, row[column_order_map[KNN_DISTANCE_SEARCH_QUERY_ALIAS]])
            )

        return documents

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        pre_filter: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """
        Perform similarity search for a given query.

        Args:
            query (str): The query string.
            k (int): The number of nearest neighbors to retrieve. Defaults to 4.
            pre_filter (Optional[str]): Pre-filter condition for the query. Defaults to None.

        Returns:
            List[Document]: List of documents most similar to the query.
        """
        embedding = self._embedding_service.embed_query(query)
        documents = self.similarity_search_with_score_by_vector(
            embedding=embedding, k=k, pre_filter=pre_filter
        )
        return [doc for doc, _ in documents]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        pre_filter: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """
        Perform similarity search for a given query with scores.

        Args:
            query (str): The query string.
            k (int): The number of nearest neighbors to retrieve. Defaults to 4.
            pre_filter (Optional[str]): Pre-filter condition for the query. Defaults to None.

        Returns:
            List[Tuple[Document, float]]: List of tuples containing Document and similarity score.
        """
        embedding = self._embedding_service.embed_query(query)
        documents = self.similarity_search_with_score_by_vector(
            embedding=embedding, k=k, pre_filter=pre_filter
        )
        return documents

    @property
    def __using_ANN(self):
        return (
            self._query_parameters.algorithm
            == QueryParameters.NearestNeighborsAlgorithm.APPROXIMATE_NEAREST_NEIGHBOR
        )

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        pre_filter: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """
        Perform similarity search by vector.

        Args:
            embedding (List[float]): The embedding vector.
            k (int): The number of nearest neighbors to retrieve. Defaults to 4.
            pre_filter (Optional[str]): Pre-filter condition for the query. Defaults to None.

        Returns:
            List[Document]: List of documents most similar to the query.
        """
        documents = self.similarity_search_with_score_by_vector(
            embedding=embedding,
            k=k,
            pre_filter=pre_filter,
            **kwargs,
        )

        return [doc for doc, _ in documents]

    def max_marginal_relevance_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        pre_filter: Optional[str] = None,
        **kwargs,
    ) -> List[Tuple[Document, float]]:
        """Return docs and their similarity scores selected using the maximal marginal
            relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch before filtering to
                     pass to MMR algorithm.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5.
        Returns:
            List of Documents and similarity scores selected by maximal marginal
                relevance and score for each.
        """
        if self.__using_ANN:
            results, column_order_map = self._get_rows_by_similarity_search_ann(
                embedding,
                fetch_k,
                pre_filter,
                **kwargs,
            )
        else:
            results, column_order_map = self._get_rows_by_similarity_search_knn(
                embedding, fetch_k, pre_filter
            )

        embeddings = [
            result[column_order_map[self._embedding_column]] for result in results
        ]

        mmr_selected = maximal_marginal_relevance(
            np.array([embedding], dtype=np.float32),
            embeddings,
            k=k,
            lambda_mult=lambda_mult,
        )

        search_results = [results[i] for i in mmr_selected]
        documents_with_scores = self._get_documents_from_query_results(
            list(search_results), column_order_map
        )

        return documents_with_scores

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        pre_filter: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5.
        Returns:
            List of Documents selected by maximal marginal relevance.
        """
        documents_with_scores = self.max_marginal_relevance_search_with_score_by_vector(
            embedding, k, fetch_k, lambda_mult, pre_filter
        )

        return [doc for doc, _ in documents_with_scores]

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        pre_filter: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5.
        Returns:
            List of Documents selected by maximal marginal relevance.
        """
        embedding = self._embedding_service.embed_query(query)
        documents = self.max_marginal_relevance_search_by_vector(
            embedding, k, fetch_k, lambda_mult, pre_filter
        )
        return documents

    @classmethod
    def from_documents(  # type: ignore[override]
        cls: Type[SpannerVectorStore],
        documents: List[Document],
        embedding: Embeddings,
        instance_id: str,
        database_id: str,
        table_name: str,
        id_column: str = ID_COLUMN_NAME,
        content_column: str = CONTENT_COLUMN_NAME,
        embedding_column: str = EMBEDDING_COLUMN_NAME,
        ids: Optional[List[str]] = None,
        client: Optional[spanner.Client] = None,
        metadata_columns: Optional[List[str]] = None,
        ignore_metadata_columns: Optional[List[str]] = None,
        metadata_json_column: Optional[str] = None,
        query_parameter: QueryParameters = QueryParameters(),
        **kwargs: Any,
    ) -> SpannerVectorStore:
        """
        Initialize SpannerVectorStore from a list of documents.

        Args:
            documents (List[Document]): List of documents.
            embedding (Embeddings): The embedding service.
            id_column (str): The name of the row ID column. Defaults to ID_COLUMN_NAME.
            content_column (str): The name of the content column. Defaults to CONTENT_COLUMN_NAME.
            embedding_column (str): The name of the embedding column. Defaults to EMBEDDING_COLUMN_NAME.
            ids (Optional[List[str]]): Optional list of IDs for the documents. Defaults to None.
            client (Client): The Spanner client. Defaults to Client().
            metadata_columns (Optional[List[str]]): List of metadata columns. Defaults to None.
            ignore_metadata_columns (Optional[List[str]]): List of metadata columns to ignore. Defaults to None.
            metadata_json_column (Optional[str]): The generic metadata column. Defaults to None.
            query_parameter (QueryParameters): The query parameters. Defaults to QueryParameters().

        Returns:
            SpannerVectorStore: Initialized SpannerVectorStore instance.
        """
        texts = [d.page_content for d in documents]
        metadatas = [d.metadata for d in documents]
        return cls.from_texts(
            texts,
            embedding,
            metadatas=metadatas,
            embedding_service=embedding,
            instance_id=instance_id,
            database_id=database_id,
            table_name=table_name,
            id_column=id_column,
            content_column=content_column,
            embedding_column=embedding_column,
            client=client,
            ids=ids,
            metadata_columns=metadata_columns,
            ignore_metadata_columns=ignore_metadata_columns,
            metadata_json_column=metadata_json_column,
            query_parameter=query_parameter,
            kwargs=kwargs,
        )

    @classmethod
    def from_texts(  # type: ignore[override]
        cls: Type[SpannerVectorStore],
        texts: List[str],
        embedding: Embeddings,
        instance_id: str,
        database_id: str,
        table_name: str,
        metadatas: Optional[List[dict]] = None,
        id_column: str = ID_COLUMN_NAME,
        content_column: str = CONTENT_COLUMN_NAME,
        embedding_column: str = EMBEDDING_COLUMN_NAME,
        ids: Optional[List[str]] = None,
        client: Optional[spanner.Client] = None,
        metadata_columns: Optional[List[str]] = None,
        ignore_metadata_columns: Optional[List[str]] = None,
        metadata_json_column: Optional[str] = None,
        query_parameter: QueryParameters = QueryParameters(),
        **kwargs: Any,
    ) -> SpannerVectorStore:
        """
        Initialize SpannerVectorStore from a list of texts.

        Args:
            texts (List[str]): List of texts.
            embedding (Embeddings): The embedding service.
            metadatas (Optional[List[dict]]): Optional list of metadatas associated with the texts. Defaults to None.
            id_column (str): The name of the row ID column. Defaults to ID_COLUMN_NAME.
            content_column (str): The name of the content column. Defaults to CONTENT_COLUMN_NAME.
            embedding_column (str): The name of the embedding column. Defaults to EMBEDDING_COLUMN_NAME.
            ids (Optional[List[str]]): Optional list of IDs for the texts. Defaults to None.
            client (Client): The Spanner client. Defaults to Client().
            metadata_columns (Optional[List[str]]): List of metadata columns. Defaults to None.
            ignore_metadata_columns (Optional[List[str]]): List of metadata columns to ignore. Defaults to None.
            metadata_json_column (Optional[str]): The generic metadata column. Defaults to None.
            query_parameter (QueryParameters): The query parameters. Defaults to QueryParameters().

        Returns:
            SpannerVectorStore: Initialized SpannerVectorStore instance.
        """
        store = cls(
            instance_id=instance_id,
            database_id=database_id,
            table_name=table_name,
            embedding_service=embedding,
            id_column=id_column,
            content_column=content_column,
            embedding_column=embedding_column,
            client=client,
            metadata_columns=metadata_columns,
            ignore_metadata_columns=ignore_metadata_columns,
            metadata_json_column=metadata_json_column,
            query_parameters=query_parameter,
        )

        store.add_texts(texts=texts, metadatas=metadatas, ids=ids)

        return store
