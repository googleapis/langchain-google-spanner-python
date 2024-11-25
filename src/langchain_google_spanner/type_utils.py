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
from typing import Any

from google.cloud.spanner_v1 import param_types


class TypeUtility(object):
    """A utiltiy class with helper functions that do type conversions related to Spanner."""

    @staticmethod
    def spanner_type_to_schema_str(
        t: param_types.Type,
        include_type_annotations: bool = False,
    ) -> str:
        """Returns a Spanner string representation of a Spanner type.

        Parameters:
        - t: spanner.param_types.Type;
        - include_type_annotations: boolean indicates whether to include type
          annotations.

        Returns:
        - str: a Spanner string representation of a Spanner type.
        """
        if t.code == param_types.TypeCode.ARRAY:
            return "ARRAY<{}>".format(
                TypeUtility.spanner_type_to_schema_str(
                    t.array_element_type,
                    include_type_annotations=include_type_annotations,
                )
            )
        if t.code == param_types.TypeCode.BOOL:
            return "BOOL"
        if t.code == param_types.TypeCode.INT64:
            return "INT64"
        if t.code == param_types.TypeCode.STRING:
            return "STRING(MAX)" if include_type_annotations else "STRING"
        if t.code == param_types.TypeCode.BYTES:
            return "BYTES(MAX)" if include_type_annotations else "BYTES"
        if t.code == param_types.TypeCode.FLOAT32:
            return "FLOAT32"
        if t.code == param_types.TypeCode.FLOAT64:
            return "FLOAT64"
        if t.code == param_types.TypeCode.TIMESTAMP:
            return "TIMESTAMP"
        raise ValueError("Unsupported type: %s" % t)

    @staticmethod
    def schema_str_to_spanner_type(s: str) -> param_types.Type:
        """Returns a Spanner type corresponding to the string representation from Spanner schema type.

        Parameters:
        - s: string representation of a Spanner schema type.

        Returns:
        - Type[Any]: the corresponding Spanner type.
        """
        if s == "BOOL":
            return param_types.BOOL
        if s == "INT64":
            return param_types.INT64
        if s == "STRING":
            return param_types.STRING
        if s == "BYTES":
            return param_types.BYTES
        if s == "FLOAT64":
            return param_types.FLOAT64
        if s == "FLOAT32" or s == "FLOAT":
            return param_types.FLOAT32
        if s == "TIMESTAMP":
            return param_types.TIMESTAMP
        if s.startswith("ARRAY<") and s.endswith(">"):
            return param_types.Array(
                TypeUtility.schema_str_to_spanner_type(s[len("ARRAY<") : -len(">")])
            )
        raise ValueError("Unsupported type: %s" % s)

    @staticmethod
    def value_to_param_type(v: Any) -> param_types.Type:
        """Returns a Spanner type corresponding to the python value.

        Parameters:
        - v: a python value.

        Returns:
        - Type[Any]: the corresponding Spanner type.
        """
        if isinstance(v, bool):
            return param_types.BOOL
        if isinstance(v, int):
            return param_types.INT64
        if isinstance(v, str):
            return param_types.STRING
        if isinstance(v, bytes):
            return param_types.BYTES
        if isinstance(v, float):
            return param_types.FLOAT64
        if isinstance(v, datetime.datetime):
            return param_types.TIMESTAMP
        if isinstance(v, list):
            if len(v) == 0:
                raise ValueError("Unknown element type of empty array")
            return param_types.Array(TypeUtility.value_to_param_type(v[0]))
        raise ValueError("Unsupported type of param: {}".format(v))
