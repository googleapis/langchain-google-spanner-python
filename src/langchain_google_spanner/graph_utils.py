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


from __future__ import annotations

import re


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
