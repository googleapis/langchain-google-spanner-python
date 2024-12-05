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

GQL_EXAMPLES = """
The following query in backtick matches all persons in the graph FinGraph
whose birthday is before 1990-01-10 and
returns their name and birthday.
```
GRAPH FinGraph
MATCH (p:Person WHERE p.birthday < '1990-01-10')
RETURN p.name as name, p.birthday as birthday;
```

The following query in backtick finds the owner of the account with the most
incoming transfers by chaining multiple graph linear statements together.
```
GRAPH FinGraph
MATCH (:Account)-[:Transfers]->(account:Account)
RETURN account, COUNT(*) AS num_incoming_transfers
GROUP BY account
ORDER BY num_incoming_transfers DESC
LIMIT 1

NEXT

MATCH (account:Account)<-[:Owns]-(owner:Person)
RETURN account.id AS account_id, owner.name AS owner_name, num_incoming_transfers;
```

The following query finds all the destination accounts one to three transfers
away from a source Account with id equal to 7.
```
GRAPH FinGraph
MATCH (src:Account {{id: 7}})-[e:Transfers]->{{1, 3}}(dst:Account)
RETURN src.id AS src_account_id, dst.id AS dst_account_id;
```
Carefully note the syntax in the example above for path quantification,
that it is `[e:Transfers]->{{1, 3}}` and NOT `[e:Transfers*1..3]->`
"""

DEFAULT_GQL_TEMPLATE_PART0 = """
Create an Spanner Graph GQL query for the question using the schema.
{gql_examples}
"""

DEFAULT_GQL_TEMPLATE_PART1 = """
Instructions:
Mention the name of the graph at the beginning.
Use only nodes and edge types, and properties included in the schema.
Do not use any node and edge type, or properties not included in the schema.
Always alias RETURN values.

Question: {question}
Schema: {schema}

Note:
Do not include any explanations or apologies.
Do not prefix query with `gql`
Do not include any backticks.
Start with GRAPH <graphname>
Output only the query statement.
Do not output any query that tries to modify or delete data.
"""

DEFAULT_GQL_TEMPLATE = (
    DEFAULT_GQL_TEMPLATE_PART0.format(gql_examples=GQL_EXAMPLES)
    + DEFAULT_GQL_TEMPLATE_PART1
)

VERIFY_EXAMPLES = """
Examples:
1.
question: Which movie has own the Oscar award in 1996?
generated_gql:
  GRAPH moviedb
  MATCH (m:movie)-[:own_award]->(a:award {{name:"Oscar", year:1996}})
  RETURN m.name

graph_schema:
{{
"Edges": {{
    "produced_by": "From movie nodes to producer nodes",
    "acts": "From actor nodes to movie nodes",
    "has_coacted_with": "From actor nodes to actor nodes",
    "own_award": "From actor nodes to award nodes"
  }}
}}

The verified gql fixes the missing node 'actor'
  MATCH (m:movie)<-[:acts]-(a:actor)-[:own_award]->(a:award {{name:"Oscar", year:1996}})
  RETURN m.name

2.
question: Which movies have been produced by production house ABC Movies?
generated_gql:
  GRAPH moviedb
  MATCH (p:producer {{name:"ABC Movies"}})-[:produced_by]->(m:movie)
  RETURN p.name

graph_schema:
{{
"Edges": {{
    "produced_by": "From movie nodes to producer nodes",
    "acts": "From actor nodes to movie nodes",
    "references": "From movie nodes to movie nodes",
    "own_award": "From actor nodes to award nodes"
  }}
}}

The verified gql fixes the edge direction:
  GRAPH moviedb
  MATCH (p:producer {{name:"ABC Movies"}})<-[:produced_by]-(m:movie)
  RETURN m.name

3.
question: Which movie references the movie "XYZ" via at most 3 hops ?
graph_schema:
{{
"Edges": {{
    "produced_by": "From movie nodes to producer nodes",
    "acts": "From actor nodes to movie nodes",
    "references": "From movie nodes to movie nodes",
    "own_award": "From actor nodes to award nodes"
  }}
}}

generated_gql:
  GRAPH moviedb
  MATCH (m:movie)-[:references*1..3]->(:movie {{name="XYZ"}})
  RETURN m.name

The path quantification syntax [:references*1..3] is wrong.
The verified gql fixes the path quantification syntax:
  GRAPH moviedb
  MATCH (m:movie)-[:references]->{{1, 3}}(:movie {{name="XYZ"}})
  RETURN m.name
"""

DEFAULT_GQL_VERIFY_TEMPLATE_PART0 = """
Given a natual language question, Spanner Graph GQL graph query and a graph schema,
validate the query.

{verify_examples}
"""

DEFAULT_GQL_VERIFY_TEMPLATE_PART1 = """
Instructions:
Add missing nodes and edges in the query if required.
Fix the path quantification syntax if required.
Carefully check the syntax.
Fix the query if required. There could be more than one correction.
Optimize if possible.
Do not make changes if not required.
Think in steps. Add the explanation in the output.

Question : {question}
Input gql: {generated_gql}
Schema: {graph_schema}

{format_instructions}
"""

DEFAULT_GQL_VERIFY_TEMPLATE = (
    DEFAULT_GQL_VERIFY_TEMPLATE_PART0.format(verify_examples=VERIFY_EXAMPLES)
    + DEFAULT_GQL_VERIFY_TEMPLATE_PART1
)

DEFAULT_GQL_FIX_TEMPLATE_PART0 = """
We generated a Spanner Graph GQL query to answer a natural language question.
Question: {question}
However the generated Spanner Graph GQL query is not valid.  ```
Input gql: {generated_gql}
```
The error obtained when executing the query is
```
{err_msg}
```
Give me a correct version of the query.
Do not generate the same query as the input gql.
"""

DEFAULT_GQL_FIX_TEMPLATE_PART1 = """
Examples of correct query :
{gql_examples}"""

DEFAULT_GQL_FIX_TEMPLATE_PART2 = """
Instructions:
Mention the name of the graph at the beginning.
Use only nodes and edge types, and properties included in the schema.
Do not use any node and edge type, or properties not included in the schema.
Do not generate the same query as the input gql.
Schema: {schema}

Note:
Do not include any explanations or apologies.
Do not prefix query with `gql`
Do not include any backticks.
Start with GRAPH <graphname>
Output only the query statement.
Do not output any query that tries to modify or delete data.
"""

DEFAULT_GQL_FIX_TEMPLATE = (
    DEFAULT_GQL_FIX_TEMPLATE_PART0
    + DEFAULT_GQL_FIX_TEMPLATE_PART1.format(gql_examples=GQL_EXAMPLES)
    + DEFAULT_GQL_FIX_TEMPLATE_PART2
)

SPANNERGRAPH_QA_TEMPLATE = """
You are a helpful AI assistant.
Create a human readable answer for the for the question.
You should only use the information provided in the context and not use your internal knowledge.
Don't add any information.
Here is an example:

Question: Which funds own assets over 10M?
Context:[name:ABC Fund, name:Star fund]"
Helpful Answer: ABC Fund and Star fund have assets over 10M.

Follow this example when generating answers.
If the provided information is empty, say that you don't know the answer.
You are given the following information:
- `Question`: the natural language question from the user
- `Graph Schema`: contains the schema of the graph database
- `Graph Query`: A Spanner Graph GQL query equivalent of the question from the user used to extract context from the graph database
- `Context`: The response from the graph database as context
Information:
Question: {question}
Graph Schema: {graph_schema}
Graph Query: {graph_query}
Context: {context}

Helpful Answer:"""
