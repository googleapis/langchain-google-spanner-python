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

import datetime
from typing import List, Optional

from google.cloud.spanner import Client, KeySet
from langchain_community.document_loaders.base import BaseLoader
from langchain_core.documents import Document

CONTENT_COL_NAME = "page_content"
METADATA_COL_NAME = "langchain_metadata"


class SpannerLoader(BaseLoader):
    """Loads data from Google CLoud Spanner."""

    def __init__(
        self,
        instance: str,
        database: str,
        query: str,
        client: Optional[Client] = Client(),
        staleness: Optional[int] = 0,
        content_columns: List[str] = [],
        metadata_columns: List[str] = [],
        format: str = "text",
        databoost: bool = False,
    ):
        """Initialize Spanner document loader.

        Args:
            instance: The Spanner instance to load data from.
            database: The Spanner database to load data from.
            query: A GoogleSQL or PostgreSQL query. Users must match dialect to their database.
            client: Optional. The connection object to use. This can be used to customize project id and credentials.
            staleness: Optional. The time bound for stale read.
            content_columns: The list of column(s) or field(s) to use for a Document's page content.
                              Page content is the default field for embeddings generation.
            metadata_columns: The list of column(s) or field(s) to use for metadata.
            format: Set the format of page content if using multiple columns or fields.
                    Format included: 'text', 'JSON', 'YAML', 'CSV'.
            databoost: Use data boost on read. Note: needs extra IAM permissions and higher cost.
        """
        self.instance = instance
        self.database = database
        self.query = query
        self.client = client
        self.staleness = staleness
        self.content_columns = content_columns
        self.metadata_columns = metadata_columns
        self.format = format
        formats = ["JSON", "text", "YAML", "CSV"]
        if self.format not in formats:
            raise Exception("Use on of 'text', 'JSON', 'YAML', 'CSV'")
        self.databoost = databoost
        if not self.client.instance(self.instance).exists():
            raise Exception("Instance doesn't exist.")
        if not self.client.instance(self.instance).database(self.database).exists():
            raise Exception("Database doesn't exist.")

    def load(self) -> List[Document]:
        """Load documents."""
        return list(self.lazy_load())

    def lazy_load(self) -> List[Document]:
        """A lazy loader for Documents."""
        instance = self.client.instance(self.instance)
        db = instance.database(self.database)
        duration = datetime.timedelta(seconds=self.staleness)
        with db.snapshot(exact_staleness=duration) as snapshot:
            keyset = KeySet(all_=True)
            try:
                results = snapshot.execute_sql(
                    sql=self.query, data_boost_enabled=self.databoost
                ).to_dict_list()
            except:
                raise Exception("Fail to execute query")
            formatted_results = [self.load_row_to_document(row) for row in results]
            print(formatted_results)
            yield formatted_results

    def load_row_to_document(self, row):
        include_headers = ["JSON", "YAML"]
        page_content = ""
        if self.format in include_headers:
            page_content = f"{CONTENT_COL_NAME}: "
        page_content += row[CONTENT_COL_NAME]
        if not page_content:
            raise Exception("column page_content doesn't exist.")

        for c in self.content_columns:
            if self.format in include_headers:
                page_content += f"\n{c}: "
            page_content += f" {row[c]}"

        if self.metadata_columns:
            metadata = {}
            for m in self.metadata_columns:
                metadata = {**metadata, **row[m]}
        else:
            metadata = row[METADATA_COL_NAME]
            for k in row:
                if (k != CONTENT_COL_NAME) and (k != METADATA_COL_NAME):
                    metadata[k] = row[k]

        return Document(page_content=page_content, metadata=metadata)
