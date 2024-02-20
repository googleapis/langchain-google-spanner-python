import pytest
from google.cloud.spanner import Client, KeySet  # type: ignore

from langchain_google_spanner.vector_store import SpannerVectorStore, TableColumn

project_id = "span-cloud-testing"
instance = "vertex-vector-search-tests"
google_database = "langchain-test"
pg_database =  "langchain-pg-db"
table_name = "vectortest1"

OPERATION_TIMEOUT_SECONDS = 240


@pytest.fixture(scope="module")
def client() -> Client:
  return Client(project=project_id)

@pytest.fixture(autouse=True)
def cleanupGSQL(client):
    print("\nPerforming cleanup after each test...")
    
    database = client.instance(instance).database(google_database)
    operation = database.update_ddl([f"DROP TABLE IF EXISTS {table_name}"])
    operation.result(OPERATION_TIMEOUT_SECONDS)

    yield

    # Code to perform teardown after each test goes here
    print("\nCleanup complete.")

@pytest.fixture(autouse=True)
def cleanupPGSQL(client):
    print("\nPerforming cleanup after each test...")
    
    database = client.instance(instance).database(pg_database)
    operation = database.update_ddl([f"DROP TABLE IF EXISTS {table_name}"])
    operation.result(OPERATION_TIMEOUT_SECONDS)

    yield

    # Code to perform teardown after each test goes here
    print("\nCleanup complete.")



class TestStaticUtilityGoogleSQL: 
    @pytest.fixture(autouse=True)
    def setup_database(self, client):
        yield

    def test_init_vector_store_table1(self):
        SpannerVectorStore.init_vector_store_table(
            instance_id= instance,
            database_id= google_database,
            table_name= table_name,
            metadata_columns=[TableColumn(name="product_name", type="STRING(1024)", is_null=False)
                            , TableColumn(name="title",  type="STRING(1024)")
                            , TableColumn(name="price",  type="INT64")])

    def test_init_vector_store_table2(self):
        SpannerVectorStore.init_vector_store_table(
            instance_id= instance,
            database_id= google_database,
            table_name= table_name,
            id_column= "custom_id1",
            content_column= "custom_content_id1",
            embedding_column= "custom_embedding_id1",
            metadata_columns=[TableColumn(name="product_name", type="STRING(1024)", is_null=False)
                            , TableColumn(name="title",  type="STRING(1024)")
                            , TableColumn(name="price",  type="INT64")])

    def test_init_vector_store_table3(self):
        SpannerVectorStore.init_vector_store_table(
            instance_id= instance,
            database_id= google_database,
            table_name= table_name,
            id_column= TableColumn(name="product_id", type="STRING(1024)", is_null=False),
            embedding_column= TableColumn(name="custom_embedding_id1", type="ARRAY<FLOAT64>", is_null=True),
            metadata_columns=[TableColumn(name="product_name", type="STRING(1024)", is_null=False)
                            , TableColumn(name="title",  type="STRING(1024)")
                            , TableColumn(name="metadata_json_column",  type="JSON")])

    def test_init_vector_store_table4(self):
        SpannerVectorStore.init_vector_store_table(
            instance_id= instance,
            database_id= google_database,
            table_name= table_name,
            id_column= TableColumn(name="product_id", type="STRING(1024)", is_null=False),
            embedding_column= TableColumn(name="custom_embedding_id1", type="ARRAY<FLOAT64>", is_null=True),
            metadata_columns=[TableColumn(name="product_name", type="STRING(1024)", is_null=False)
                            , TableColumn(name="title",  type="STRING(1024)")
                            , TableColumn(name="metadata_json_column",  type="JSON")],
                            primary_key= "product_name, title, product_id") 
        

class TestStaticUtilityPGSQL: 
    @pytest.fixture(autouse=True)
    def setup_database(self, client):
        yield

    def test_init_vector_store_table1(self):
        SpannerVectorStore.init_vector_store_table(
            instance_id= instance,
            database_id= pg_database,
            table_name= table_name,
            metadata_columns=[TableColumn(name="product_name", type="TEXT", is_null=False)
                            , TableColumn(name="title",  type="varchar(36)")
                            , TableColumn(name="price",  type="bigint")])

    def test_init_vector_store_table2(self):
        SpannerVectorStore.init_vector_store_table(
            instance_id= instance,
            database_id= pg_database,
            table_name= table_name,
            id_column= "custom_id1",
            content_column= "custom_content_id1",
            embedding_column= "custom_embedding_id1",
            metadata_columns=[TableColumn(name="product_name", type="TEXT", is_null=False)
                            , TableColumn(name="title",  type="varchar(36)")
                            , TableColumn(name="price",  type="bigint")])

    def test_init_vector_store_table3(self):
        SpannerVectorStore.init_vector_store_table(
            instance_id= instance,
            database_id= pg_database,
            table_name= table_name,
            id_column= TableColumn(name="product_id", type="varchar(36)", is_null=False),
            embedding_column= TableColumn(name="custom_embedding_id1", type="float8[]", is_null=True),
            metadata_columns=[TableColumn(name="product_name", type="TEXT", is_null=False)
                            , TableColumn(name="title",  type="varchar(36)")
                            , TableColumn(name="price",  type="bigint")])

    def test_init_vector_store_table4(self):
        SpannerVectorStore.init_vector_store_table(
            instance_id= instance,
            database_id= pg_database,
            table_name= table_name,
            id_column= TableColumn(name="product_id", type="varchar(36)", is_null=False),
            embedding_column= TableColumn(name="custom_embedding_id1", type="float8[]", is_null=True),
            metadata_columns=[TableColumn(name="product_name", type="TEXT", is_null=False)
                            , TableColumn(name="title",  type="varchar(36)")
                            , TableColumn(name="price",  type="bigint")],
                primary_key= "product_name, title, product_id")
    