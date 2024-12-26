Spanner for LangChain
=================================

|preview| |pypi| |versions|

- `Client Library Documentation`_
- `Product Documentation`_

.. |preview| image:: https://img.shields.io/badge/support-preview-orange.svg
   :target: https://cloud.google.com/products#product-launch-stages
.. |pypi| image:: https://img.shields.io/pypi/v/langchain-google-spanner.svg
   :target: https://pypi.org/project/langchain-google-spanner/
.. |versions| image:: https://img.shields.io/pypi/pyversions/langchain-google-spanner.svg
   :target: https://pypi.org/project/langchain-google-spanner/
.. _Client Library Documentation: https://cloud.google.com/python/docs/reference/langchain-google-spanner/latest
.. _Product Documentation: https://cloud.google.com/spanner

Quick Start
-----------

In order to use this library, you first need to go through the following
steps:

1. `Select or create a Cloud Platform project.`_
2. `Enable billing for your project.`_
3. `Enable the Google Cloud Spanner API.`_
4. `Setup Authentication.`_

.. _Select or create a Cloud Platform project.: https://console.cloud.google.com/project
.. _Enable billing for your project.: https://cloud.google.com/billing/docs/how-to/modify-project#enable_billing_for_a_project
.. _Enable the Google Cloud Spanner API.: https://console.cloud.google.com/flows/enableapi?apiid=spanner.googleapis.com
.. _Setup Authentication.: https://googleapis.dev/python/google-api-core/latest/auth.html

Installation
~~~~~~~~~~~~

Install this library in a `virtualenv`_ using pip. `virtualenv`_ is a tool to create isolated Python environments. The basic problem it addresses is
one of dependencies and versions, and indirectly permissions.

With `virtualenv`_, it’s possible to install this library without needing system install permissions, and without clashing with the installed system dependencies.

.. _`virtualenv`: https://virtualenv.pypa.io/en/latest/

Supported Python Versions
^^^^^^^^^^^^^^^^^^^^^^^^^

Python >= 3.9

Mac/Linux
^^^^^^^^^

.. code-block:: console

   pip install virtualenv
   virtualenv <your-env>
   source <your-env>/bin/activate
   <your-env>/bin/pip install langchain-google-spanner

Windows
^^^^^^^

.. code-block:: console

   pip install virtualenv
   virtualenv <your-env>
   <your-env>\Scripts\activate
   <your-env>\Scripts\pip.exe install langchain-google-spanner

Vector Store Usage
~~~~~~~~~~~~~~~~~~~

Use a vector store to store embedded data and perform vector search.

.. code-block:: python

    from langchain_google_sapnner import SpannerVectorstore
    from langchain.embeddings import VertexAIEmbeddings

    embeddings_service = VertexAIEmbeddings(model_name="textembedding-gecko@003")
    vectorstore = SpannerVectorStore(
        instance_id="my-instance",
        database_id="my-database",
        table_name="my-table",
        embeddings=embedding_service
    )

See the full `Vector Store`_ tutorial.

.. _`Vector Store`: https://github.com/googleapis/langchain-google-spanner-python/blob/main/docs/vector_store.ipynb

Document Loader Usage
~~~~~~~~~~~~~~~~~~~~~

Use a document loader to load data as LangChain ``Document``\ s.

.. code-block:: python

   from langchain_google_spanner import SpannerLoader


    loader = SpannerLoader(
        instance_id="my-instance",
        database_id="my-database",
        query="SELECT * from my_table_name"
    )
    docs = loader.lazy_load()

See the full `Document Loader`_ tutorial.

.. _`Document Loader`: https://github.com/googleapis/langchain-google-spanner-python/blob/main/docs/document_loader.ipynb

Chat Message History Usage
~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``ChatMessageHistory`` to store messages and provide conversation
history to LLMs.

.. code:: python

   from langchain_google_spanner import SpannerChatMessageHistory


    history = SpannerChatMessageHistory(
        instance_id="my-instance",
        database_id="my-database",
        table_name="my_table_name",
        session_id="my-session_id"
    )

See the full `Chat Message History`_ tutorial.

.. _`Chat Message History`: https://github.com/googleapis/langchain-google-spanner-python/blob/main/docs/chat_message_history.ipynb

Spanner Graph Store Usage
~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``SpannerGraphStore`` to store nodes and edges extracted from documents.

.. code:: python

   from langchain_google_spanner import SpannerGraphStore


    graph = SpannerGraphStore(
        instance_id="my-instance",
        database_id="my-database",
        graph_name="my_graph",
    )

See the full `Spanner Graph Store`_ tutorial.

.. _`Spanner Graph Store`: https://github.com/googleapis/langchain-google-spanner-python/blob/main/docs/graph_store.ipynb

Spanner Graph QA Chain Usage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``SpannerGraphQAChain`` for question answering over a graph stored in Spanner Graph.

.. code:: python

    from langchain_google_spanner import SpannerGraphStore, SpannerGraphQAChain
    from langchain_google_vertexai import ChatVertexAI


    graph = SpannerGraphStore(
        instance_id="my-instance",
        database_id="my-database",
        graph_name="my_graph",
    )
    llm = ChatVertexAI()
    chain = SpannerGraphQAChain.from_llm(
        llm,
        graph=graph,
        allow_dangerous_requests=True
    )
    chain.invoke("query=Where does Sarah's sibling live?")

See the full `Spanner Graph QA Chain`_ tutorial.

.. _`Spanner Graph QA Chain`: https://github.com/googleapis/langchain-google-spanner-python/blob/main/docs/graph_qa_chain.ipynb

Spanner Graph Retrievers Usage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``SpannerGraphTextToGQLRetriever`` to translate natural language question to GQL and query SpannerGraphStore.

.. code:: python

    from langchain_google_spanner import SpannerGraphStore, SpannerGraphTextToGQLRetriever
    from langchain_google_vertexai import ChatVertexAI


    graph = SpannerGraphStore(
        instance_id="my-instance",
        database_id="my-database",
        graph_name="my_graph",
    )
    llm = ChatVertexAI()
    retriever = SpannerGraphTextToGQLRetriever.from_params(
        graph_store=graph,
        llm=llm
    )
    retriever.invoke("Where does Elias Thorne's sibling live?")

Use ``SpannerGraphVectorContextRetriever`` to perform vector search on embeddings that are stored in the nodes in a SpannerGraphStore. If expand_by_hops is provided, the nodes and edges at a distance upto the expand_by_hops from the nodes found in the vector search will also be returned.

.. code:: python

    from langchain_google_spanner import SpannerGraphStore, SpannerGraphVectorContextRetriever
    from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings


    graph = SpannerGraphStore(
        instance_id="my-instance",
        database_id="my-database",
        graph_name="my_graph",
    )
    embedding_service = VertexAIEmbeddings(model_name="text-embedding-004")
    retriever = SpannerGraphVectorContextRetriever.from_params(
            graph_store=graph,
            embedding_service=embedding_service,
            label_expr="Person",
            embeddings_column="embeddings",
            top_k=1,
            expand_by_hops=1,
        )
    retriever.invoke("Who lives in desert?")


Contributions
~~~~~~~~~~~~~

Contributions to this library are always welcome and highly encouraged.

See `CONTRIBUTING`_ for more information how to get started.

Please note that this project is released with a Contributor Code of Conduct. By participating in
this project you agree to abide by its terms. See `Code of Conduct`_ for more
information.

.. _`CONTRIBUTING`: https://github.com/googleapis/langchain-google-spanner-python/blob/main/CONTRIBUTING.md
.. _`Code of Conduct`: https://github.com/googleapis/langchain-google-spanner-python/blob/main/CODE_OF_CONDUCT.md

License
-------

Apache 2.0 - See
`LICENSE <https://github.com/googleapis/langchain-google-spanner-python/blob/main/LICENSE>`_
for more information.

Disclaimer
----------

This is not an officially supported Google product.


Limitations
----------

* Approximate Nearest Neighbors (ANN) strategies are only supported for the GoogleSQL dialect
* ANN's `ALTER VECTOR INDEX` is not yet supported by [Google Cloud Spanner](https://cloud.google.com/spanner/docs/find-approximate-nearest-neighbors#limitations)
