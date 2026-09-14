"""
Top-level orchestration: wires ingest -> chunk -> embed -> store, and
separately, query -> retrieve -> generate.

Two entry points kept separate because indexing the corpus and answering
question are different operations with different frequencies.
"""

from scripts import fetch_corpus
from src import config, ingest, chunker, embedder, vectorstore


def build_index(strategy: str) -> None:
    """
    Full indexing pipeline: load raw corpus -> chunk -> embed -> write
    to vector store.
    """

    fingerprint = (
        f"{strategy}_{config.CHUNKING_STRATEGY_VERSION}_{config.CHUNK_SIZE}_{config.CHUNK_OVERLAP}"
        f"_{config.EMBEDDING_MODEL_NAME}"
    )

    # 1. Fetch raw files to data/raw/ (skips files already on disk)
    fetch_corpus.main()

    # 2. Parse raw files into clean text + metadata
    documents = ingest.load_corpus(config.DATA_RAW_DIR)
    print(f"Loaded {len(documents)} documents")

    # 3. Skip documents already chunked/embedded/stored under the current
    # chunking strategy + params + embedding model. This helps to avoid 
    # re-paying for embeddings and re-writing chunks that haven't changed.
    collection = vectorstore.get_or_create_collection(config.VECTOR_STORE_COLLECTION_NAME)
    already_indexed = vectorstore.get_indexed_doc_ids(collection, fingerprint)

    # Create an empty list to hold the documents we actually need to process
    unprocessed_documents = []

    # Iterate through the original list of documents
    for doc in documents:
        # Check if the document's ID already exist in the collection
        if doc.doc_id not in already_indexed:
            # If it is new, add it to our new list
            unprocessed_documents.append(doc)

    print(f"{len(already_indexed)} document(s) already indexed for '{fingerprint}', "
          f"{len(unprocessed_documents)} remaining to process")

    if not unprocessed_documents:
        print("Nothing new to index.")
        return

    # 4. Split each document into chunks based on the given strategy
    chunks = []

    match strategy:
        case "naive":
            for doc in unprocessed_documents:
                chunks.extend(
                    chunker.naive_fixed_size(doc, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
                )
        case "recursive":
            for doc in unprocessed_documents:
                chunks.extend(
                    chunker.recursive_overlap_chunk(doc, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
                )
        case "structure_aware":
            for doc in unprocessed_documents:
                chunks.extend(
                    chunker.structure_aware_chunk(doc, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
                )
        case _:
            print("Unknown chunking strategy")

    print(f"Split into {len(chunks)} chunks")

    # 5. Embed chunk text
    vectors = embedder.embed_chunks(chunks, config.EMBEDDING_MODEL_NAME)
    print(f"Embedded {len(vectors)} chunks")

    # 6. Write chunks + vectors to the vector store
    collection = vectorstore.get_or_create_collection(collection_name=config.VECTOR_STORE_COLLECTION_NAME)
    vectorstore.add_chunks(collection=collection, chunks=chunks , vectors=vectors, fingerprint=fingerprint)
    print(f"Wrote {len(chunks)} chunks to '{config.VECTOR_STORE_COLLECTION_NAME}'")

def answer_question(strategy: str, question: str, top_k: int = 5) -> dict:
    """
    Full query pipeline: embed the question -> retrieve top_k chunks ->
    Returns a dict: {"question": question, "retrieved_chunks": [...], "answer": ...}
    """

    fingerprint = (
        f"{strategy}_{config.CHUNKING_STRATEGY_VERSION}_{config.CHUNK_SIZE}_{config.CHUNK_OVERLAP}"
        f"_{config.EMBEDDING_MODEL_NAME}"
    )

    # 1. Embed the question with the exact same embedder used for chunks
    question_chunk = chunker.Chunk(chunk_id="query", doc_id="query", text=question, metadata={})
    query_embedding = embedder.embed_chunks([question_chunk], config.EMBEDDING_MODEL_NAME)[0]

    # 2. Retrieve the top_k nearest chunks from the vector store
    collection = vectorstore.get_or_create_collection(config.VECTOR_STORE_COLLECTION_NAME)
    retrieved_chunks = vectorstore.query_collection(collection, query_embedding, top_k, fingerprint=fingerprint)

    return {
        "question": question,
        "retrieved_chunks": retrieved_chunks,
        "answer": None,
    }

