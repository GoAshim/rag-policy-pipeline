"""
Vector store: to store the embedded vectors in the chosen vector
database (Weaviate), and to retrieve vectors matching to the 
user query.
"""

# Import the class/method to get the chunks
from src.chunker import Chunk
from src import config
from src.keywords import extract_keywords

# Import Weaviate and associated modules
import weaviate
import weaviate.classes as wvc
from weaviate.classes.config import Property, DataType, Configure
from weaviate.util import generate_uuid5
from weaviate.classes.query import Filter


# Connect to Local Weaviate Instance, which automatically binds to 
# http://localhost:8080 and gRPC port 50051
client = weaviate.connect_to_local(
    port=config.WEAVIATE_HTTP_PORT, grpc_port=config.WEAVIATE_GRPC_PORT
)


def get_or_create_collection(collection_name: str):
    """
    Return a handle to the vector store collection, creating it if it
    doesn't exist.
    """

    # Return the collection if already exists, else create new
    if client.collections.exists(collection_name):
        collection = client.collections.get(collection_name)
    else:
        collection = client.collections.create(
            name=collection_name,

            # Setting Vectorizer to none because we don't want Weaviate to 
            # embed the chunks, rather we are going to provide the vectors 
            # along with the chunks
            vectorizer_config=Configure.Vectorizer.none(),

            # vector_index_config determines how the vector db will find the 
            # similarity between vectors. We will be using the cosine distance
            # under the Hierarchical Navigable Small World (HNSW) algorithm
            vector_index_config=Configure.VectorIndex.hnsw(
                distance_metric=wvc.config.VectorDistances.COSINE
            ),

            # Define metadata properties to store alongside text
            properties=[
                Property(name="file_path",data_type=DataType.TEXT),
                Property(name="document_id",data_type=DataType.TEXT),
                Property(name="chunk_id",data_type=DataType.TEXT),
                Property(name="chunk_index",data_type=DataType.INT),
                Property(name="token_count",data_type=DataType.INT),
                Property(name="heading_path",data_type=DataType.TEXT),
                Property(name="content",data_type=DataType.TEXT),
                Property(name="fingerprint",data_type=DataType.TEXT),
                # Known names (drugs, states) found in the chunk text.
                # FIELD tokenization = match the whole name exactly.
                Property(name="keywords", data_type=DataType.TEXT_ARRAY,
                         tokenization=wvc.config.Tokenization.FIELD),
            ]
        )

    # Collections created before the "keywords" property existed get it
    # added here (existing objects just have it empty until re-indexed).
    existing = {p.name for p in collection.config.get().properties}
    if "keywords" not in existing:
        collection.config.add_property(
            Property(name="keywords", data_type=DataType.TEXT_ARRAY,
                     tokenization=wvc.config.Tokenization.FIELD)
        )

    return collection


def get_indexed_doc_ids(collection, fingerprint: str) -> set[str]:
    """
    Return the document_ids already present in the collection under the
    given fingerprint. One paginated scan up front using a simple property 
    filter (no vector search), instead of running one query per document.

    Note: Weaviate's cursor-based pagination (`after`) can't be combined
    with a `where` filter, so this fetches document_id + fingerprint for
    every object and matches the fingerprint in Python instead of pushing
    the filter down to Weaviate.
    
    For large-scale enterprise RAG application, we should maintain a 
    separate record of what's processed per (document_id, fingerprint) in 
    a normal relational table (Postgres, DynamoDB, etc.), with index on 
    those columns. That makes checking "is this indexed" becomes a fast, 
    and cheap processing, rather than checking in the vector DB.
    """

    doc_ids = set()
    after = None
    page_size = 200

    while True:
        response = collection.query.fetch_objects(
            return_properties=["document_id", "fingerprint"],
            limit=page_size,
            after=after,
        )
        if not response.objects:
            break

        for obj in response.objects:
            if obj.properties.get("fingerprint") == fingerprint:
                doc_ids.add(obj.properties["document_id"])

        if len(response.objects) < page_size:
            break
        after = response.objects[-1].uuid

    return doc_ids


def add_chunks(collection, chunks: list[Chunk], vectors: list[list[float]], fingerprint: str) -> None:
    """
    Write chunks + their embeddings + metadata into the collection.
    `fingerprint` identifies the chunking strategy + params + embedding
    model that produced these chunks, so get_indexed_doc_ids() can  
    identify if a chunk was "already indexed under this exact config".
    """

    if not chunks:
        return

    if len(chunks) != len(vectors):
        raise ValueError("chunks and vectors must be the same length")

    with collection.batch.dynamic() as batch:
        for chunk, vector in zip(chunks, vectors):

            # Generate a deterministic UUIDv5 from the combination of
            # fingerprint and chunk id. This will produce unique UUID 
            # if a same chunk id is generated through a different strategy
            chunk_uuid = generate_uuid5(f"{fingerprint}:{chunk.chunk_id}")
                                        
            batch.add_object(
                # Explicitly set the UUID of the object in the collection to prevent duplicates
                uuid=chunk_uuid,
                properties={
                    "file_path": chunk.metadata.get("source_path"),
                    "document_id": chunk.doc_id,
                    "chunk_id": chunk.chunk_id,
                    "chunk_index": chunk.metadata.get("chunk_index"),
                    "token_count": chunk.metadata.get("token_count"),
                    "heading_path": chunk.metadata.get("heading_path"),
                    "content": chunk.text,
                    "fingerprint": fingerprint,
                    "keywords": extract_keywords(chunk.text),
                },
                vector=vector,
            )

    if collection.batch.failed_objects:
        for failed in collection.batch.failed_objects:
            print(failed.message)

def query_collection(
    collection,
    query_embedding: list[float],
    top_k: int,
    fingerprint: str,
    query_text: str | None = None,
    mode: str = "vector",
    alpha: float = 0.75,
    keyword_filter: bool = False,
) -> list[dict]:
    """
    Return the top_k best-matching chunks for the question, with their
    text, metadata, and score, restricted to chunks written under the
    given fingerprint.

    mode="vector": nearest vectors only (original behaviour).
    mode="hybrid": blend vector search with keyword (BM25) search on the
                   question text. alpha=1.0 is pure vector, 0.0 pure keyword.
    keyword_filter: if the question names a known drug/state, only
                   search chunks tagged with that name.
    """

    # Build the query to filter the collection in Weaviate
    filter_collection = (
        wvc.query.Filter.by_property("fingerprint").equal(fingerprint) 
        # we can add additional filters as needed
        # & wvc.query.Filter.by_property("is_archived").equal(False)
    )

    # This will cause the vector search on only the chunks which were 
    # built by the same chunking strategy defined in the config file, 
    # and not on the entire collection which will have chunks from the 
    # same documents using other chunking strategies
    matched_keywords = []
    if keyword_filter and query_text:
        matched_keywords = extract_keywords(query_text)
        if matched_keywords:
            filter_collection = filter_collection & (
                wvc.query.Filter.by_property("keywords").contains_any(matched_keywords)
            )

    if mode == "hybrid":
        if not query_text:
            raise ValueError("hybrid mode needs the question text (query_text)")
        response = collection.query.hybrid(
            query=query_text,
            vector=query_embedding,
            alpha=alpha,
            fusion_type=wvc.query.HybridFusion.RELATIVE_SCORE,
            query_properties=["content"],
            filters=filter_collection,
            limit=top_k,
            return_metadata=wvc.query.MetadataQuery(score=True),
        )
    else:
        response = collection.query.near_vector(
            near_vector=query_embedding,
            filters=filter_collection,
            limit=top_k,
            return_metadata=wvc.query.MetadataQuery(distance=True),
        )

    results = []
    for obj in response.objects:
        properties = obj.properties
        results.append({
            "text": properties.get("content"),
            "metadata": {
                "file_path": properties.get("file_path"),
                "document_id": properties.get("document_id"),
                "chunk_id": properties.get("chunk_id"),
                "chunk_index": properties.get("chunk_index"),
                "token_count": properties.get("token_count"),
                "heading_path": properties.get("heading_path"),
            },
            # vector: cosine similarity. hybrid: Weaviate's fused score (0-1).
            "score": (obj.metadata.score if mode == "hybrid" else 1 - obj.metadata.distance),
            "matched_keywords": matched_keywords,
        })

    return results


def backfill_keywords(collection) -> int:
    """
    Tag chunks that were indexed before keyword tagging existed, without
    re-embedding them (no OpenAI cost). Safe to re-run.
    Returns the number of chunks updated.
    """
    updated = 0
    for obj in collection.iterator(return_properties=["content", "keywords"]):
        tags = extract_keywords(obj.properties.get("content") or "")
        if sorted(obj.properties.get("keywords") or []) != tags:
            collection.data.update(uuid=obj.uuid, properties={"keywords": tags})
            updated += 1
    return updated


def close_client() -> None:
    # Close the Weaviate client instance at the end of the process
    if client.is_ready():
        client.close()