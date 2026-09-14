"""
One-off debug utility: run chunk -> embed -> store for a single document,
so you can eyeball the heading_path breadcrumbs chunk-by-chunk and confirm
what actually lands in Weaviate for the configured strategy.

Kept out of src/ for the same reason as run_test_questions.py -- this is a
manual inspection tool, not part of the pipeline itself.
"""

from src import config, chunker, embedder, vectorstore
from src.ingest import load_corpus

TARGET_FILENAME = "rural-health-transformation-50-state-spotlights.pdf"

CHUNKERS = {
    "naive": chunker.naive_fixed_size,
    "recursive": chunker.recursive_overlap_chunk,
    "structure_aware": chunker.structure_aware_chunk,
}


def load_target_document():
    documents = load_corpus(config.DATA_RAW_DIR)
    for doc in documents:
        if doc.source_path.endswith(TARGET_FILENAME):
            return doc
    raise FileNotFoundError(f"{TARGET_FILENAME} not found under {config.DATA_RAW_DIR}")


if __name__ == "__main__":
    strategy = config.CHUNKING_STRATEGY
    fingerprint = (
        f"{strategy}_{config.CHUNK_SIZE}_{config.CHUNK_OVERLAP}"
        f"_{config.EMBEDDING_MODEL_NAME}"
    )

    doc = load_target_document()
    print(f"Loaded '{TARGET_FILENAME}' as doc_id={doc.doc_id}\n")

    chunk_fn = CHUNKERS[strategy]
    chunks = chunk_fn(doc, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    print(f"Split into {len(chunks)} chunks using '{strategy}'\n")

    for c in chunks:
        print(f"[{c.chunk_id}] heading_path={c.metadata.get('heading_path')!r}")
        print("  " + c.text[:160].replace("\n", " ") + "...")
        print("---")

    try:
        vectors = embedder.embed_chunks(chunks, config.EMBEDDING_MODEL_NAME)
        print(f"\nEmbedded {len(vectors)} chunks")

        collection = vectorstore.get_or_create_collection(config.VECTOR_STORE_COLLECTION_NAME)
        vectorstore.add_chunks(collection, chunks, vectors, fingerprint=fingerprint)
        print(f"Wrote {len(chunks)} chunks to '{config.VECTOR_STORE_COLLECTION_NAME}' "
              f"under fingerprint '{fingerprint}'")

        # Round-trip a few objects straight from Weaviate to confirm what's
        # actually stored, not just what we sent.
        sample = collection.query.fetch_objects(
            limit=5,
            return_properties=["chunk_id", "heading_path", "content"],
            filters=vectorstore.Filter.by_property("fingerprint").equal(fingerprint)
            & vectorstore.Filter.by_property("document_id").equal(doc.doc_id),
        )
        print("\nSample objects read back from Weaviate:")
        for obj in sample.objects:
            props = obj.properties
            print(f"  {props['chunk_id']} -> heading_path={props['heading_path']!r}")

    finally:
        vectorstore.close_client()
