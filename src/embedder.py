"""
Embedding: turn Chunk text into vectors using whichever embedding
model defined in the config.py.
"""

from src.chunker import Chunk
from dotenv import load_dotenv
from openai import OpenAI


def embed_chunks(chunks: list[Chunk], model_name: str, batch_size=100) -> list[list[float]]:
    """
    Return one embedding vector corresponding to the input chunks
    by using the embedding model defined in the config.py file.
    """

    if not chunks:
        return []

    # Initialize OpenAI client with the OPENAI_API_KEY from env
    load_dotenv()
    client = OpenAI()
    embedded_vectors = []


    # ***** Embedding one chunk at a time *****
    # for chunk in chunks:
    #     # Prepare kwargs for the API call
    #     embed_kwargs = {
    #         "model": model_name,
    #         "input": chunk.text,
    #     }
    #     # Call OpenAI Embeddings API
    #     response = client.embeddings.create(**embed_kwargs)
    #     embedded_vector = response.data[0].embedding
    #     embedded_vectors.append(embedded_vector)


    # ***** Batch Embedding *****
    
    # Instead of calling the OpenAI API for embedding one chunk 
    # at a time, the batch embedding combines list of chunks 
    # together and call the OpenAI API once for every batch. 
    # This strategy helps - a. to avoid Requests Per Minute (RPM) 
    # rate limit from OpenAI and reduce latency.

    # Populate list with the raw text from all chunk objects
    chunk_list = [chunk.text for chunk in chunks]

    # Process the chunklist in batches
    for i in range(0, len(chunk_list), batch_size):
        batch = chunk_list[i : i + batch_size]

        # Pass the entire chunklist to the 'input' parameter
        response = client.embeddings.create(
            model=model_name,
            input=batch
        )

        # Extract embeddings for just this batch fresh on every loop
        batch_embeddings = [data.embedding for data in response.data]
        
        embedded_vectors.extend(batch_embeddings)

    return embedded_vectors
