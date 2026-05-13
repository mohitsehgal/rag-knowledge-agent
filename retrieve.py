from typing import List
from langchain.schema import Document

import config
from ingest import get_vector_store


def retrieve_chunks(
    query: str,
    collection_name: str = config.DEFAULT_COLLECTION,
    top_k: int = None,
) -> List[Document]:
    """Retrieve top-k most relevant chunks for a query."""
    store = get_vector_store(collection_name)
    return store.similarity_search(query, k=top_k or config.TOP_K)