from pathlib import Path
from typing import List
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document

import config


def load_file(file_path: str) -> List[Document]:
    """Load a single file into a list of Documents based on extension."""
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext == ".pdf":
        loader = PyPDFLoader(str(path))
    elif ext in (".md", ".markdown"):
        loader = UnstructuredMarkdownLoader(str(path))
    elif ext in (".txt", ".text"):
        loader = TextLoader(str(path))
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    docs = loader.load()
    # Add filename to metadata so we can cite it later
    for d in docs:
        d.metadata["source_file"] = path.name
    return docs


def chunk_documents(docs: List[Document]) -> List[Document]:
    """Split documents into chunks with overlap."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)


def get_vector_store(collection_name: str = config.DEFAULT_COLLECTION) -> Chroma:
    """Get or create the ChromaDB vector store for a collection."""
    embeddings = OpenAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        openai_api_key=config.OPENAI_API_KEY,
    )
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=config.CHROMA_PERSIST_DIR,
    )


def ingest_file(file_path: str, collection_name: str = config.DEFAULT_COLLECTION) -> int:
    """Load, chunk, embed, and store a file. Returns chunk count."""
    docs = load_file(file_path)
    chunks = chunk_documents(docs)
    store = get_vector_store(collection_name)
    store.add_documents(chunks)
    return len(chunks)