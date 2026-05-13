import tempfile
from pathlib import Path

import streamlit as st

import config
from ingest import ingest_file
from retrieve import retrieve_chunks
from generate import generate_answer


st.set_page_config(page_title="RAG Knowledge Agent", layout="wide")
st.title("Internal Knowledge Agent")
st.caption("Upload company documents. Ask questions. Get cited answers.")


# Sidebar — collection + upload
with st.sidebar:
    st.header("Document Library")
    collection_name = st.text_input(
        "Collection name", value=config.DEFAULT_COLLECTION
    )

    uploaded_files = st.file_uploader(
        "Upload documents",
        type=["pdf", "md", "txt"],
        accept_multiple_files=True,
    )

    if uploaded_files and st.button("Ingest"):
        for f in uploaded_files:
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=Path(f.name).suffix
            ) as tmp:
                tmp.write(f.getvalue())
                tmp_path = tmp.name

            with st.spinner(f"Ingesting {f.name}..."):
                count = ingest_file(tmp_path, collection_name)
                st.success(f"Added {count} chunks from {f.name}")
            Path(tmp_path).unlink(missing_ok=True)


# Main — chat
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("citations"):
            with st.expander(f"View {len(msg['citations'])} sources"):
                for i, c in enumerate(msg["citations"], start=1):
                    header = f"**[Source {i}]** {c['source']}"
                    if c.get("page") is not None:
                        header += f", page {c['page']}"
                    st.markdown(header)
                    st.text(c["content"])


if query := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            chunks = retrieve_chunks(query, collection_name)

        if not chunks:
            answer = "No relevant content found in this collection. Upload documents first."
            st.write(answer)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )
        else:
            with st.spinner("Generating answer..."):
                result = generate_answer(query, chunks)

            st.write(result["answer"])
            with st.expander(f"View {len(result['citations'])} sources"):
                for i, c in enumerate(result["citations"], start=1):
                    header = f"**[Source {i}]** {c['source']}"
                    if c.get("page") is not None:
                        header += f", page {c['page']}"
                    st.markdown(header)
                    st.text(c["content"])

            st.session_state.messages.append({
                "role": "assistant",
                "content": result["answer"],
                "citations": result["citations"],
            })