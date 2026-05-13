from typing import List, Dict
from langchain.schema import Document
from anthropic import Anthropic

import config

client = Anthropic(api_key=config.ANTHROPIC_API_KEY)


SYSTEM_PROMPT = """You are a knowledge agent that answers questions using ONLY the provided context from company documents.

Rules:
1. Answer only from the provided context. If the context does not contain the answer, say "The provided documents don't cover this. You may need to ask [the relevant team] directly." Do not guess or use external knowledge.
2. Cite specific source chunks for every factual claim using [Source N] format inline, where N matches the source number shown in the context.
3. If the context is partial or ambiguous, say what you know and explicitly flag what you don't know.
4. Use direct language. Skip phrases like "Based on the provided context..." or "According to the documents...". Just answer.
5. If multiple sources support the same claim, cite all of them: [Source 1, Source 3].
6. Format with short paragraphs, bullets, or numbered steps when it improves clarity. Don't over-structure short answers.

Quality bar:
- Specific over vague
- Concise over verbose
- Cited over confident
- Honest about uncertainty"""


def format_context(chunks: List[Document]) -> str:
    """Format retrieved chunks as numbered sources for the LLM."""
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        source = chunk.metadata.get("source_file", "unknown")
        page = chunk.metadata.get("page")
        page_str = f" (page {page})" if page is not None else ""
        parts.append(
            f"[Source {i}] from {source}{page_str}:\n{chunk.page_content}"
        )
    return "\n\n---\n\n".join(parts)


def generate_answer(
    query: str,
    chunks: List[Document],
    model: str = config.LLM_MODEL,
) -> Dict:
    """Generate an answer with citations from retrieved chunks."""
    context = format_context(chunks)

    user_message = (
        "Context from company documents:\n\n"
        f"{context}\n\n"
        "---\n\n"
        f"Question: {query}\n\n"
        "Answer the question using only the context above. Cite sources inline with [Source N]."
    )

    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    answer = response.content[0].text

    citations = [
        {
            "source": chunk.metadata.get("source_file", "unknown"),
            "page": chunk.metadata.get("page"),
            "content": (
                chunk.page_content[:300] + "..."
                if len(chunk.page_content) > 300
                else chunk.page_content
            ),
        }
        for chunk in chunks
    ]

    return {"answer": answer, "citations": citations, "model": model}