"""Gradio UI for the policy Q&A RAG agent.

Two-tab layout:
- **Ingest**: point at a corpus directory, click Ingest -> shows
  IngestionStats.
- **Ask**: question textbox + top-K slider, click Ask -> shows the
  answer with inline `[chunk_id]` markers, citation cards (each
  showing quoted_text + source path + section), and a confidence
  indicator.

Under `LLM_PROVIDER=mock` both tabs work without any real dep beyond
Gradio + Pydantic (fastembed / sqlite-vec / anthropic all bypassed).
Under a real provider the Ingest tab still doesn't need Anthropic
(only fastembed for embedding), so a user can ingest without ever
providing an API key and only needs it at Ask time.
"""

from __future__ import annotations

import time
from pathlib import Path

import gradio as gr

try:
    from .agent import (
        DEFAULT_INDEX_PATH,
        DEFAULT_TOP_K,
        PolicyQAError,
        ask,
        ingest_corpus,
        resolve_provider,
    )
except ImportError:
    from agent import (
        DEFAULT_INDEX_PATH,
        DEFAULT_TOP_K,
        PolicyQAError,
        ask,
        ingest_corpus,
        resolve_provider,
    )

_EXAMPLES_HANDBOOK = Path(__file__).parent / "examples" / "handbook"


def _run_ingest(corpus_dir: str, index_path: str) -> str:
    """Wrapper the Gradio Interface calls. Returns a Markdown block."""
    if not corpus_dir or not corpus_dir.strip():
        return "**Missing input.** Enter a corpus directory path."
    try:
        stats = ingest_corpus(corpus_dir, index_path=index_path or DEFAULT_INDEX_PATH)
    except PolicyQAError as exc:
        return f"**Ingestion failed:** {exc.message}"
    return (
        f"**Ingested successfully.**\n\n"
        f"- Files processed: {stats.total_files}\n"
        f"- Chunks created: {stats.total_chunks}\n"
        f"- Backend: `{stats.vector_store_backend}`\n"
        f"- Index size: {stats.index_size_bytes / 1024:.1f} KB\n"
        f"- Embedding model: `{stats.embedding_model}`\n"
        f"- Index path: `{stats.index_path}`"
    )


def _run_ask(question: str, index_path: str, top_k: int) -> tuple[str, str, str]:
    """Returns (answer_md, citations_md, status_md)."""
    if not question or not question.strip():
        return "", "", "**Missing input.** Type a question."
    start = time.perf_counter()
    try:
        answer = ask(
            question,
            index_path=index_path or DEFAULT_INDEX_PATH,
            top_k=int(top_k),
        )
    except PolicyQAError as exc:
        return "", "", f"**Failed:** {exc.message}"
    elapsed = time.perf_counter() - start

    answer_md = (
        f"### Answer\n\n{answer.answer}\n\n"
        f"*Confidence: `{answer.confidence}`*"
    )
    if answer.unanswered_reason:
        answer_md += f"\n\n**Not answered because:** {answer.unanswered_reason}"

    if answer.citations:
        citation_blocks = []
        for c in answer.citations:
            chunk = next(
                (ch for ch in answer.retrieved_chunks if ch.chunk_id == c.chunk_id),
                None,
            )
            section = f" ({chunk.section})" if chunk and chunk.section else ""
            source = chunk.source_path if chunk else "(unknown)"
            citation_blocks.append(
                f"**`{c.chunk_id}`** -- `{source}`{section}\n\n"
                f"> {c.quoted_text}"
            )
        citations_md = "### Citations\n\n" + "\n\n---\n\n".join(citation_blocks)
    else:
        citations_md = "### Citations\n\n*(none)*"

    status_md = (
        f"Answered in {elapsed:.1f}s using {len(answer.retrieved_chunks)} "
        f"retrieved chunk(s)."
    )
    return answer_md, citations_md, status_md


def build_ui() -> gr.Blocks:
    """Build and return the Gradio Blocks app."""
    provider = resolve_provider()
    default_corpus = str(_EXAMPLES_HANDBOOK) if _EXAMPLES_HANDBOOK.exists() else ""

    with gr.Blocks(title="Policy Q&A", theme=gr.themes.Soft()) as app:
        gr.Markdown(
            "# Policy Q&A over a handbook (agent #10)\n"
            "RAG with cross-field-verified citations, on-prem/private. "
            "Embed locally via fastembed, store in sqlite-vec (FAISS fallback), "
            "answer via Anthropic. The handbook contents never leave the box "
            "at index time.\n\n"
            f"**Current provider:** `{provider}`. Under `mock` both tabs return "
            "canned results instantly; under `anthropic` the Ask tab makes a "
            "real API call (~$0.01 per question)."
        )

        with gr.Tab("Ingest"):
            gr.Markdown(
                "Point at a directory of `.md` / `.txt` / `.pdf` files. "
                "The shipped example handbook works out of the box."
            )
            corpus_input = gr.Textbox(
                label="Corpus directory",
                value=default_corpus,
                placeholder="/path/to/handbook",
            )
            index_input = gr.Textbox(
                label="Index file path (created if missing)",
                value=DEFAULT_INDEX_PATH,
            )
            ingest_btn = gr.Button("Ingest", variant="primary")
            ingest_output = gr.Markdown()
            ingest_btn.click(
                fn=_run_ingest,
                inputs=[corpus_input, index_input],
                outputs=ingest_output,
            )

        with gr.Tab("Ask"):
            gr.Markdown(
                "Ask a question about the ingested corpus. Every factual claim "
                "in the answer will be backed by a verbatim citation from a "
                "retrieved chunk."
            )
            question_input = gr.Textbox(
                label="Question",
                placeholder="e.g. How many days of PTO do I get?",
            )
            with gr.Row():
                index_input_ask = gr.Textbox(
                    label="Index file path",
                    value=DEFAULT_INDEX_PATH,
                )
                top_k_slider = gr.Slider(
                    minimum=1, maximum=15, value=DEFAULT_TOP_K, step=1,
                    label="Top-K chunks to retrieve",
                )
            ask_btn = gr.Button("Ask", variant="primary")
            status_output = gr.Markdown()
            with gr.Row():
                with gr.Column():
                    answer_output = gr.Markdown()
                with gr.Column():
                    citations_output = gr.Markdown()
            ask_btn.click(
                fn=_run_ask,
                inputs=[question_input, index_input_ask, top_k_slider],
                outputs=[answer_output, citations_output, status_output],
            )

    return app


if __name__ == "__main__":
    build_ui().launch()
