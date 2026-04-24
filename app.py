
from __future__ import annotations

import os
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gradio as gr
from dotenv import load_dotenv
from groq import Groq
from pypdf import PdfReader

load_dotenv()

DEFAULT_MODEL = os.environ.get("GROQ_MODEL", "llama-3.1-70b-versatile")
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
TOP_K = 6

SYSTEM_INSTRUCTION = (
    "You are a helpful assistant. Answer using the CONTEXT when it is relevant. "
    "If the context does not contain the answer, say so briefly and answer from general knowledge only "
    "when that is still useful. Keep answers concise."
)


@dataclass
class Chunk:
    text: str
    source: str


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


def _split_paragraphs(text: str) -> list[str]:
    parts = re.split(r"\n\s*\n", text)
    return [p.strip() for p in parts if p.strip()]


def chunk_text(text: str, source_label: str) -> list[Chunk]:
    """Split document text into overlapping character windows."""
    paragraphs = _split_paragraphs(text) or [text.strip()]
    chunks: list[Chunk] = []
    buf = ""
    for p in paragraphs:
        if len(buf) + len(p) + 2 <= CHUNK_SIZE:
            buf = f"{buf}\n\n{p}" if buf else p
            continue
        if buf:
            chunks.extend(_window_chunk(buf, source_label))
        buf = p
    if buf:
        chunks.extend(_window_chunk(buf, source_label))
    return chunks


def _window_chunk(text: str, source_label: str) -> list[Chunk]:
    out: list[Chunk] = []
    start = 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        piece = text[start:end].strip()
        if piece:
            out.append(Chunk(text=piece, source=source_label))
        if end >= len(text):
            break
        start = end - CHUNK_OVERLAP
    return out


def extract_pdf_text(path: str | Path) -> str:
    reader = PdfReader(str(path))
    pages: list[str] = []
    for page in reader.pages:
        t = page.extract_text() or ""
        pages.append(t)
    return "\n\n".join(pages).strip()


def score_chunk(query_tokens: set[str], chunk: Chunk) -> float:
    doc_tokens = _tokenize(chunk.text)
    if not doc_tokens or not query_tokens:
        return 0.0
    overlap = len(query_tokens & doc_tokens)
    return overlap / (len(query_tokens) ** 0.5 + 1e-6)


def retrieve(query: str, store: list[Chunk], k: int = TOP_K) -> list[Chunk]:
    q_tokens = _tokenize(query)
    ranked = sorted(store, key=lambda c: score_chunk(q_tokens, c), reverse=True)
    return ranked[:k]


def _groq_api_key() -> str | None:
    return os.environ.get("GROQ_API_KEY")


class RagChatbot:
    def __init__(self) -> None:
        api_key = _groq_api_key()
        if not api_key:
            raise RuntimeError(
                "Missing GROQ_API_KEY. Set it in .env — see .env.example. "
                "Create a key at https://console.groq.com/keys"
            )
        self._client = Groq(api_key=api_key)
        self.model_name = os.environ.get("GROQ_MODEL", DEFAULT_MODEL)
        self.temperature = float(os.environ.get("GROQ_TEMPERATURE", "0.7"))
        self.top_p = float(os.environ.get("GROQ_TOP_P", "0.95"))
        self.max_output_tokens = int(os.environ.get("GROQ_MAX_OUTPUT_TOKENS", "2048"))
        self.store: list[Chunk] = []

    def ingest_pdf(self, file_path: str | None) -> str:
        if not file_path:
            return f"No file selected.\n{self.kb_status_line()}"
        path = Path(str(file_path))
        if not path.exists():
            return f"File not found: {path}\n{self.kb_status_line()}"
        try:
            raw = extract_pdf_text(path)
        except Exception as exc:  # noqa: BLE001 — surface errors to UI
            return f"Could not read PDF: {exc}\n{self.kb_status_line()}"
        if not raw:
            return (
                "No extractable text in that PDF (it may be scanned images only).\n"
                f"{self.kb_status_line()}"
            )
        label = path.name
        new_chunks = chunk_text(raw, label)
        self.store.extend(new_chunks)
        return (
            f"Added {len(new_chunks)} chunks from \"{label}\". "
            f"Total chunks: {len(self.store)}.\n{self.kb_status_line()}"
        )

    def chat(
        self, message: str, history: list[dict[str, Any]] | None
    ) -> Iterator[tuple[str, list[dict[str, Any]]]]:
        history = list(history or [])
        message = (message or "").strip()
        if not message:
            yield "", history
            return

        context_blocks: list[str] = []
        if self.store:
            hits = retrieve(message, self.store)
            for i, ch in enumerate(hits, start=1):
                context_blocks.append(f"[{i} | source: {ch.source}]\n{ch.text}")
        context = "\n\n---\n\n".join(context_blocks) if context_blocks else "(no documents ingested yet)"
        user_prompt = f"CONTEXT:\n{context}\n\nQUESTION:\n{message}"

        user_msg: dict[str, Any] = {"role": "user", "content": message}

        history_messages = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in history
        ]
        history_messages.append({"role": "user", "content": user_prompt})

        acc = ""
        try:
            stream = self._client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "system", "content": SYSTEM_INSTRUCTION}] + history_messages,
                max_tokens=self.max_output_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                stream=True,
            )
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    acc += chunk.choices[0].delta.content
                    yield "", history + [user_msg, {"role": "assistant", "content": acc}]
            if not acc:
                yield "", history + [
                    user_msg,
                    {"role": "assistant", "content": "(No text returned; the response may have been blocked.)"},
                ]
        except Exception as exc:  # noqa: BLE001
            reply = f"Error calling Groq API: {exc}"
            yield "", history + [user_msg, {"role": "assistant", "content": reply}]

    def clear_kb(self) -> str:
        n = len(self.store)
        self.store.clear()
        return f"Cleared knowledge base ({n} chunks removed). {self.kb_status_line()}"

    def kb_status_line(self) -> str:
        return (
            f"Indexed PDF chunks: {len(self.store)} · Model: {self.model_name} · "
            f"max_output_tokens={self.max_output_tokens}"
        )


def build_config_missing_ui() -> tuple[gr.Blocks, gr.Theme | None, str | None]:
    theme = gr.themes.Soft(primary_hue="blue")
    with gr.Blocks(title="PDF RAG · Groq", fill_width=True) as demo:
        gr.Markdown(
            "## PDF RAG (Groq)\n"
            "The app is installed, but **no API key** was found.\n\n"
            "1. Copy `.env.example` to `.env` in this folder.\n"
            "2. Create a free key at [Groq Console](https://console.groq.com/keys).\n"
            "3. Set `GROQ_API_KEY` in `.env`.\n"
            "4. Run `python app.py` again.\n"
        )
    return demo, theme, None


def build_ui(bot: RagChatbot) -> tuple[gr.Blocks, gr.Theme | None, str | None]:
    theme = gr.themes.Soft(primary_hue="blue")
    custom_css = """
    .rag-header { margin-bottom: 0.5rem; }
    .rag-sidebar { border-right: 1px solid var(--border-color-primary); padding-right: 1rem; }
    """

    with gr.Blocks(title="PDF RAG · Groq", fill_width=True) as demo:
        gr.Markdown(
            "## PDF RAG · Groq\n"
            "Add PDFs on the left; answers use retrieved excerpts plus **Groq** (streaming). "
            "Scanned PDFs without a text layer will not index well.",
            elem_classes=["rag-header"],
        )

        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=280, elem_classes=["rag-sidebar"]):
                gr.Markdown("### Knowledge base")
                pdf = gr.File(
                    label="PDF",
                    file_types=[".pdf"],
                    type="filepath",
                )
                ingest_btn = gr.Button("Index this PDF", variant="primary", size="lg")
                clear_kb_btn = gr.Button("Clear indexed PDFs", variant="secondary")
                status = gr.Textbox(
                    label="Status",
                    value=bot.kb_status_line(),
                    interactive=False,
                    lines=4,
                    max_lines=8,
                )
                ingest_btn.click(bot.ingest_pdf, inputs=pdf, outputs=status)
                clear_kb_btn.click(bot.clear_kb, outputs=status)

            with gr.Column(scale=2):
                gr.Markdown("### Chat")
                chatbot = gr.Chatbot(
                    height=480,
                    label="Conversation",
                    layout="bubble",
                    placeholder="Upload a PDF and index it, then ask questions here. "
                    "Example: “Summarize the main conclusions.”",
                )
                msg = gr.Textbox(
                    label="Message",
                    placeholder="Type your question and press Enter or Send…",
                    lines=2,
                    max_lines=8,
                    show_label=False,
                )
                with gr.Row():
                    send = gr.Button("Send", variant="primary", scale=1)
                    clear_chat_btn = gr.Button("Clear conversation", variant="secondary", scale=1)

                send.click(bot.chat, inputs=[msg, chatbot], outputs=[msg, chatbot])
                msg.submit(bot.chat, inputs=[msg, chatbot], outputs=[msg, chatbot])
                clear_chat_btn.click(lambda: ([], ""), outputs=[chatbot, msg])

        demo.queue(default_concurrency_limit=4)

    return demo, theme, custom_css


def main() -> None:
    if not _groq_api_key():
        demo, theme, _css = build_config_missing_ui()
        demo.launch(inbrowser=True, theme=theme)
        return
    bot = RagChatbot()
    demo, theme, css = build_ui(bot)
    demo.launch(inbrowser=True, theme=theme, css=css or None)


if __name__ == "__main__":
    main()