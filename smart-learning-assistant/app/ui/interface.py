# -*- coding: utf-8 -*-
"""
app/ui/interface.py
--------------------
Gradio Blocks chat interface for the DIP AI Tutor.

Layout
------
Two tabs:
    1. 💬 Chat   — stateful RAG Q&A via POST /chat
  2. 📄 Upload — PDF ingestion via POST /ingest + status via GET /status

The UI always calls the FastAPI backend at http://localhost:8000.
The optional ``rag_chain`` parameter is kept for backward compatibility
with main.py's ``gr.mount_gradio_app`` call but is not used at runtime.

Standalone launch
-----------------
    python app/ui/interface.py
    # → http://localhost:7860

Mount into FastAPI (main.py)
-----------------------------
    # UI runs as a separate process: python app/ui/interface.py
    import gradio as gr
    from app.ui.interface import build_interface
    app = gr.mount_gradio_app(app, build_interface(), path="/ui")
"""

from __future__ import annotations

import logging
import re
import uuid

import gradio as gr
import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Backend configuration
# ---------------------------------------------------------------------------
_BACKEND = "http://localhost:8000"
_CHAT_URL = f"{_BACKEND}/chat"
_SUMMARIZE_URL = f"{_BACKEND}/summarize"
_INGEST_URL = f"{_BACKEND}/ingest"
_STATUS_URL = f"{_BACKEND}/status"
_CHAT_TIMEOUT = 60    # seconds — Gemini can be slow on first call
_INGEST_TIMEOUT = 60  # seconds — ingestion can be slow


# ---------------------------------------------------------------------------
# Helper: citation formatter
# ---------------------------------------------------------------------------

def _format_citations(text: str) -> str:
    """Bold-emphasise ``[Source: X, Page Y]`` markers in LLM output."""
    return re.sub(
        r"\[Source:\s*([^,\]]+),\s*Page[:\s]*(\d+)\]",
        r"**📖 [Source: \1, Page \2]**",
        text,
    )


# ---------------------------------------------------------------------------
# Helper: call conversational chat API  (stateful, per-session memory)
# ---------------------------------------------------------------------------

def _call_chat_api(question: str, session_id: str) -> str:
    """
    POST to ``/chat`` (``ConversationalRetrievalChain``) and return the answer.

    Appends formatted source citations when the backend returns them.
    Handles:
    - ``ConnectionError`` → backend offline message
    - ``Timeout``         → timeout message
    - HTTP errors         → status code + truncated body
    - Unexpected errors   → generic message with exc string
    """
    payload = {"question": question, "session_id": session_id}
    try:
        resp = requests.post(_CHAT_URL, json=payload, timeout=_CHAT_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        answer = data.get("answer", "")
        sources = data.get("sources", [])
        if sources:
            citations = "\n\n---\n**📚 Sources:**\n"
            for src_item in sources[:3]:
                src = src_item.get("source", "")
                page = src_item.get("page", "")
                if src:
                    citations += (
                        f"- 📖 `{src}`"
                        + (f", Page {page}" if page else "")
                        + "\n"
                    )
            answer = answer + citations
        return _format_citations(answer)

    except requests.exceptions.ConnectionError:
        return (
            "⚠️ Cannot reach the backend server. "
            "Please ensure FastAPI is running on port 8000."
        )
    except requests.exceptions.Timeout:
        return (
            "⚠️ Response timed out. The server may be processing a "
            "large context. Please try again."
        )
    except requests.exceptions.HTTPError as exc:
        body = exc.response.text[:300] if exc.response is not None else str(exc)
        return f"❌ Server error ({exc.response.status_code}): {body}"
    except Exception as exc:
        logger.error("Unexpected error calling /chat API: %s", exc, exc_info=True)
        return f"❌ Unexpected error: {exc}"


# ---------------------------------------------------------------------------
# Helper: call summarize API
# ---------------------------------------------------------------------------

def _call_summarize(
    source: str,
    n_questions: int,
    progress: gr.Progress = gr.Progress(track_tqdm=False),
) -> tuple:
    """
    POST to ``/summarize`` and return ``(summary_markdown, questions_data)``.

    Args:
        source:      Exact filename stored in chunk metadata.
        n_questions: Number of study questions to generate.

    Returns:
        Tuple of ``(summary_str, [[question], ...])`` suitable for
        ``gr.Markdown`` and ``gr.Dataframe`` outputs respectively.
    """
    if not source or not str(source).strip():
        return "⚠️ Please enter a document filename.", []

    payload = {
        "source": str(source).strip(),
        "include_questions": True,
        "n_questions": int(n_questions),
    }
    progress(0.05, desc="Preparing summarize request...")
    try:
        resp = requests.post(
            _SUMMARIZE_URL,
            json=payload,
            timeout=180,  # map-reduce can take 2–3 min for large PDFs
        )
        resp.raise_for_status()
        data = resp.json()
        summary_md = data.get("summary", "⚠️ No summary returned.")
        questions = data.get("study_questions", [])
        questions_data = [[q] for q in questions] if questions else []
        progress(0.75, desc="Received summarize response. Parsing output...")
        progress(1.0, desc="Done")
        return summary_md, questions_data

    except requests.exceptions.ConnectionError:
        return "⚠️ Cannot reach the backend server.", []
    except requests.exceptions.Timeout:
        return "⚠️ Summarisation timed out. The document may be very large.", []
    except requests.exceptions.HTTPError as exc:
        body = exc.response.text[:300] if exc.response is not None else str(exc)
        return f"❌ Server error ({exc.response.status_code}): {body}", []
    except Exception as exc:
        logger.error("Unexpected error calling /summarize API: %s", exc, exc_info=True)
        return f"❌ Unexpected error: {exc}", []


# ---------------------------------------------------------------------------
# Helper: fetch /status
# ---------------------------------------------------------------------------

def _fetch_status() -> str:
    """``GET /status`` → formatted markdown summary string."""
    try:
        resp = requests.get(_STATUS_URL, timeout=5)
        resp.raise_for_status()
        d = resp.json()
        ts = d.get("server_time", "")[:19].replace("T", " ")
        chunks = d.get("total_chunks", "?")
        chunks_fmt = f"{chunks:,}" if isinstance(chunks, int) else str(chunks)
        return (
            f"**Backend:** `{d.get('llm_backend', 'unknown').upper()}`  ·  "
            f"**Embedding:** `{d.get('embedding_model', 'unknown')}`  ·  "
            f"**Chunks:** {chunks_fmt}  ·  "
            f"**Collection:** `{d.get('collection', '?')}`  ·  "
            f"**As of:** {ts} UTC"
        )
    except requests.exceptions.ConnectionError:
        return "⚠️ Backend offline — start the FastAPI server to see live status."
    except Exception as exc:
        return f"⚠️ Could not fetch status: {exc}"


def _fetch_status_and_sources() -> tuple[str, dict]:
    """Return ``(status_markdown, dropdown_update)`` from ``GET /status``."""
    status_md = _fetch_status()
    try:
        resp = requests.get(_STATUS_URL, timeout=5)
        resp.raise_for_status()
        d = resp.json()
        sources = d.get("sources", [])
        source_choices = [s for s in sources if isinstance(s, str) and s.strip()]
        return status_md, gr.update(choices=sorted(set(source_choices)))
    except Exception:
        return status_md, gr.update(choices=[])


# ---------------------------------------------------------------------------
# Helper: upload PDFs
# ---------------------------------------------------------------------------

def _upload_files(files) -> str:
    """
    POST each uploaded PDF to ``/ingest`` as multipart form-data.
    Returns a multi-line status string suitable for a Textbox component.
    """
    if not files:
        return "⚠️ No files selected."

    lines: list[str] = []
    for file_obj in files:
        # Gradio 4.x passes either a NamedString or a file-like with .name
        path = getattr(file_obj, "name", str(file_obj))
        filename = re.split(r"[/\\]", path)[-1]

        try:
            with open(path, "rb") as fh:
                resp = requests.post(
                    _INGEST_URL,
                    files={"file": (filename, fh, "application/pdf")},
                    timeout=_INGEST_TIMEOUT,
                )
            resp.raise_for_status()
            data = resp.json()

            if data.get("status") == "processing":
                lines.append(
                    f"⏳ {filename} — Queued for background ingestion (file > 5 MB). "
                    "Poll 🔄 Refresh Status to check when done."
                )
            else:
                chunks = data.get("chunks_added", 0)
                pages = data.get("pages_processed", 0)
                lines.append(
                    f"✅ {filename} — {chunks:,} chunks added ({pages} pages processed)"
                )

        except requests.exceptions.ConnectionError:
            lines.append(f"❌ {filename} — Error: Cannot reach backend server.")
        except requests.exceptions.HTTPError as exc:
            err = exc.response.text[:200] if exc.response is not None else str(exc)
            lines.append(f"❌ {filename} — Error: {err}")
        except Exception as exc:
            lines.append(f"❌ {filename} — Error: {exc}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Chat event handlers
# ---------------------------------------------------------------------------

def _handle_send(
    user_message: str,
    chat_history: list,
    session_id: str,
) -> tuple[list, str]:
    """Append user message + RAG answer to chat history; clear the textbox."""
    if not user_message.strip():
        return chat_history, user_message

    answer = _call_chat_api(user_message.strip(), session_id)
    return chat_history + [[user_message, answer]], ""


def _handle_clear(session_id: str) -> tuple[list, str]:
    """Wipe chat history locally and clear server-side session memory."""
    try:
        requests.delete(f"{_BACKEND}/chat/{session_id}", timeout=5)
    except Exception:
        pass  # Best-effort — don't block the UI clear if backend is unreachable
    return [], ""


# ---------------------------------------------------------------------------
# Interface builder
# ---------------------------------------------------------------------------

def build_interface(rag_chain=None) -> gr.Blocks:
    """
    Construct and return the full Gradio Blocks interface.

    Parameters
    ----------
    rag_chain : optional
        Accepted but unused — kept for compatibility with
        ``gr.mount_gradio_app`` calls in main.py.

    Returns
    -------
    gr.Blocks
    """
    with gr.Blocks(
        title="DIP AI Tutor",
    ) as demo:

        # ── Session state (uuid4, generated once on page load) ──────────
        session_id = gr.State(value=lambda: str(uuid.uuid4()))

        # ── Top status bar (always visible) ─────────────────────────────
        with gr.Row():
            status_bar = gr.Markdown(
                value="⏳ Connecting to backend...",
                elem_id="status-bar",
            )

        # ── Main tabs ───────────────────────────────────────────────────
        with gr.Tabs():

            # ── TAB 1: CHAT ─────────────────────────────────────────────
            with gr.Tab("💬 Chat"):
                gr.Markdown(
                    "# 🎓 Digital Image Processing AI Tutor\n"
                    "Powered by **Gonzalez & Woods 4th Ed** · OpenCV · NumPy · SciPy"
                )

                chatbot = gr.Chatbot(
                    height=500,
                    show_label=False,
                )

                question_box = gr.Textbox(
                    placeholder=(
                        "Ask about spatial filtering, Fourier transforms, "
                        "morphological operations..."
                    ),
                    label="Your Question",
                    lines=2,
                )

                with gr.Row():
                    send_btn = gr.Button("Send", variant="primary")
                    clear_btn = gr.Button("🗑️ Clear Conversation")

                with gr.Accordion("💡 Example Questions", open=False):
                    _EXAMPLES = [
                        "What is histogram equalization and when is it used?",
                        "Derive the discrete Fourier Transform for 2D images.",
                        "Explain morphological erosion vs dilation with OpenCV code.",
                        "What noise models are common in DIP and how do we remove them?",
                        "How does the Canny edge detector work step by step?",
                    ]
                    for _ex in _EXAMPLES:
                        _btn = gr.Button(
                            _ex, size="sm", elem_classes=["example-btn"]
                        )
                        # Each button click populates the textbox
                        _btn.click(
                            fn=lambda e=_ex: e,
                            inputs=[],
                            outputs=[question_box],
                        )

                # ── Chat events ─────────────────────────────────────────
                send_btn.click(
                    fn=_handle_send,
                    inputs=[question_box, chatbot, session_id],
                    outputs=[chatbot, question_box],
                )
                question_box.submit(
                    fn=_handle_send,
                    inputs=[question_box, chatbot, session_id],
                    outputs=[chatbot, question_box],
                )
                clear_btn.click(
                    fn=_handle_clear,
                    inputs=[session_id],
                    outputs=[chatbot, question_box],
                )

            # ── TAB 2: UPLOAD ────────────────────────────────────────────
            with gr.Tab("📄 Upload Documents"):
                gr.Markdown(
                    "## 📥 Add Documents to Knowledge Base\n"
                    "Upload PDF files to expand the tutor's knowledge. "
                    "Supported: textbooks, papers, documentation."
                )

                file_upload = gr.File(
                    file_types=[".pdf"],
                    file_count="multiple",
                    label="Select PDF(s) to upload",
                )

                upload_btn = gr.Button("Add to Knowledge Base", variant="primary")

                ingestion_status = gr.Textbox(
                    label="Ingestion Status",
                    interactive=False,
                    lines=4,
                )

                refresh_btn = gr.Button("🔄 Refresh Status", size="sm")

                status_display = gr.Markdown(
                    value="",
                    elem_id="status_display",
                )

                # ── Upload / status events ──────────────────────────────
                upload_btn.click(
                    fn=_upload_files,
                    inputs=[file_upload],
                    outputs=[ingestion_status],
                )
                # ── Summarize Document section ──────────────────────────────────
                gr.Markdown("---")
                gr.Markdown(
                    "## 🔍 Summarize a Document\n"
                    "Generate an academic summary and exam-style study questions "
                    "for any ingested PDF."
                )

                summarize_filename = gr.Dropdown(
                    choices=[],
                    allow_custom_value=True,
                    label="Document filename (select from KB or type manually)",
                    info="Examples: Digital_Image_Processing_Gonzalez_Woods_4th_Ed.pdf",
                )

                n_questions_slider = gr.Slider(
                    minimum=2,
                    maximum=10,
                    value=5,
                    step=1,
                    label="Number of Study Questions",
                )

                summarize_btn = gr.Button(
                    "📝 Generate Summary & Study Questions",
                    variant="secondary",
                )

                gr.Markdown(
                    "_⏳ Generating summary... this may take 1\u20133 minutes. "
                    "Please wait and do not close the page._",
                    visible=False,
                    elem_id="summarize-info",
                )

                summary_output = gr.Markdown(
                    value="",
                    label="📄 Document Summary",
                )

                questions_output = gr.Dataframe(
                    headers=["Study Questions"],
                    label="🎓 Study Questions",
                    interactive=False,
                    wrap=True,
                )

                refresh_btn.click(
                    fn=_fetch_status_and_sources,
                    inputs=[],
                    outputs=[status_display, summarize_filename],
                )

                summarize_btn.click(
                    fn=lambda: (
                        "⏳ Generating summary\u2026 this may take 1\u20133 minutes. Please wait.",
                        [],
                    ),
                    inputs=[],
                    outputs=[summary_output, questions_output],
                    queue=False,
                ).then(
                    fn=_call_summarize,
                    inputs=[summarize_filename, n_questions_slider],
                    outputs=[summary_output, questions_output],
                )
        # ── On page load: populate top status bar ───────────────────────
        demo.load(
            fn=_fetch_status_and_sources,
            inputs=[],
            outputs=[status_bar, summarize_filename],
        )

    return demo


# ---------------------------------------------------------------------------
# Standalone launch
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    demo = build_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        theme=gr.themes.Soft(),
        css="#status-bar{font-size:.80em;opacity:.88;padding:4px 0} .example-btn{margin:2px 0!important}",
    )
