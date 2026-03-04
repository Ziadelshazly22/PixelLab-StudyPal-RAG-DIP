"""
app/ui/interface.py
--------------------
Gradio chat interface for the Smart Learning Assistant.

Usage (standalone):
    python -m app.ui.interface

The interface is also mounted into the FastAPI app via Gradio's
``mount_gradio_app`` helper when running main.py.
"""

from __future__ import annotations

import gradio as gr


def build_interface(rag_chain=None) -> gr.Blocks:
    """
    Build and return a Gradio Blocks chat interface.

    Parameters
    ----------
    rag_chain : Runnable, optional
        A LangChain runnable chain.  When ``None`` a stub response is used.

    Returns
    -------
    gr.Blocks
    """

    def _respond(message: str, history: list[list[str]]) -> str:
        if rag_chain is None:
            return (
                "⚠️  RAG chain not initialised. "
                "Run the ingestion pipeline first, then restart the server."
            )
        try:
            return rag_chain.invoke(message)
        except Exception as exc:  # noqa: BLE001
            return f"❌ Error: {exc}"

    with gr.Blocks(
        title="Smart Learning Assistant – DIP Tutor",
    ) as demo:
        gr.Markdown(
            """
            # 📚 Smart Learning Assistant
            **RAG-Powered AI Tutor for Digital Image Processing**
            
            Ask any question about concepts from the *Gonzalez & Woods* textbook
            or OpenCV / scikit-image documentation.
            """
        )
        chatbot = gr.ChatInterface(
            fn=_respond,
            chatbot=gr.Chatbot(height=500),
            textbox=gr.Textbox(
                placeholder="e.g. Explain the Sobel edge detection operator...",
                container=False,
                scale=7,
            ),
            examples=[
                "What is the difference between spatial and frequency domain filtering?",
                "Explain the Canny edge detection algorithm step by step.",
                "How does histogram equalisation improve image contrast?",
                "What is the DFT and how is it used in image processing?",
            ],
        )  # noqa: F841

    return demo


if __name__ == "__main__":
    demo = build_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, theme=gr.themes.Soft())
