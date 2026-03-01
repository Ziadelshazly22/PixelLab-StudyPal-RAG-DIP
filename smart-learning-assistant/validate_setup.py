"""
validate_setup.py – Verifies that all critical libraries can be imported.
Run with: python validate_setup.py
"""

import importlib

LIBRARIES = [
    ("langchain", "langchain"),
    ("chromadb", "chromadb"),
    ("gradio", "gradio"),
    ("fastapi", "fastapi"),
    ("fitz", "fitz"),
    ("cv2", "cv2"),
    ("sentence_transformers", "sentence_transformers"),
    ("langserve", "langserve"),
    ("google.generativeai", "google.generativeai"),
    ("ragas", "ragas"),
]


def main() -> None:
    all_ok = True
    for display_name, module_name in LIBRARIES:
        try:
            importlib.import_module(module_name)
            print(f"✅ {display_name} OK")
        except ImportError as exc:
            print(f"❌ {display_name} FAILED – {exc}")
            all_ok = False

    if all_ok:
        print("\nAll critical libraries imported successfully.")
    else:
        print("\nSome libraries failed to import. Check the errors above.")


if __name__ == "__main__":
    main()
