"""Check L2 distance scores to calibrate guardrail threshold for MiniLM."""
import sys, os
os.chdir('B:/PixelLab-StudyPal-RAG-DIP/smart-learning-assistant')
sys.path.insert(0, '.')

from app.ingestion.pipeline import load_vectorstore

vs = load_vectorstore()

queries = [
    ("DIP in-domain",  "What is spatial filtering and how does it work?"),
    ("DIP in-domain",  "Explain histogram equalization algorithm"),
    ("DIP in-domain",  "What is the Sobel edge detection operator?"),
    ("Python util",    "How do I apply a gaussian blur in OpenCV?"),
    ("Off-topic",      "What is the capital of France?"),
    ("Off-topic",      "Give me a recipe for chocolate cake"),
    ("Off-topic",      "Who won the football match last night?"),
]

print(f"{'Type':<15}  {'Score':>8}  Query")
print("-" * 80)
for qtype, query in queries:
    results = vs.similarity_search_with_score(query, k=1)
    if results:
        _, score = results[0]
        print(f"{qtype:<15}  {score:>8.4f}  {query[:60]}")
    else:
        print(f"{qtype:<15}  {'N/A':>8}  {query[:60]}")
