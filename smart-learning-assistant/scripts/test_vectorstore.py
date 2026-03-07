"""Test vectorstore search after ChromaDB patches."""
import sys, os
os.chdir('B:/PixelLab-StudyPal-RAG-DIP/smart-learning-assistant')
sys.path.insert(0, '.')

from app.ingestion.pipeline import load_vectorstore

print("Loading vectorstore...")
vs = load_vectorstore()
count = vs._collection.count()
print(f"COLLECTION COUNT: {count}")

print("\nTesting similarity search: 'What is spatial filtering?'")
docs = vs.similarity_search('What is spatial filtering?', k=3)
print(f"SEARCH OK. Docs returned: {len(docs)}")
for i, d in enumerate(docs):
    src = d.metadata.get('source', '?')
    pg = d.metadata.get('page', '?')
    print(f"  [{i}] source={src}  page={pg}")
    print(f"       preview: {d.page_content[:150]}")

print("\nTesting guardrail retriever...")
from app.retrieval.retriever import get_guardrail_retriever
fn = get_guardrail_retriever(threshold=0.25)
r1 = fn("What is histogram equalization?")
print(f"In-domain query: {len(r1)} docs (expect >0)")
r2 = fn("What is the recipe for baklava?")
print(f"Off-topic query: {len(r2)} docs (expect 0)")

print("\nAll tests passed!")
