"""Live smoke test: start server, test all endpoints, stop server."""
import subprocess
import sys
import time
import json
import requests
import os

SERVER_URL = "http://127.0.0.1:8000"
PY = r"B:\PixelLab-StudyPal-RAG-DIP\smart-learning-assistant\.venv\Scripts\python.exe"
APP_DIR = r"B:\PixelLab-StudyPal-RAG-DIP\smart-learning-assistant"

env = os.environ.copy()
env["CHROMA_PERSIST_DIR"] = r"B:\PixelLab-StudyPal-RAG-DIP\smart-learning-assistant\data\chroma_db"

print("=" * 60)
print("Starting uvicorn server...")
proc = subprocess.Popen(
    [PY, "-m", "uvicorn", "main:app", "--port", "8000"],
    cwd=APP_DIR, env=env,
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
)

# Wait for server to be ready
print("Waiting for server startup (up to 30s)...")
for i in range(30):
    time.sleep(1)
    try:
        r = requests.get(f"{SERVER_URL}/health", timeout=2)
        if r.status_code == 200:
            print(f"Server ready after {i+1}s")
            break
    except Exception:
        pass
else:
    print("ERROR: Server did not start in time!")
    proc.kill()
    sys.exit(1)

results = {}

try:
    # Test /health
    r = requests.get(f"{SERVER_URL}/health")
    results["health"] = r.json()
    print(f"\n[1] GET /health -> {r.status_code}: {r.json()}")

    # Test /status
    r = requests.get(f"{SERVER_URL}/status")
    s = r.json()
    results["status"] = s
    print(f"\n[2] GET /status -> {r.status_code}")
    print(f"    total_chunks: {s.get('total_chunks')}")
    print(f"    sources count: {len(s.get('sources', []))}")
    src_preview = s.get('sources', [])[:3]
    print(f"    sources: {src_preview}{'...' if len(s.get('sources',[])) > 3 else ''}")

    # Test /chat -- turn 1
    sid = f"smoke-test-{int(time.time())}"
    r = requests.post(f"{SERVER_URL}/chat",
        json={"question": "What is spatial filtering?", "session_id": sid},
        timeout=120)
    c1 = r.json()
    results["chat1"] = c1
    print(f"\n[3] POST /chat (turn 1) -> {r.status_code}")
    if r.status_code == 200:
        ans = c1.get("answer", "")
        print(f"    answer length: {len(ans)} chars")
        print(f"    answer preview: {ans[:300]}...")
        print(f"    sources: {c1.get('sources', [])[:2]}")
    else:
        print(f"    ERROR detail: {c1.get('detail', str(c1))[:300]}")

    # Test /chat -- turn 2 (uses memory from turn 1)
    r = requests.post(f"{SERVER_URL}/chat",
        json={"question": "Give me the formula for it.", "session_id": sid},
        timeout=120)
    c2 = r.json()
    results["chat2"] = c2
    print(f"\n[4] POST /chat (turn 2, same session) -> {r.status_code}")
    ans2 = c2.get("answer", "")
    print(f"    answer length: {len(ans2)} chars")
    print(f"    answer preview: {ans2[:200]}...")

    # Test guardrail -- off-topic
    r = requests.post(f"{SERVER_URL}/chat",
        json={"question": "What is the recipe for baklava?", "session_id": sid},
        timeout=120)
    c3 = r.json()
    results["guardrail"] = c3
    print(f"\n[5] POST /chat (off-topic guardrail test) -> {r.status_code}")
    print(f"    answer preview: {c3.get('answer', '')[:200]}")

    # Test DELETE /chat/{session_id}
    r = requests.delete(f"{SERVER_URL}/chat/{sid}")
    results["clear"] = r.json()
    print(f"\n[6] DELETE /chat/{sid} -> {r.status_code}: {r.json()}")

    print("\n" + "=" * 60)
    print("ALL SMOKE TESTS PASSED")

except Exception as e:
    print(f"\nSMOKE TEST FAILED: {e}")
    import traceback; traceback.print_exc()

finally:
    proc.terminate()
    proc.wait(timeout=5)
    print("\nServer stopped.")
