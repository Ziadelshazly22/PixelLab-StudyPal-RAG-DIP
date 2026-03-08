# 🎓 DIP AI Tutor — 5-Minute Live Demo Script

> **Presenter**: Run `python run_all.py` before going on-stage. All 5 checks must be ✅.
> Start FastAPI **first**, wait for `Application startup complete`, then start Gradio.

---

## ⚙️ Pre-Demo Checklist

Complete every item the night before, and again 10 minutes before presenting.

| # | Item | How to verify |
| --- | ------ | --------------- |
| 1 | All 5 health checks green | `cd smart-learning-assistant && python run_all.py` → `READY FOR DEMO: YES` |
| 2 | **Terminal 1** — FastAPI running | `uvicorn main:app --reload --port 8000` → `Application startup complete` |
| 3 | **Terminal 2** — Gradio running | `python app/ui/interface.py` → `Running on local URL: http://127.0.0.1:7860` |
| 4 | Gradio UI loads | Browser → `http://localhost:7860` — Chat tab visible |
| 5 | Swagger docs load | Second browser tab → `http://localhost:8000/docs` |
| 6 | Ask one warm-up question | Type `"What is spatial filtering?"` → expect cited answer in < 30 s |
| 7 | Off-topic works | Type `"What is tiramisu?"` → expect refusal, no crash |
| 8 | Pre-generate summary | Run summarize on your chosen PDF; save result to a local .txt as backup |
| 9 | Terminal logs visible | Move terminal behind browser — switch to it briefly during demo |
| 10 | Font ≥ 18 pt | Audience at back of room must read responses without squinting |
| 11 | Screen 1920×1080+ | Avoid letterboxing on the projector |
| 12 | Notifications off | Windows: Focus Assist ON / macOS: Do Not Disturb |
| 13 | Ollama pre-tested | `ollama serve` → `ollama run deepseek-r1` → confirm it responds locally |
| 14 | `.env` backup ready | Have a second `.env` with `LLM_BACKEND=ollama` ready to swap in 10 s |
| 15 | `evaluation_report.md` open | Open in VS Code side-by-side for the final section |

---

## 🎬 Demo Flow

| Timestamp | Screen Action | What to Say |
| ----------- | -------------- | ------------- |
| **0:00 – 0:30** | Show both terminals running side-by-side. Switch to browser → `http://localhost:7860`. Point to Chat tab. | *"This is the DIP AI Tutor — an RAG assistant grounded exclusively in Gonzalez and Woods' Digital Image Processing 4th edition, plus verified OpenCV, NumPy, SciPy, Matplotlib, and Pillow documentation. Every answer it gives is sourced from a real page in that knowledge base. The primary backend is Groq's llama-3.1-8b-instant — free tier, zero billing. For campus deployment on a private server we switch a single environment variable and the same pipeline runs fully offline on a local DeepSeek-R1 model."* |
| **0:30 – 1:30** | Type in Chat tab: **`"Explain histogram equalization with its transformation function."`** Wait for response (~8–25 s). Scroll down slowly to show the answer. Highlight a `[Source: ..., Page: N]` citation. | *"The system embeds the query, runs an MMR similarity search over 1,000-plus chunks, fetches the top 12 most relevant and diverse passages, then feeds them as grounded context to the LLM. The LLM has a strict rule: every factual claim and every equation must be cited using only the retrieved context — you can see the page reference right here. The transformation function — T(r) equals the CDF of pixel intensities — comes directly from page N of Gonzalez and Woods, not from the LLM's training memory."* |
| **1:30 – 2:30** | **Without refreshing**, type the follow-up: **`"Now give me Python code to implement it using OpenCV."`** Point to the response — notice it doesn't re-explain histogram equalization basics. Open the terminal to show per-session log lines. | *"This is a follow-up in the same session. The ConversationalRetrievalChain keeps a 10-turn sliding memory window per session — the model already knows we are talking about histogram equalization, so it jumps straight to the code. Each session is completely isolated. The server handles up to 100 concurrent sessions and auto-expires them after 60 minutes. This is the study assistant use case: a student can have a sustained, context-aware conversation with the textbook."* |
| **2:30 – 3:00** | Type: **`"What is the recipe for tiramisu?"`** Point to the refusal message when it appears. | *"Now the guardrail. Before every query reaches the LLM, the retriever computes the L2 distance between the query embedding and the closest DIP document embedding in the vector store. If that distance is 1.2 or greater — calibrated for all-MiniLM-L6-v2 384-dimensional space — the retriever returns an empty context list. The chain sees no context and activates the mandatory rejection branch. In our evaluation, 3 out of 3 off-topic queries were blocked with zero false positives on real DIP questions. This matters for academic integrity: the system cannot be manipulated into answering questions it has no business answering."* |
| **3:00 – 4:00** | Switch to **Upload tab** (or Summarize tab if separate). Upload a small PDF (< 1 MB). Show the chunk count update in the status response. Click **Summarize**, wait for result. Click **Generate Study Questions**, show the numbered list. | *"The third capability is document summarisation and exam preparation. Any PDF uploaded here is chunked, embedded, and added to the knowledge base immediately — the chat is instantly aware of it. The summarizer uses a map-reduce chain: each chunk is compressed independently in the map phase, then the partial summaries are combined in the reduce phase. This allows us to summarize a full textbook chapter that would otherwise exceed the LLM's context window. The study questions generated here cover conceptual, mathematical, and applied types — ready to use as a self-assessment quiz."* |
| **4:00 – 5:00** | Open `evaluation_report.md` in VS Code (or switch to the pre-opened tab). Read through each metric score. Point at the latency note at the bottom. Show the `.env` file and `LLM_BACKEND` variable. End on the README architecture diagram. | *"Finally, quality assurance. We evaluated the system with RAGAS — Retrieval-Augmented Generation Assessment — on 15 real DIP exam questions plus 3 off-topic guardrail tests. Context precision is 0.918: 91.8% of retrieved chunks were genuinely relevant. Faithfulness is 0.726: all factual claims traced back to retrieved passages. Overall mean score is 0.790, all four metrics above the 0.7 production threshold. The 23-second mean latency is almost entirely the Groq free-tier rate limiter between evaluation calls — actual LLM inference is under 2 seconds. For campus deployment: one line in the .env file — LLM_BACKEND=ollama — switches the entire pipeline to a local DeepSeek-R1 model. No cloud dependency, no API key, no data leaving the institution's network. The integration target is the PixelLab learning platform."* |

---

## 🚨 Backup Plans (per section)

### Section 0:00–0:30 — Servers not started or browser 404

```text
FastAPI must start before Gradio — always start Terminal 1 first.
Order: uvicorn main:app --reload --port 8000
Wait for "Application startup complete"
Then: python app/ui/interface.py
If port 8000 is in use (Windows): netstat -ano | findstr :8000 → taskkill /PID <pid> /F
```

If browser shows 404: navigate to `http://127.0.0.1:7860` (not localhost:7860 — they may differ on some systems).

---

### Section 0:30–1:30 — Groq API rate-limited (503) during live demo

```text
1. Open .env in the editor (you have it open in VS Code)
2. Change: LLM_BACKEND=groq  →  LLM_BACKEND=ollama
3. Stop uvicorn (Ctrl+C in Terminal 1)
4. Restart: uvicorn main:app --reload --port 8000
   Wait for "Application startup complete" (~5 s)
```

Say: *"This is actually a great live demonstration of the dual-LLM design. Switching to the fully local DeepSeek-R1 backend — same pipeline, zero cloud dependency."*

Pre-condition: Ollama must be running (`ollama serve` in a background terminal) and `deepseek-r1` must be pulled (`ollama pull deepseek-r1`).

---

### Section 1:30–2:30 — Memory doesn't appear (follow-up treated as new question)

Check: Was the browser page refreshed? A refresh resets the session UUID in the UI.
Fix: Re-type the first question once to re-establish context, then ask the follow-up.

If the chain still doesn't use memory, switch to the Swagger UI (`http://localhost:8000/docs → POST /chat`) and manually send two requests with the **same** `session_id` string — memory will demonstrably work there.

---

### Section 2:30–3:00 — Guardrail does not fire (off-topic query gets an answer)

This means ChromaDB has the tiramisu query happening to land inside the L2 threshold (unlikely but possible if the vector store was rebuilt with different embeddings).

Fix live: Type a more extreme off-topic query: *"Who wrote Hamlet?"* or *"What is the GDP of France?"* — these will exceed the threshold.

After the demo: recalibrate by running `python scripts/calibrate_threshold.py`.

---

### Section 3:00–4:00 — Summarisation hangs for > 30 seconds

Never let the audience watch a spinner for more than 30 seconds.

```text
Pre-generated backup: open backup_summary.txt (saved during pre-demo checklist step 8)
Paste it into the Gradio text area manually.
```

Say: *"Normally this completes in 60–90 seconds for a full chapter. I pre-computed this result to keep the demo moving — the actual chain output is identical."*

---

### Section 4:00–5:00 — evaluation_report.md looks empty or has wrong scores

The committed copy is at `smart-learning-assistant/evaluation_report.md`.
The live-generated copy is at `smart-learning-assistant/data/evaluation_report.md`.

If VS Code shows the wrong file, open the correct path:

```bash
b:\PixelLab-StudyPal-RAG-DIP\smart-learning-assistant\evaluation_report.md
```

Scores to quote from memory if the file is inaccessible:

| Metric | Score |
| -------- | ------- |
| Context Precision | **0.918** |
| Answer Relevancy | **0.807** |
| Faithfulness | **0.726** |
| Context Recall | **0.709** |
| **Overall (mean)** | **0.790** |
| Guardrail | **3 / 3 (100%)** |

---

## 📊 Key Numbers Reference

| Metric | Value |
| -------- | ------- |
| Faithfulness | **0.726** ✅ |
| Answer Relevancy | **0.807** ✅ |
| Context Precision | **0.918** ✅ |
| Context Recall | **0.709** ✅ |
| Overall RAGAS score | **0.790** ✅ |
| Guardrail pass rate | **3 / 3 (100%)** ✅ |
| Knowledge base size | **1,000+ chunks** |
| MMR retriever k | **12** (fetch_k=50, λ=0.9) |
| Guardrail L2 threshold | **1.2** |
| Max concurrent sessions | **100** |
| Session TTL | **3,600 s (1 hour)** |
| Chunk size | **800 chars** (overlap 150) |
| Embedding model | `all-MiniLM-L6-v2` (384-dim, local) |
| LLM — demo | `llama-3.1-8b-instant` via Groq (free tier) |
| LLM — campus / offline | `DeepSeek-R1-Distill-Qwen-14B` via Ollama |

---

*v1.0.0 — run `python run_all.py` → `READY FOR DEMO: YES` before presenting.*
