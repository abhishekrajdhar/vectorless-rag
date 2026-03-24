# Vectorless RAG / Page Index

This project implements a production-oriented, explainable Vectorless RAG pipeline (page-index retrieval). It purposely avoids embeddings and vector databases: retrieval is done by reasoning over structured document representations (pages, headings, and metadata) and deterministic LLM-guided selection.

This README explains the architecture, how the pipeline works, how to run the Streamlit demo, and guidance for production hardening.

## Goals

- Provide page-level retrieval (no embeddings) using symbolic indices (topic → pages).
- Use LLMs (OpenAI or Gemini) for structured reasoning steps: query decomposition, candidate selection, and final answer generation with citations.
- Produce traceable, auditable retrieval decisions and step-by-step reasoning in the UI.

## Key components

- `page_index/llm.py` — low-level LLM wrapper and optional Gemini support.
- `page_index/providers.py` — LLM provider abstraction (Mock / OpenAI / Gemini) and factory.
- `page_index/utils.py` — utilities: token counting, JSON extraction, PDF helpers, safe LLM wrappers.
- `page_index/page_index_md.py` — Markdown → hierarchical structure (tree) extractor.
- `page_index/engine.py` — the Vectorless RAG engine (parsing, symbolic index, query decomposition, candidate selection, LLM-guided selection, context construction, answer generation).
- `streamlit_app.py` — explainability-focused Streamlit UI that shows decomposition, candidates, selection reasoning, context, and final answers.
- `tests/` — unit and simple integration tests used for validation.

## How the pipeline works (conceptual)

1. Document parsing
   - Markdown: converted into a hierarchical tree (top-level nodes → synthetic pages).
   - PDF: true page-level extraction (one page object per physical PDF page, stored as `page` entries).

2. Symbolic index
   - Build lightweight index mapping tokenized keywords (from titles and sections) to page numbers.

3. Query decomposition
   - Heuristic token extraction augmented by an LLM-assisted decomposition step that returns `{ intent, topics, constraints }`.

4. Candidate selection
   - Use keyword lookup in the symbolic index, title matches, and heuristic scoring to shortlist candidate pages (no similarity search).

5. LLM-guided selection
   - Provide the LLM with candidate titles/summaries and ask it to pick the most relevant pages and explain why (JSON response). This is deterministic when using low temperature.

6. Context construction
   - Concatenate selected pages (with page markers), apply token-aware truncation, and preserve page numbers and section references.

7. Answer generation
   - Final LLM call must output enforced JSON with `{ answer, sources, reasoning }`. Sources list page numbers and reasons.

8. Explainability
   - The Streamlit UI shows decomposition, candidates, LLM selection, context sources, and the final answer with source attributions.

## Quick start

1. Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run tests:

```bash
pytest -q
```

3. Run the Streamlit demo (local):

```bash
streamlit run streamlit_app.py
```

Open the displayed local URL in your browser. The app UI lets you upload a Markdown or PDF file, enter a query, choose an LLM provider (mock / openai / gemini) and run the explainable retrieval pipeline.

## Configuring LLM providers

- OpenAI: provide your key as an environment variable or via the UI sidebar (dev only).
  - Environment: `export CHATGPT_API_KEY="sk-..."`
  - In the Streamlit UI select `openai` and either paste a key in the sidebar (development) or set it server-side for production.

- Gemini: provide a Gemini API key via `GEMINI_API_KEY` or use the UI option.
  - Environment: `export GEMINI_API_KEY="..."`
  - The project implements a minimal Gemini HTTP wrapper; for production we recommend using Google Cloud authentication (ADC / service account) — tell me if you want this flow added.

Security note: Do not paste production API keys into the UI in a shared environment. Use server-side configuration or a secrets manager in production.

## Running with PDFs

- Upload a PDF in the Streamlit UI. The engine will parse each physical page and create true page objects.
- If a PDF is scanned (image without text), consider adding OCR (Tesseract) — this is not enabled by default.

## Observability & Logging

- The engine returns a structured trace including:
  - `decomposition` (intent/topics)
  - `symbolic_index` (topic→pages)
  - `candidates` (page, score, reasons)
  - `guided_selection` (selected pages + LLM reasoning)
  - `context_sources` (pages included in the final prompt)
  - `answer` (final JSON answer with sources)

- For production, wire module loggers to a structured JSON log sink and capture metrics: LLM latency, token usage (via `count_tokens`), retrieval precision.

## Testing & Evaluation

- Unit tests exist in `tests/` for core utilities and an MD end-to-end path. Add integration tests for PDF parsing and provider-specific logic.
- Recommended evaluation:
  - Retrieval precision (page-level): % of times gold pages are included in selected pages.
  - Explainability quality: human-evaluated score of selection justifications.
  - Hallucination regression: unit tests that fail when the LLM output contains unsupported facts.

## Production hardening recommendations

1. Provider fallback and retries
   - Add OpenAI → Gemini fallback for availability. Use exponential backoff and circuit breaker patterns.

2. Persistent caching
   - Cache parsed documents and page summaries (e.g., Redis or file store) to avoid repeated LLM calls.

3. Secret management
   - Use environment-only keys or vaults. Do not accept production keys via the UI.

4. Authentication & Access control
   - Add an app-level auth around the Streamlit app (e.g., reverse proxy + auth) when deploying.

5. Monitoring & observability
   - Export metrics (Prometheus) and logs (structured JSON). Record retrieval traces for audits.

6. Safety
   - Sanitize document text before sending to LLM (remove leading/trailing instruction-like lines).
   - Apply rate-limits and quotas.

## Extensibility and next steps

- Add OCR fallback for scanned PDFs (pytesseract).
- Add page-level heading/section extraction inside each PDF page to populate `sections` with real headings.
- Add provider-specific adapters using official SDKs for better rate-limit handling and streaming.
- Improve prompts and create a small prompt library for decomposition / selection / final-answer templates.

## Contributing

- Fork the repo and open PRs. Add unit tests for any new behavior. Keep provider keys out of commits.

## Contact

If you want me to implement any of the recommended next steps (provider fallback, PDF OCR, persistent cache, or production deployment with CI), tell me which and I'll implement it.
