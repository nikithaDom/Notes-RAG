"""
evaluate.py
-----------
Loads scratch/test_set.json (actually plain-text format), runs every question
through the same retrieval + generation pipeline as read_pdfs.py, then:

  1. Prints retrieval hit rate — was the correct page in the top-3 chunks?
  2. Prints each generated answer next to the expected answer for hand-grading.
  3. Saves a full results report to: scratch/eval_results.txt

Run:
    cd "C:\\Users\\anil\\OneDrive\\Desktop\\notes-rag"
    python evaluate.py
"""

import os
import re
import time
import json
# pyrefly: ignore[missing-import]
import numpy as np
from pathlib import Path
from datetime import datetime
# pyrefly: ignore[missing-import]
from dotenv import load_dotenv
# pyrefly: ignore[missing-import]
from pypdf import PdfReader
# pyrefly: ignore[missing-import]
from sentence_transformers import SentenceTransformer
# pyrefly: ignore[missing-import]
from google import genai
# pyrefly: ignore[missing-import]
from google.genai import errors as genai_errors


# ══════════════════════════════════════════════════════════════════════════════
# PART 1 — Configuration
# ══════════════════════════════════════════════════════════════════════════════
CHUNK_SIZE    = 400
OVERLAP       = 50
TOP_K         = 3
EMBED_MODEL   = "all-MiniLM-L6-v2"
GEMINI_MODEL  = "gemini-3.5-flash-lite"    # switched: 3.6-flash hit daily quota (20 req/day free tier)

TEST_FILE     = Path("scratch/test_set.txt")
OUTPUT_FILE   = Path("scratch/eval_results.txt")
RESULTS_JSON  = Path("scratch/results.json")   # incremental save — one entry per question
DATA_DIR      = Path("data")

INTER_Q_DELAY = 14      # seconds between questions → stays under 5 req/min
RETRY_BUFFER  = 2       # extra seconds added on top of the suggested retry delay


# ══════════════════════════════════════════════════════════════════════════════
# PART 2 — Load API key
# ══════════════════════════════════════════════════════════════════════════════
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY or GEMINI_API_KEY == "your_api_key_here":
    raise SystemExit("ERROR: Set GEMINI_API_KEY in your .env file.")


# ══════════════════════════════════════════════════════════════════════════════
# PART 3 — Parse the test set
#
# The file uses this plain-text format per question:
#   <N>.  <question text>
#   Answer: <expected answer>
#   Source: <source description, e.g. "Unit 3 Part 2, p. 14">
#
# We extract:
#   - question  : what to ask the RAG system
#   - answer    : the expected correct answer (for hand-grading)
#   - pages     : list of page numbers that count as a retrieval hit
#   - source_raw: the original source string (shown in the report)
# ══════════════════════════════════════════════════════════════════════════════
def parse_test_set(path: Path) -> list[dict]:
    """
    Read the plain-text test file and return a list of test-case dicts.
    Each dict has keys: id, question, answer, source_raw, pages.
    """
    raw = path.read_text(encoding="utf-8")

    # Split on blank lines — each block is one test case
    blocks = [b.strip() for b in re.split(r"\n\s*\n", raw) if b.strip()]

    test_cases = []
    for block in blocks:
        lines = block.splitlines()

        # ── Question: collect lines until Answer:/Source:, strip tags/numbers ──
        question_parts = []
        i = 0
        while i < len(lines):
            l = lines[i].strip()
            if l.startswith("Answer:") or l.startswith("Source:"):
                break
            # Remove leading "N. " number
            l = re.sub(r"^\d+\.\s*", "", l)
            # Remove any [bracketed tag] anywhere in the line
            l = re.sub(r"\[.*?\]\s*", "", l).strip()
            if l:
                question_parts.append(l)
            i += 1

        question = " ".join(question_parts).strip()

        # ── Expected answer ───────────────────────────────────────────────────
        answer_lines = []
        source_raw   = "not specified"
        for line in lines:
            if line.strip().startswith("Answer:"):
                answer_lines.append(line.strip()[len("Answer:"):].strip())
            elif line.strip().startswith("Source:"):
                source_raw = line.strip()[len("Source:"):].strip()

        expected_answer = " ".join(answer_lines).strip()

        # ── Extract page numbers from the source string ───────────────────────
        # Matches "p. 14", "p.14", "p. 11", "(p. 15)"
        pages = [int(m) for m in re.findall(r"p\.?\s*(\d+)", source_raw)]

        test_cases.append({
            "id"            : len(test_cases) + 1,
            "question"      : question,
            "answer"        : expected_answer,
            "source_raw"    : source_raw,
            "pages"         : pages,          # page numbers that count as a hit
        })

    return test_cases


# ══════════════════════════════════════════════════════════════════════════════
# PART 4 — Build the chunk index (same logic as read_pdfs.py)
# ══════════════════════════════════════════════════════════════════════════════
def chunk_text(text, filename, page_num):
    words, chunks, start = text.split(), [], 0
    while start < len(words):
        chunks.append({
            "text"     : " ".join(words[start : start + CHUNK_SIZE]),
            "filename" : filename,
            "page"     : page_num,
        })
        start += CHUNK_SIZE - OVERLAP
    return chunks

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)

def build_index(data_dir: Path, embedder: SentenceTransformer):
    all_chunks = []
    for pdf_path in sorted(data_dir.glob("*.pdf")):
        reader = PdfReader(pdf_path)
        for page_num, page in enumerate(reader.pages, start=1):
            raw = page.extract_text()
            if raw and raw.strip():
                all_chunks.extend(chunk_text(raw, pdf_path.name, page_num))

    embeddings = embedder.encode(
        [c["text"] for c in all_chunks],
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    return all_chunks, embeddings


# ══════════════════════════════════════════════════════════════════════════════
# PART 5 — Retrieval
# ══════════════════════════════════════════════════════════════════════════════
def retrieve(query, all_chunks, embeddings, embedder, top_k=TOP_K):
    q_vec  = embedder.encode(query, convert_to_numpy=True)
    scores = [cosine_similarity(q_vec, embeddings[i]) for i in range(len(all_chunks))]
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return [(all_chunks[i], scores[i]) for i in ranked[:top_k]]


# ══════════════════════════════════════════════════════════════════════════════
# PART 6 — Retrieval hit check
#
# A "hit" means: at least one of the top-3 retrieved chunks came from
# a page that the test set lists as a correct source.
#
# Why page-level?  We can't match exact chunks (the chunk boundaries are
# arbitrary), but we can check whether the right page was surfaced.
# ══════════════════════════════════════════════════════════════════════════════
def is_hit(results, expected_pages):
    if not expected_pages:
        return None                         # unanswerable — skip hit counting
    retrieved_pages = {chunk["page"] for chunk, _ in results}
    return bool(retrieved_pages & set(expected_pages))


# ══════════════════════════════════════════════════════════════════════════════
# PART 7 — Prompt builder + Gemini call (same as read_pdfs.py)
# ══════════════════════════════════════════════════════════════════════════════
def build_prompt(question, results):
    context_block = ""
    for rank, (chunk, score) in enumerate(results, start=1):
        context_block += (
            f"[Source {rank}] File: {chunk['filename']} | Page: {chunk['page']}\n"
            f"{chunk['text']}\n\n"
        )
    return (
        "You are a study assistant. Answer the question using ONLY the context below.\n"
        "- Cite each source as (File: <filename>, Page: <page>) after each relevant sentence.\n"
        '- If the answer is not in the context, say exactly: "I don\'t know based on these notes."\n'
        "- Do not use any outside knowledge.\n\n"
        f"CONTEXT:\n{context_block}"
        f"QUESTION: {question}\n\n"
        "ANSWER:"
    )

def generate(prompt, api_key):
    """
    Call Gemini with automatic 429 retry.

    Distinguishes two kinds of 429:
      - Per-minute rate limit  -> wait the suggested delay + RETRY_BUFFER, retry once.
      - Daily quota exhausted  -> print a clear message and exit immediately.
    """
    client = genai.Client(api_key=api_key)

    for attempt in (1, 2):
        try:
            response = client.models.generate_content(
                model    = GEMINI_MODEL,
                contents = prompt,
            )
            return response.text.strip()

        except Exception as exc:
            err_str = str(exc)

            # ── Only handle 429; re-raise everything else ─────────────────────
            if "429" not in err_str and "RESOURCE_EXHAUSTED" not in err_str:
                raise

            # ── Daily quota exhausted -> exit cleanly ─────────────────────────
            if "per day" in err_str.lower() or "per_day" in err_str.lower():
                raise SystemExit(
                    "\nDAILY QUOTA EXHAUSTED for model: " + GEMINI_MODEL + "\n"
                    "Results saved so far: " + str(RESULTS_JSON) + "\n"
                    "Options:\n"
                    "  1. Wait until tomorrow (free tier resets at midnight UTC).\n"
                    "  2. Change GEMINI_MODEL in evaluate.py to another working model.\n"
                    "  3. Re-run -- already-answered questions will be skipped.\n"
                )

            # ── Per-minute rate limit -> retry once after suggested delay ──────
            if attempt == 2:
                raise

            delay_match = re.search(r"['\"]retryDelay['\"]:\s*['\"](\d+)s['\"]", err_str)
            wait_secs   = int(delay_match.group(1)) if delay_match else 60
            total_wait  = wait_secs + RETRY_BUFFER

            print(f"\n    [429] Rate limited. Waiting {total_wait}s before retry...",
                  flush=True)
            time.sleep(total_wait)


# ══════════════════════════════════════════════════════════════════════════════
# PART 8 — Format and save the results report
#
# The report has three sections:
#   A. Summary: hit rate and timestamp
#   B. Per-question detail: question / retrieved sources / generated answer /
#      expected answer — everything you need to grade by hand
#   C. Unanswerable questions listed separately
# ══════════════════════════════════════════════════════════════════════════════
def format_report(test_cases, row_results, hit_rate, total_answerable):
    lines = []
    ts    = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines.append("=" * 72)
    lines.append("  RAG EVALUATION REPORT")
    lines.append(f"  Generated: {ts}")
    lines.append("=" * 72)
    lines.append("")
    lines.append("SUMMARY")
    lines.append("-" * 40)
    lines.append(f"  Total questions    : {len(test_cases)}")
    lines.append(f"  Answerable (w/ src): {total_answerable}")
    lines.append(f"  Retrieval hit rate : {hit_rate:.1%}  "
                 f"({sum(r['hit'] for r in row_results if r['hit'] is not None)}"
                 f" / {total_answerable} correct pages retrieved in top-3)")
    lines.append("")
    lines.append("=" * 72)
    lines.append("  PER-QUESTION RESULTS  (grade the 'Generated' column by hand)")
    lines.append("=" * 72)

    for r in row_results:
        tc = r["test_case"]
        lines.append("")
        lines.append(f"Q{tc['id']:02d}. {tc['question']}")
        lines.append(f"  Expected source : {tc['source_raw']}")

        # Retrieved chunks
        lines.append("  Retrieved chunks:")
        for rank, (chunk, score) in enumerate(r["results"], start=1):
            hit_marker = ""
            if tc["pages"] and chunk["page"] in tc["pages"]:
                hit_marker = "  << HIT"
            lines.append(f"    [{rank}] {chunk['filename']}  Page {chunk['page']}"
                         f"  sim={score:.3f}{hit_marker}")

        # Hit/miss label
        if r["hit"] is None:
            lines.append("  Retrieval       : N/A (unanswerable question)")
        elif r["hit"]:
            lines.append("  Retrieval       : HIT")
        else:
            lines.append("  Retrieval       : MISS")

        lines.append("")
        lines.append("  Generated answer:")
        for line in r["generated"].splitlines():
            lines.append(f"    {line}")
        lines.append("")
        lines.append("  Expected answer:")
        lines.append(f"    {tc['answer']}")
        lines.append("")
        lines.append("  [ ] Correct   [ ] Partially correct   [ ] Wrong")
        lines.append("-" * 72)

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# PART 9 — Incremental JSON helpers
#
# results.json holds a list of completed result objects.
# Each entry stores the question id, hit flag, generated answer, and sources.
# We load it at startup and skip any question id already present.
# After each question we append the new result and overwrite the file.
# This means a crash or keyboard interrupt loses at most one question.
# ══════════════════════════════════════════════════════════════════════════════
def load_saved_results(path: Path) -> dict:
    """Return a dict mapping question id → saved result, or {} if file missing."""
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return {entry["id"]: entry for entry in data}
    except (json.JSONDecodeError, KeyError):
        return {}

def save_results(path: Path, saved: dict) -> None:
    """Write all saved results (sorted by id) to the JSON file."""
    ordered = sorted(saved.values(), key=lambda e: e["id"])
    path.write_text(json.dumps(ordered, indent=2, ensure_ascii=False),
                    encoding="utf-8")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    total_qs = None     # filled after parsing

    # ── Step 1: Parse test set ────────────────────────────────────────────────
    print(f"Loading test set from {TEST_FILE}...")
    test_cases = parse_test_set(TEST_FILE)
    total_qs   = len(test_cases)
    print(f"  {total_qs} test cases loaded.\n")

    # ── Step 2: Load previously saved results (skip-if-done) ─────────────────
    saved = load_saved_results(RESULTS_JSON)
    if saved:
        print(f"  Resuming — {len(saved)} question(s) already saved in {RESULTS_JSON}.")
        print(f"  Skipping: {sorted(saved.keys())}\n")

    # ── Step 3: Load embedding model ─────────────────────────────────────────
    print("Loading embedding model...")
    embedder = SentenceTransformer(EMBED_MODEL)
    print(f"  Model ready: {EMBED_MODEL}\n")

    # ── Step 4: Build chunk index ─────────────────────────────────────────────
    print("Building chunk index from PDFs...")
    all_chunks, embeddings = build_index(DATA_DIR, embedder)
    print(f"  {len(all_chunks)} chunks indexed.\n")

    # ── Step 5: Run each test case ────────────────────────────────────────────
    row_results = []

    print("=" * 72)
    print(f"  Running {total_qs} test cases  |  {INTER_Q_DELAY}s between calls")
    print("=" * 72)

    for idx, tc in enumerate(test_cases, start=1):
        qid = tc["id"]

        # ── Progress line ─────────────────────────────────────────────────────
        print(f"\nQ{idx:02d}/{total_qs}  id={qid}  "
              f"{tc['question'][:60]}{'...' if len(tc['question']) > 60 else ''}")

        # ── Skip if already in results.json ───────────────────────────────────
        if qid in saved:
            print("  SKIPPED (already saved)")
            # Rebuild row_result from saved data for the final report
            entry   = saved[qid]
            # Re-run retrieval only (no API call needed)
            results = retrieve(tc["question"], all_chunks, embeddings, embedder)
            hit     = is_hit(results, tc["pages"])
            row_results.append({
                "test_case" : tc,
                "results"   : results,
                "hit"       : hit,
                "generated" : entry["generated"],
            })
            continue

        # ── Retrieve ──────────────────────────────────────────────────────────
        results = retrieve(tc["question"], all_chunks, embeddings, embedder)
        hit     = is_hit(results, tc["pages"])

        hit_label = "N/A" if hit is None else ("HIT" if hit else "MISS")
        print(f"  Retrieval: {hit_label}")

        # ── Generate (with 429 retry built in) ────────────────────────────────
        print(f"  Generating...", end=" ", flush=True)
        prompt    = build_prompt(tc["question"], results)
        generated = generate(prompt, GEMINI_API_KEY)
        print("done.")

        # ── Save result to JSON immediately ───────────────────────────────────
        saved[qid] = {
            "id"        : qid,
            "question"  : tc["question"],
            "generated" : generated,
            "hit"       : hit,
            "sources"   : [
                {"filename": c["filename"], "page": c["page"], "score": round(float(s), 4)}
                for c, s in results
            ],
            "expected_answer" : tc["answer"],
            "source_raw"      : tc["source_raw"],
        }
        save_results(RESULTS_JSON, saved)
        print(f"  Saved to {RESULTS_JSON}.")

        row_results.append({
            "test_case" : tc,
            "results"   : results,
            "hit"       : hit,
            "generated" : generated,
        })

        # ── Rate-limit pause (skip after the last question) ───────────────────
        if idx < total_qs:
            remaining = total_qs - idx
            print(f"  Waiting {INTER_Q_DELAY}s  ({remaining} question(s) remaining)...",
                  end=" ", flush=True)
            time.sleep(INTER_Q_DELAY)
            print("continuing.")

    # ── Step 6: Compute hit rate ──────────────────────────────────────────────
    answerable = [r for r in row_results if r["hit"] is not None]
    hits       = sum(1 for r in answerable if r["hit"])
    hit_rate   = hits / len(answerable) if answerable else 0.0

    # ── Step 7: Print summary ─────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  RESULTS SUMMARY")
    print("=" * 72)
    print(f"  Retrieval hit rate : {hit_rate:.1%}  ({hits} / {len(answerable)})")
    print()
    for r in row_results:
        tc     = r["test_case"]
        label  = "N/A  " if r["hit"] is None else ("HIT  " if r["hit"] else "MISS ")
        preview = r["generated"][:60].replace("\n", " ")
        print(f"  Q{tc['id']:02d} [{label}]  {preview}...")
    print()

    # ── Step 8: Save full human-readable report ───────────────────────────────
    report = format_report(test_cases, row_results, hit_rate, len(answerable))
    OUTPUT_FILE.write_text(report, encoding="utf-8")
    print(f"  Full report : {OUTPUT_FILE.resolve()}")
    print(f"  Raw JSON    : {RESULTS_JSON.resolve()}")
    print("=" * 72)


if __name__ == "__main__":
    main()
