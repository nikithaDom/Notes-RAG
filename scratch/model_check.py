"""
model_check.py
--------------
Lists every model visible to your Gemini API key,
then tests each one with a tiny prompt and prints WORKS or FAILED.

Run:
    cd "C:\\Users\\anil\\OneDrive\\Desktop\\notes-rag"
    python scratch\\model_check.py
"""

import os
# pyrefly: ignore[missing-import]
from dotenv import load_dotenv
# pyrefly: ignore[missing-import]
from google import genai

# ── Part 1: Load API key ──────────────────────────────────────────────────────
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY or API_KEY == "your_api_key_here":
    raise SystemExit("ERROR: Set GEMINI_API_KEY in your .env file.")

client = genai.Client(api_key=API_KEY)

# ── Part 2: List every model the key can see ──────────────────────────────────
#   client.models.list() returns a page of model objects.
#   Each model has a .name like "models/gemini-2.0-flash".
print("Fetching model list...\n")

all_models = list(client.models.list())

if not all_models:
    raise SystemExit("No models returned. Check your API key.")

# Keep only models that support generateContent (i.e. text generation).
# Some entries are embedding-only or tuning-only models.
generative = [
    m for m in all_models
    if hasattr(m, "supported_actions") and "generateContent" in (m.supported_actions or [])
]

# Fall back to all models if the field is missing
if not generative:
    generative = all_models

print(f"Found {len(all_models)} models total, "
      f"{len(generative)} support generateContent.\n")
print(f"{'MODEL':<45}  {'STATUS':<8}  NOTE")
print("-" * 75)

# ── Part 3: Test each model with a tiny prompt ────────────────────────────────
#   We send "Reply with one word: hello" and expect any non-empty response.
#   If the call raises an exception we print FAILED plus the error type.
TEST_PROMPT = "Reply with one word: hello"
results = []

for m in generative:
    model_id = m.name   # e.g. "models/gemini-2.0-flash"

    try:
        response = client.models.generate_content(
            model    = model_id,
            contents = TEST_PROMPT,
        )
        reply = (response.text or "").strip().replace("\n", " ")[:40]
        status = "WORKS"
        note   = f'"{reply}"'
    except Exception as exc:
        status = "FAILED"
        note   = type(exc).__name__ + ": " + str(exc)[:60]

    results.append((model_id, status, note))
    # Print immediately so you can watch progress
    print(f"{model_id:<45}  {status:<8}  {note}")

# ── Part 4: Summary ───────────────────────────────────────────────────────────
works  = sum(1 for _, s, _ in results if s == "WORKS")
failed = sum(1 for _, s, _ in results if s == "FAILED")

print("-" * 75)
print(f"\nSUMMARY:  {works} WORKS  |  {failed} FAILED  |  {len(results)} total tested\n")
print("Models that WORK:")
for model_id, status, _ in results:
    if status == "WORKS":
        print(f"  {model_id}")

