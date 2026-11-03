"""
read_pdfs.py  —  A simple local RAG pipeline
----------------------------------------------
Steps:
  1. Read every PDF in ./data
  2. Split pages into overlapping chunks
  3. Embed every chunk once (all-MiniLM-L6-v2, runs offline)
  4. On each question:
       a. Embed the question
       b. Find top-3 chunks by cosine similarity  (Retrieve)
       c. Build a prompt with those chunks        (Augment)
       d. Call Gemini to answer                   (Generate)
"""

import os
# pyrefly: ignore[missing-import]
import numpy as np
from pathlib import Path
# pyrefly: ignore[missing-import]
from dotenv import load_dotenv                          # Part 1 – read .env file
# pyrefly: ignore[missing-import]
from pypdf import PdfReader                             # Part 2 – open PDFs
# pyrefly: ignore[missing-import]
from sentence_transformers import SentenceTransformer   # Part 3 – local embeddings
# pyrefly: ignore[missing-import]
from google import genai                                # Part 4 – Gemini API client


# ── Part 5: Load the API key from .env ───────────────────────────────────────
#   .env is a plain text file that holds secrets.
#   We never put secrets directly in the code (someone might share the file).
#   load_dotenv() reads .env and puts the values into os.environ so the
#   rest of the program can access them with os.getenv().
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY or GEMINI_API_KEY == "your_api_key_here":
    raise SystemExit(
        "ERROR: Add your Gemini API key to the .env file.\n"
        "  Open .env and replace 'your_api_key_here' with your real key."
    )


# ── Part 6: Settings ──────────────────────────────────────────────────────────
CHUNK_SIZE    = 400
OVERLAP       = 50
TOP_K         = 3
EMBED_MODEL   = "all-MiniLM-L6-v2"
GEMINI_MODEL  = "gemini-3.6-flash"


# ── Part 7: Chunking function ─────────────────────────────────────────────────
def chunk_text(text, filename, page_num,
               chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    words  = text.split()
    chunks = []
    start  = 0
    while start < len(words):
        chunk_str = " ".join(words[start : start + chunk_size])
        chunks.append({
            "text"      : chunk_str,
            "filename"  : filename,
            "page"      : page_num,
            "chunk_idx" : len(chunks),
        })
        start += chunk_size - overlap
    return chunks


# ── Part 8: Cosine similarity ─────────────────────────────────────────────────
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)


# ── Part 9: Load the local embedding model ────────────────────────────────────
print("Loading embedding model...")
embedder = SentenceTransformer(EMBED_MODEL)
print(f"Model ready: {EMBED_MODEL}\n")


# ── Part 10: Load, chunk and embed all PDFs ───────────────────────────────────
DATA_DIR  = Path("data")
pdf_files = sorted(DATA_DIR.glob("*.pdf"))

if not pdf_files:
    raise SystemExit("No PDF files found in the 'data' folder.")

print(f"Found {len(pdf_files)} PDF(s). Loading and chunking...")
all_chunks = []

for pdf_path in pdf_files:
    reader = PdfReader(pdf_path)
    for page_num, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text()
        if not raw_text or not raw_text.strip():
            continue
        all_chunks.extend(chunk_text(raw_text, pdf_path.name, page_num))

print(f"Chunking done: {len(all_chunks)} chunks.\n")
print("Embedding all chunks (one-time setup)...")

chunk_embeddings = embedder.encode(
    [c["text"] for c in all_chunks],
    show_progress_bar=True,
    convert_to_numpy=True,
)
print(f"Done. Each chunk = {chunk_embeddings.shape[1]}-number vector.\n")


# ── Part 11: Retrieval — find the top-K most similar chunks ──────────────────
def retrieve(query, top_k=TOP_K):
    """
    Convert the question into a vector, then score every chunk.
    Return the top_k chunks and their similarity scores.
    """
    q_vec  = embedder.encode(query, convert_to_numpy=True)
    scores = [cosine_similarity(q_vec, chunk_embeddings[i])
              for i in range(len(all_chunks))]
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return [(all_chunks[i], scores[i]) for i in ranked[:top_k]]


# ── Part 12: Build the prompt ─────────────────────────────────────────────────
def build_prompt(question, results):
    """
    Assemble the text we send to Gemini.

    We give the model:
      - A strict instruction: answer ONLY from the provided context.
      - The actual context (the top-3 chunks), each labelled with its source.
      - The user's question.

    Why the strict instruction?
      Without it, the model uses its general knowledge and may invent answers
      that are not in the notes. This is called 'hallucination'.
      The instruction forces it to stay grounded in the retrieved text.
    """
    context_block = ""
    for rank, (chunk, score) in enumerate(results, start=1):
        context_block += (
            f"[Source {rank}] File: {chunk['filename']} | Page: {chunk['page']}\n"
            f"{chunk['text']}\n\n"
        )

    prompt = f"""You are a study assistant. Answer the question using ONLY the context below.
- Cite each source as (File: <filename>, Page: <page>) after each relevant sentence.
- If the answer is not in the context, say exactly: "I don't know based on these notes."
- Do not use any outside knowledge.

CONTEXT:
{context_block}
QUESTION: {question}

ANSWER:"""
    return prompt


# ── Part 13: Generate — call Gemini ──────────────────────────────────────────
def generate(prompt):
    """
    Send the prompt to Gemini and return the text response.

    genai.Client uses GEMINI_API_KEY to authenticate.
    client.models.generate_content() sends the prompt and gets a reply.
    """
    client   = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(
        model    = GEMINI_MODEL,
        contents = prompt,
    )
    return response.text


# ── Part 14: Interactive RAG loop ─────────────────────────────────────────────
print("=" * 70)
print("  NOTES RAG  —  ask questions about your PDFs  (type 'quit' to exit)")
print("=" * 70)

while True:
    question = input("\nYour question: ").strip()

    if not question:
        continue
    if question.lower() in ("quit", "exit", "q"):
        print("Bye!")
        break

    # Step A: Retrieve top-3 chunks
    print("\nSearching notes...")
    results = retrieve(question)

    # Step B: Build the prompt
    prompt = build_prompt(question, results)

    # Step C: Generate the answer
    print("Asking Gemini...\n")
    answer = generate(prompt)

    # Step D: Print the answer
    print("-" * 70)
    print("ANSWER:")
    print(answer)

    # Step E: Print the sources used
    print("\nSOURCES RETRIEVED:")
    for rank, (chunk, score) in enumerate(results, start=1):
        print(f"  [{rank}] {chunk['filename']}  |  Page {chunk['page']}"
              f"  |  similarity: {score:.3f}")
    print("-" * 70)
