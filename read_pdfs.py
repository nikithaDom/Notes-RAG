"""
read_pdfs.py
------------
1. Reads every PDF in ./data
2. Splits each page into overlapping chunks
3. Embeds every chunk once using a local model (all-MiniLM-L6-v2)
4. At search time: embeds the question, finds top-3 chunks by cosine similarity
"""

import numpy as np                              # Part 1a - math on arrays of numbers
from pathlib import Path                        # Part 1b - find files on disk
from pypdf import PdfReader                     # Part 1c - open and read PDF files
from sentence_transformers import SentenceTransformer  # Part 1d - the embedding model


# ── Part 2: Settings ──────────────────────────────────────────────────────────
CHUNK_SIZE  = 400                       # target words per chunk
OVERLAP     = 50                        # words shared between neighbouring chunks
TOP_K       = 3                         # how many results to show
MODEL_NAME  = "all-MiniLM-L6-v2"       # small, fast, free, runs fully offline


# ── Part 3: The chunking function (unchanged) ─────────────────────────────────
def chunk_text(text, filename, page_num,
               chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    """
    Split `text` into overlapping word chunks.
    Each chunk is a dict with: text, filename, page, chunk_idx.
    """
    words  = text.split()
    chunks = []
    start  = 0

    while start < len(words):
        end       = start + chunk_size
        chunk_str = " ".join(words[start:end])

        chunks.append({
            "text"      : chunk_str,
            "filename"  : filename,
            "page"      : page_num,
            "chunk_idx" : len(chunks),
        })
        start += chunk_size - overlap   # slide forward, keeping `overlap` words

    return chunks


# ── Part 4: Cosine similarity function ───────────────────────────────────────
def cosine_similarity(vec_a, vec_b):
    """
    Measure how similar two embeddings are.

    An embedding is a list of numbers (a vector) that captures meaning.
    Cosine similarity compares the *direction* of two vectors:
      - Score of  1.0  → identical meaning
      - Score of  0.0  → completely unrelated
      - Score of -1.0  → opposite meaning (rare in practice)

    Formula:
        similarity = (A · B) / (|A| × |B|)

    Plain English:
        "Multiply matching positions, add them up,
         then adjust so size does not matter — only direction does."
    """
    dot_product = np.dot(vec_a, vec_b)            # Step A: multiply & sum
    norm_a      = np.linalg.norm(vec_a)           # Step B: length of vector A
    norm_b      = np.linalg.norm(vec_b)           # Step C: length of vector B
    return dot_product / (norm_a * norm_b + 1e-10)  # Step D: divide (avoid ÷0)


# ── Part 5: Load the embedding model ─────────────────────────────────────────
#   This downloads ~90 MB once, then caches it locally.
#   After the first run it works fully offline.
print("Loading embedding model...")
model = SentenceTransformer(MODEL_NAME)
print(f"Model ready: {MODEL_NAME}\n")


# ── Part 6: Load and chunk all PDFs ──────────────────────────────────────────
DATA_DIR  = Path("data")
pdf_files = sorted(DATA_DIR.glob("*.pdf"))

if not pdf_files:
    print("No PDF files found in the 'data' folder.")
    exit()

print(f"Found {len(pdf_files)} PDF(s). Loading and chunking...\n")

all_chunks = []

for pdf_path in pdf_files:
    reader      = PdfReader(pdf_path)
    total_pages = len(reader.pages)

    for page_num, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text()

        if not raw_text or not raw_text.strip():
            continue                    # skip blank / image-only pages

        page_chunks = chunk_text(
            text     = raw_text,
            filename = pdf_path.name,
            page_num = page_num,
        )
        all_chunks.extend(page_chunks)

print(f"Chunking done: {len(all_chunks)} chunks from {len(pdf_files)} PDF(s).")


# ── Part 7: Embed every chunk ─────────────────────────────────────────────────
#   We do this ONCE at startup.
#   Each chunk's text → a list of 384 numbers that captures its meaning.
#   These numbers are stored in a 2-D array: one row per chunk.
#
#   Why embed at startup?
#     Embedding is slow (~1 second per 100 chunks).
#     If we embedded on every search, a 1000-chunk library would take 10 seconds
#     per question. Doing it once means search is instant.
print("\nEmbedding all chunks (this may take a moment)...")

chunk_texts      = [c["text"] for c in all_chunks]   # pull out just the text
chunk_embeddings = model.encode(                       # convert to numbers
    chunk_texts,
    show_progress_bar = True,
    convert_to_numpy  = True,
)

print(f"Done. Each chunk is now a vector of {chunk_embeddings.shape[1]} numbers.\n")


# ── Part 8: The search function ───────────────────────────────────────────────
def search(query, top_k=TOP_K):
    """
    Find the top_k chunks most similar in meaning to the query.

    Steps:
      A. Embed the question → a vector of 384 numbers.
      B. Compute cosine similarity between the question vector
         and every chunk vector.
      C. Sort chunks by similarity score (highest = most similar).
      D. Return the top_k.
    """
    # Step A: turn the question into a vector
    query_embedding = model.encode(query, convert_to_numpy=True)

    # Step B: score every chunk
    scores = [
        cosine_similarity(query_embedding, chunk_embeddings[i])
        for i in range(len(all_chunks))
    ]

    # Step C: sort indices by score, best first
    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True,
    )

    # Step D: return top_k (chunk dict + its score)
    return [
        (all_chunks[i], scores[i])
        for i in ranked_indices[:top_k]
    ]


# ── Part 9: Interactive search loop ──────────────────────────────────────────
print("=" * 70)
print("  SEMANTIC SEARCH  (type 'quit' to exit)")
print("=" * 70)

while True:
    query = input("\nYour question: ").strip()

    if not query:
        continue
    if query.lower() in ("quit", "exit", "q"):
        print("Bye!")
        break

    results = search(query)

    print()
    for rank, (chunk, score) in enumerate(results, start=1):
        print(f"  -- Result {rank}  (similarity: {score:.3f}) " + "-" * 28)
        print(f"  File : {chunk['filename']}")
        print(f"  Page : {chunk['page']}  |  Chunk: {chunk['chunk_idx']}")
        print(f"  Text : {chunk['text'][:300]}...")
        print()
