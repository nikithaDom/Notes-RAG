from pathlib import Path
from pypdf import PdfReader

CHUNK_SIZE = 400
OVERLAP    = 50

def chunk_text(text, filename, page_num, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    words  = text.split()
    chunks = []
    start  = 0
    while start < len(words):
        end         = start + chunk_size
        chunk_words = words[start:end]
        chunks.append({
            "text"      : " ".join(chunk_words),
            "filename"  : filename,
            "page"      : page_num,
            "chunk_idx" : len(chunks),
        })
        start += chunk_size - overlap
    return chunks

pdf_path = Path("data/Unit3 Part2 Basic Computer Organization.pdf")
r = PdfReader(pdf_path)
for i, page in enumerate(r.pages, 1):
    txt = page.extract_text()
    if txt and txt.strip():
        chunks = chunk_text(txt, pdf_path.name, i)
        print("FILE :", pdf_path.name)
        print("PAGE :", i, " | chunks:", len(chunks))
        for c in chunks[:3]:
            print()
            print("  Chunk", c["chunk_idx"], "| words:", len(c["text"].split()))
            print("  Text:", c["text"][:150])
        break
