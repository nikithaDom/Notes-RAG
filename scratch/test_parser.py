import re
from pathlib import Path

def parse_test_set(path):
    raw    = path.read_text(encoding="utf-8")
    blocks = [b.strip() for b in re.split(r"\n\s*\n", raw) if b.strip()]
    cases  = []
    for block in blocks:
        lines  = block.splitlines()
        q_line = re.sub(r"^\d+\.\s*", "", lines[0])
        q_line = re.sub(r"\[.*?\]\s*", "", q_line).strip()
        question_parts = [q_line]
        i = 1
        while i < len(lines):
            l = lines[i].strip()
            if l.startswith("Answer:") or l.startswith("Source:"):
                break
            if re.match(r"^\[.*?\]\s*$", l):
                i += 1
                continue
            question_parts.append(l)
            i += 1
        question   = " ".join(p for p in question_parts if p).strip()
        source_raw = "not specified"
        for line in lines:
            if line.strip().startswith("Source:"):
                source_raw = line.strip()[len("Source:"):].strip()
        pages = [int(m) for m in re.findall(r"p\.?\s*(\d+)", source_raw)]
        cases.append({"id": len(cases)+1, "question": question, "pages": pages, "source_raw": source_raw})
    return cases

cases = parse_test_set(Path("scratch/test_set.txt"))
print("Parsed", len(cases), "test cases")
for c in cases[:5]:
    qid  = c["id"]
    qtxt = c["question"][:65]
    pg   = c["pages"]
    src  = c["source_raw"]
    print("  Q" + str(qid) + ": " + qtxt)
    print("      pages=" + str(pg) + "  src=" + src)

