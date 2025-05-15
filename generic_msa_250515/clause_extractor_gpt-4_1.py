#!/usr/bin/env python3
"""
extract_clauses.py
──────────────────
Convert plain-text OCR contracts into clause-tagged files with Azure OpenAI.
"""

import os, re, sys, textwrap
from pathlib import Path
from typing import List
from dotenv import load_dotenv
from tqdm import tqdm
from openai import AzureOpenAI

# ────────────────────────────────────────
# 1.  Azure / environment
# ────────────────────────────────────────
ROOT = Path(__file__).resolve().parent  # script folder
load_dotenv(ROOT.parent / ".env")

client = AzureOpenAI(
    api_key        = os.getenv("AZURE_OPENAI_API_KEY"),
    api_version    = os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"),
)
DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME") or sys.exit("❌ Deployment name missing")

# ────────────────────────────────────────
# 2.  I/O paths
# ────────────────────────────────────────
SRC_DIR  = ROOT / "contracts_generic_txt"
OUT_DIR  = ROOT / "contracts_split_clauses"
PROMPT_FILE = ROOT / "clause_prompt.txt"

if not PROMPT_FILE.exists():
    sys.exit("❌ clause_prompt.txt not found next to the script")

PROMPT = PROMPT_FILE.read_text(encoding="utf-8")
OUT_DIR.mkdir(exist_ok=True)

# ────────────────────────────────────────
# 3.  OCR noise cleaner
#     (tweak patterns as needed)
# ────────────────────────────────────────
NOISE_RX = re.compile(
    r"""
    ^\s*(                             # start-of-line noise
        page\s+\d+                     # page 3
      | \d+\s*/\s*\d+\s*               # 2 / 14
      | .*confidential.*               # CONFIDENTIAL footer
      | msa.*\(\d{4}\)                 # "MSA (2021)"
    ).*$                               # whole line
    """,
    re.I | re.X,
)

def clean_ocr(text: str) -> str:
    """Drop header/footer noise lines from OCR output."""
    keep: List[str] = []
    for ln in text.splitlines():
        if NOISE_RX.match(ln):
            continue
        keep.append(ln)
    return "\n".join(keep)

# ────────────────────────────────────────
# 4.  Split contract into USER messages
# ────────────────────────────────────────
MAX_CHARS = 25_000   # stay < 32 KB / message

def build_messages(contract_txt: str) -> list[dict]:
    parts = [contract_txt[i : i + MAX_CHARS] for i in range(0, len(contract_txt), MAX_CHARS)]
    n = len(parts)
    msgs: list[dict] = [{"role": "system", "content": PROMPT}]
    for i, part in enumerate(parts, 1):
        header = f"CONTRACT PART {i}/{n}\n"
        msgs.append({"role": "user", "content": header + part})
    return msgs

# ────────────────────────────────────────
# 5.  Call Azure OpenAI once per contract
# ────────────────────────────────────────
def call_openai(contract_txt: str) -> str:
    rsp = client.chat.completions.create(
        model       = DEPLOYMENT,
        messages    = build_messages(contract_txt),
        temperature = 0.1,
        max_tokens  = 16_384,   # plenty of room for long annexes
    )
    return rsp.choices[0].message.content

# ────────────────────────────────────────
# 6.  Deduplicate overlapping clause blocks
# ────────────────────────────────────────
CLAUSE_START_RX = re.compile(r"^\[CLAUSE_START:.*?$", re.M)

def dedupe(text: str) -> str:
    seen: set[str] = set()
    lines_out: List[str] = []
    for line in text.splitlines():
        if line.startswith("[CLAUSE_START"):
            if line in seen:
                continue
            seen.add(line)
        lines_out.append(line)
    return "\n".join(lines_out)

# ────────────────────────────────────────
# 7.  Main loop
# ────────────────────────────────────────
txt_files = sorted(SRC_DIR.glob("*.txt"))
if not txt_files:
    sys.exit(f"No OCR files in {SRC_DIR}")

print(f"📄  Found {len(txt_files)} contract(s). Starting extraction…\n")

for txt_path in tqdm(txt_files, desc="🔍 Contracts", unit="file"):
    out_file = OUT_DIR / txt_path.with_suffix(".clauses.txt").name
    if out_file.exists():
        continue   # skip processed

    try:
        raw   = txt_path.read_text(encoding="utf-8", errors="ignore")
        clean = clean_ocr(raw)
        result = call_openai(clean)
        result = dedupe(result)
        out_file.write_text(result, encoding="utf-8")
    except Exception as exc:
        print(f"⚠️  {txt_path.name}: {exc}")

print("\n✅  Finished. Clause files saved to:", OUT_DIR.resolve())
