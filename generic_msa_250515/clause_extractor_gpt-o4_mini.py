#!/usr/bin/env python3
"""
extract_clauses.py
──────────────────
Convert plain-text OCR contracts into clause-tagged files with Azure OpenAI.
"""

import os, re, sys
from pathlib import Path
from typing import List
from dotenv import load_dotenv
from tqdm import tqdm
from openai import AzureOpenAI

# ────────────────────────────────────────
# 1.  Azure / environment
# ────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT.parent / ".env")

client = AzureOpenAI(
    api_key        = os.getenv("AZURE_OPENAI_API_KEY"),
    api_version    = os.getenv("AZURE_OPENAI_API_VERSION")                # ► ensure 2024-05-15-preview or later
                  or sys.exit("❌ AZURE_OPENAI_API_VERSION not set"),
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
                  or sys.exit("❌ AZURE_OPENAI_ENDPOINT not set"),
)

DEPLOYMENT = (
    os.getenv("AZURE_OPENAI_REASONING_DEPLOYMENT_NAME")
    or sys.exit("❌ AZURE_OPENAI_REASONING_DEPLOYMENT_NAME missing")
)

# ────────────────────────────────────────
# 2.  I/O paths
# ────────────────────────────────────────
SRC_DIR      = ROOT / "contracts_generic_txt"
OUT_DIR      = ROOT / "contracts_split_clauses"
PROMPT_FILE  = ROOT / "clause_prompt.txt"

if not PROMPT_FILE.exists():
    sys.exit("❌ clause_prompt.txt not found next to the script")

PROMPT = PROMPT_FILE.read_text(encoding="utf-8")
OUT_DIR.mkdir(exist_ok=True)

# ────────────────────────────────────────
# 3.  OCR noise cleaner
# ────────────────────────────────────────
NOISE_RX = re.compile(
    r"""
    ^\s*(
          page\s+\d+\s*(of\s+\d+)?       # Page 6  |  Page 6 of 32        – added
        | \d+\s*/\s*\d+\s*               # 2 / 14
        | .*confidential.*               # CONFIDENTIAL footer
        | msa.*\(\d{4}\)                 # MSA (2021)
        | celonis\s+proprietary.*        # CELONIS PROPRIETARY …         – added
    )\s*$
    """,
    re.I | re.X,
)

def clean_ocr(text: str) -> str:
    """Drop header/footer noise lines from OCR output."""
    return "\n".join(
        ln for ln in text.splitlines() if not NOISE_RX.match(ln)
    )

# ────────────────────────────────────────
# 4.  Split contract into USER messages
# ────────────────────────────────────────
MAX_CHARS_IN   = 25_000   # < 32 KB / message keeps latency low
MAX_TOKENS_OUT = 8_000    # empirical worst-case for 400-page MSA  – ► changed

def build_messages(contract_txt: str) -> list[dict]:
    parts = [
        contract_txt[i : i + MAX_CHARS_IN]
        for i in range(0, len(contract_txt), MAX_CHARS_IN)
    ]
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
        max_tokens  = MAX_TOKENS_OUT,         # ► changed
    )
    return rsp.choices[0].message.content or ""

# ────────────────────────────────────────
# 6.  Deduplicate *adjacent* duplicate starts
# ────────────────────────────────────────
def dedupe(text: str) -> str:
    lines = text.splitlines()
    if not lines:
        return text
    out: List[str] = [lines[0]]
    for ln in lines[1:]:
        if ln.startswith("[CLAUSE_START") and ln == out[-1]:
            continue                     # skip only *consecutive* duplicates
        out.append(ln)
    return "\n".join(out)

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
        continue  # skip processed

    try:
        raw    = txt_path.read_text(encoding="utf-8", errors="ignore")
        clean  = clean_ocr(raw)
        result = call_openai(clean)
        result = dedupe(result)
        out_file.write_text(result, encoding="utf-8")
    except Exception as exc:
        print(f"⚠️  {txt_path.name}: {exc}")

print("\n✅  Finished. Clause files saved to:", OUT_DIR.resolve())
