#!/usr/bin/env python3
"""
extract_clauses.py  ·  o4-mini edition
───────────────────────────────────────
Convert plain-text OCR contracts into clause-tagged files
using Azure OpenAI *reasoning* models via the Responses API.
"""
import os, re, sys
from pathlib import Path
from typing import List
from dotenv import load_dotenv
from tqdm import tqdm
from openai import AzureOpenAI

# ───────────────────────── 1. Azure env
ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT.parent / ".env")

client = AzureOpenAI(
    api_key        = os.getenv("AZURE_OPENAI_API_KEY"),
    api_version    = os.getenv("AZURE_REASONING_OPENAI_API_VERSION")
                  or sys.exit("❌ AZURE_REASONING_OPENAI_API_VERSION not set "
                              "(try 2025-04-01-preview)"),
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
                  or sys.exit("❌ AZURE_OPENAI_ENDPOINT not set"),
)

DEPLOYMENT = os.getenv("AZURE_OPENAI_REASONING_DEPLOYMENT_NAME") \
          or sys.exit("❌ AZURE_OPENAI_REASONING_DEPLOYMENT_NAME missing")

# ───────────────────────── 2. I/O
SRC_DIR     = ROOT / "contracts_generic_txt"
OUT_DIR     = ROOT / "contracts_split_clauses"
PROMPT_FILE = ROOT / "clause_prompt.txt"
PROMPT      = PROMPT_FILE.read_text(encoding="utf-8") \
              if PROMPT_FILE.exists() else sys.exit("❌ clause_prompt.txt missing")
OUT_DIR.mkdir(exist_ok=True)

# ───────────────────────── 3. OCR noise cleaner (unchanged)
NOISE_RX = re.compile(r"""^\s*(page\s+\d+(\s+of\s+\d+)?|\d+\s*/\s*\d+|.*confidential.*|
                           msa.*\(\d{4}\)|celonis\s+proprietary.*)\s*$""",
                      re.I | re.X)
def clean_ocr(txt:str)->str: return "\n".join(ln for ln in txt.splitlines()
                                              if not NOISE_RX.match(ln))

# ───────────────────────── 4. Messages builder
def build_input(contract_txt:str)->list[dict]:
    """Return a list for the `input=` parameter of Responses API."""
    # reasoning models ignore temperature; one big user message is fine.
    return [
        {"role": "developer", "content": PROMPT},
        {"role": "user",      "content": contract_txt},
    ]

# ───────────────────────── 5. Single call — Responses API
MAX_OUTPUT_TOKENS = 100000          # budget for reasoning + visible output

def call_openai(contract_txt:str)->str:
    rsp = client.responses.create(               # ⇦ Responses API
        model      = DEPLOYMENT,
        input      = build_input(contract_txt),
        reasoning  = {"effort": "low"},       # default is medium; explicit for clarity
        max_output_tokens = MAX_OUTPUT_TOKENS,   # reasoning + visible tokens
    )
    if rsp.status == "incomplete":
        raise RuntimeError(f"Incomplete due to {rsp.incomplete_details}")
    content = rsp.output_text
    if not content:
        raise RuntimeError("Model returned empty output")
    return content

# ───────────────────────── 6. Dedupe adjacent duplicates (unchanged)
def dedupe(text:str)->str:
    out, prev = [], ""
    for ln in text.splitlines():
        if ln.startswith("[CLAUSE_START") and ln == prev:
            continue
        out.append(ln); prev = ln
    return "\n".join(out)

# ───────────────────────── 7. Main loop
txt_files = sorted(SRC_DIR.glob("*.txt")) or sys.exit(f"No OCR files in {SRC_DIR}")
print(f"📄  Found {len(txt_files)} contract(s). Starting extraction…\n")

for txt_path in tqdm(txt_files, desc="🔍 Contracts", unit="file"):
    out_path = OUT_DIR / txt_path.with_suffix(".clauses.txt").name
    if out_path.exists():
        continue
    try:
        raw    = txt_path.read_text(encoding="utf-8", errors="ignore")
        clean  = clean_ocr(raw)
        result = dedupe(call_openai(clean))
        out_path.write_text(result, encoding="utf-8")
    except Exception as exc:
        print(f"⚠️  {txt_path.name}: {exc}")

print("\n✅  Finished. Clause files saved to:", OUT_DIR.resolve())
