import os, sys
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm
from openai import AzureOpenAI     # SDK ≥1.0

# ─────────── ENV ───────────
ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

client = AzureOpenAI(
    api_key        = os.getenv("AZURE_OPENAI_API_KEY"),
    api_version    = os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
)
DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME") or sys.exit("Missing dep name")

# ─────────── PATHS ───────────
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR / "contracts_generic_txt"
OUT_DIR = SCRIPT_DIR / "contracts_split_clauses"
PROMPT_PATH = SCRIPT_DIR / "clause_prompt.txt"          # <─ new
OUT_DIR.mkdir(exist_ok=True)

# load prompt once
PROMPT_TEMPLATE = PROMPT_PATH.read_text(encoding="utf-8")

def build_prompt(contract_txt: str) -> str:
    return PROMPT_TEMPLATE.replace("{{contract_text}}", contract_txt)

def call_llm(contract_txt: str) -> str:
    rsp = client.chat.completions.create(
        model      = DEPLOYMENT,
        messages   = [{"role": "user", "content": build_prompt(contract_txt)}],
        temperature= 0.2,
        max_tokens = 4096
    )
    return rsp.choices[0].message.content

# ─────────── MAIN ───────────
txt_files = sorted(SRC_DIR.glob("*.txt"))
if not txt_files:
    sys.exit("No .txt contracts found; run OCR first.")

print(f"📄 Found {len(txt_files)} contract(s)…\n")

for txt_path in tqdm(txt_files, desc="🧠 Extracting clauses", unit="file"):
    out_path = OUT_DIR / txt_path.with_suffix(".clauses.txt").name
    if out_path.exists():
        continue                    # skip already processed

    print(f"   📝 Now processing: {txt_path.name}")
    try:
        raw = txt_path.read_text(encoding="utf-8")
        clauses = call_llm(raw)
        out_path.write_text(clauses, encoding="utf-8")
    except Exception as e:
        print(f"⚠️  Failed on {txt_path.name}: {e}")

print("\n✅ Clause extraction complete.")
print("   ➤ Outputs:", OUT_DIR.resolve())
