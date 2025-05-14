import os, sys
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm
from openai import AzureOpenAI          # NEW import

# ─────────── ENV ───────────
ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

api_key      = os.getenv("AZURE_OPENAI_API_KEY")
api_version  = os.getenv("AZURE_OPENAI_API_VERSION")
azure_ep     = os.getenv("AZURE_OPENAI_ENDPOINT")
DEPLOYMENT   = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

if not all([api_key, api_version, azure_ep, DEPLOYMENT]):
    sys.exit("❌  Missing Azure OpenAI vars in .env")

client = AzureOpenAI(                   # NEW client style
    api_key       = api_key,
    api_version   = api_version,
    azure_endpoint= azure_ep
)

# ─────────── PATHS ───────────
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR / "contracts_generic_txt"
OUT_DIR = SCRIPT_DIR / "contracts_split_clauses"
OUT_DIR.mkdir(exist_ok=True)

# ─────────── PROMPT ───────────
PROMPT_TEMPLATE = """You are a legal document analyst.

Split the contract below into distinct clauses, using:
[CLAUSE_START: <title>]
<clause text>
[CLAUSE_END]

CONTRACT:
---------------------
{text}
"""

def call_llm(text: str) -> str:
    prompt = PROMPT_TEMPLATE.format(text=text[:12000])
    rsp = client.chat.completions.create(              # NEW call
        model     = DEPLOYMENT,
        messages  = [{"role":"user","content":prompt}],
        temperature = 0.2,
    )
    return rsp.choices[0].message.content

# ─────────── MAIN ───────────
txt_files = sorted(SRC_DIR.glob("*.txt"))
if not txt_files:
    sys.exit("❗ No .txt files found. Run OCR phase first.")

print(f"📄 Found {len(txt_files)} contract(s) to process...\n")

for txt_path in tqdm(txt_files, desc="🧠 Extracting clauses", unit="file"):
    out_path = OUT_DIR / txt_path.with_suffix(".clauses.txt").name
    if out_path.exists():
        continue

    print(f"   📝 Now processing: {txt_path.name}")
    try:
        raw = txt_path.read_text(encoding="utf-8")
        clauses = call_llm(raw)
        out_path.write_text(clauses, encoding="utf-8")
    except Exception as e:
        print(f"⚠️  Failed on {txt_path.name}: {e}")

print("\n✅ Clause extraction finished.")
print("   ➤ Outputs:", OUT_DIR.resolve())
