from pathlib import Path
import os, sys
from dotenv import load_dotenv
from tqdm import tqdm

from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient

# ─────────────── SETUP ───────────────
ROOT_DIR = Path(__file__).resolve().parent.parent        # one level up
load_dotenv(dotenv_path=ROOT_DIR / ".env")

ENDPOINT = os.getenv("DOCUMENTINTELLIGENCE_ENDPOINT")
KEY      = os.getenv("DOCUMENTINTELLIGENCE_API_KEY")
if not ENDPOINT or not KEY:
    sys.exit("❌  Missing DOCUMENTINTELLIGENCE_… values in .env")

client = DocumentIntelligenceClient(
    endpoint=ENDPOINT,
    credential=AzureKeyCredential(KEY)
)

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR / "contracts_generic_pdf"
DST_DIR = SCRIPT_DIR / "contracts_generic_txt"
DST_DIR.mkdir(exist_ok=True)

# ─────────────── OCR LOOP ───────────────
pdf_files = sorted(SRC_DIR.glob("*.pdf"))
if not pdf_files:
    sys.exit(f"❗  No PDFs found in {SRC_DIR.resolve()}")

print(f"🔍  Found {len(pdf_files)} PDF(s). Starting OCR...\n")

for pdf in tqdm(pdf_files, desc="📄  Processing PDFs", unit="file"):
    try:
        with pdf.open("rb") as fh:
            # ✨ FIX: pass stream as 2nd positional arg, remove wrong kw
            poller = client.begin_analyze_document(
                "prebuilt-read",                        # model_id
                fh,                                     # <-- body / stream
                content_type="application/pdf",
            )
        result = poller.result()

        pages_txt = [
            "\n".join(ln.content for ln in (p.lines or []))
            for p in (result.pages or [])
        ]
        raw_text = "\n\n".join(pages_txt)

        txt_out = DST_DIR / pdf.with_suffix(".txt").name
        txt_out.write_text(raw_text, encoding="utf-8")

    except Exception as e:
        print(f"⚠️  Failed to process {pdf.name}: {e}")

print("\n✅  All contracts processed and saved to:", DST_DIR.resolve())
