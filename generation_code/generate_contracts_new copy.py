#!/usr/bin/env python
"""
generate_contracts.py  (v6 – DejaVu font fix + 15-page contracts)

• Registers DejaVuSans TTF so ReportLab stops raising “Can't map” errors.
• Contracts are now 15 pages (3-page chunks → 5 calls per contract).
• All other logic unchanged: rolling context, realistic contamination,
  image-tables, hidden AcroForm fields, cross-ref patch, logging.
"""

# ── Std-lib ────────────────────────────────────────────────────────────────
import os, math, time, random, re, logging
from pathlib import Path

# ── Third-party ────────────────────────────────────────────────────────────
from dotenv import load_dotenv
from tqdm import tqdm
import tiktoken
from openai import AzureOpenAI

from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, PageBreak, Image
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ╔══════════════════════════════════════════════════════════════════════════╗
# LOGGING
# ╚══════════════════════════════════════════════════════════════════════════╝
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s",
                    datefmt="%H:%M:%S")

# ╔══════════════════════════════════════════════════════════════════════════╗
# REGISTER DejaVuSans TTF (adjust path if necessary)
# ╚══════════════════════════════════════════════════════════════════════════╝
FONT_PATHS = [
    "./fonts/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/Library/Fonts/DejaVuSans.ttf",
    "C:\\Windows\\Fonts\\DejaVuSans.ttf",
]
for fp in FONT_PATHS:
    if Path(fp).exists():
        pdfmetrics.registerFont(TTFont("DejaVuSans", fp))
        break
else:
    logging.warning("DejaVuSans.ttf not found – falling back to Helvetica")
    DEJAVU_AVAILABLE = False
else:
    DEJAVU_AVAILABLE = True

# ╔══════════════════════════════════════════════════════════════════════════╗
# CONFIG
# ╚══════════════════════════════════════════════════════════════════════════╝
load_dotenv(override=True)
API_KEY       = os.getenv("AZURE_OPENAI_API_KEY")
ENDPOINT      = os.getenv("AZURE_OPENAI_ENDPOINT")
API_VERSION   = os.getenv("AZURE_OPENAI_API_VERSION") or "2024-12-01-preview"
DEPLOYMENT    = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME") or "gpt-4.1"
logging.info(f"Using deployment: {DEPLOYMENT}")

PROMPT_FILE   = "prompt_real.txt"
N_CONTRACTS   = 10
PAGES_TOTAL   = 15                      # ← now 15 pages/contract
PAGES_PER_CALL = 3
RASTERISE     = True

OUT_TXT = Path("contracts_txt"); OUT_TXT.mkdir(exist_ok=True)
OUT_PDF = Path("contracts_pdf"); OUT_PDF.mkdir(exist_ok=True)
OUT_PNG = Path("contracts_png"); OUT_PNG.mkdir(exist_ok=True)

ZW_CHARS  = ['\u200b', '\u200d', '\u00ad']
HOMO_MAP  = str.maketrans({'a':'а','e':'е','i':'і','o':'о','p':'р','c':'с'})
TAIL_TOKENS = 2_000

TABLE_IMAGES = {
    "[TABLE_EX_A]": "fee_schedule.png",
    "[TABLE_EX_B]": "uptime_targets.png",
    "[TABLE_EX_C]": "escalation_matrix.png",
}

enc = tiktoken.encoding_for_model("gpt-4")

# ╔══════════════════════════════════════════════════════════════════════════╗
# HELPERS
# ╚══════════════════════════════════════════════════════════════════════════╝
def contaminate(text: str,
                zw_every: int = 60,
                homoglyph_ratio: float = 0.01,
                lig_ratio: float = 0.03) -> str:
    out = []
    for idx, w in enumerate(text.split()):
        lw = w.lower()
        if random.random() < lig_ratio and lw in ("fi", "fl", "ffi"):
            w = (w.replace("fi", "ﬁ")
                   .replace("fl", "ﬂ")
                   .replace("ffi", "ﬃ"))
        elif w.islower() and random.random() < homoglyph_ratio:
            w = w.translate(HOMO_MAP)
        out.append(w)
        if (idx + 1) % zw_every == 0:
            out.append(random.choice(ZW_CHARS))
    return " ".join(out)

def last_tokens(txt: str, limit: int = TAIL_TOKENS) -> str:
    ids = enc.encode(txt)
    return txt if len(ids) <= limit else enc.decode(ids[-limit:])

def patch_crossrefs(text: str) -> str:
    refs = re.findall(r'§(\d+\.\d+)', text)
    existing = set(re.findall(r'(\d+\.\d+)[\)]', text))
    for bad in refs:
        if bad not in existing and existing:
            good = random.choice(list(existing))
            text = text.replace(f'§{bad}', f'§{good}')
    return text

# ╔══════════════════════════════════════════════════════════════════════════╗
# PDF BUILD
# ╚══════════════════════════════════════════════════════════════════════════╝
def build_pdf(text: str, pdf_path: Path):
    style = ParagraphStyle(
        "Flat",
        fontName="DejaVuSans" if DEJAVU_AVAILABLE else "Helvetica",
        fontSize=7,
        leading=7.5,
        alignment=TA_LEFT,
    )
    doc   = SimpleDocTemplate(str(pdf_path), pagesize=letter)
    story = []
    for raw in text.split("[PAGE BREAK]"):
        for line in raw.splitlines():
            tok = line.strip()
            if tok in TABLE_IMAGES and Path(TABLE_IMAGES[tok]).exists():
                story.append(Image(TABLE_IMAGES[tok], width=480, height=120))
            else:
                story.append(Paragraph(line.replace("\n", " "), style))
        story.append(PageBreak())
    doc.build(story, onFirstPage=_add_hidden, onLaterPages=_add_hidden)

def _add_hidden(canvas, doc):
    # invisible timestamp field
    canvas.acroForm.textfield(name=f"sig_ts_{doc.page}",
                              x=50, y=60, width=180, height=12,
                              borderWidth=0, fillColor=None,
                              textColor=None, forceBorder=False)

def pdf_to_pngs(pdf_path: Path):
    from pdf2image import convert_from_path
    for i, img in enumerate(convert_from_path(str(pdf_path)), 1):
        img.save(OUT_PNG / f"{pdf_path.stem}_p{i:02d}.png", "PNG")

# ╔══════════════════════════════════════════════════════════════════════════╗
# OPENAI
# ╚══════════════════════════════════════════════════════════════════════════╝
client = AzureOpenAI(api_key=API_KEY,
                     azure_endpoint=ENDPOINT,
                     api_version=API_VERSION)

MASTER_PROMPT = Path(PROMPT_FILE).read_text(encoding="utf-8")
SYSTEM_MSG = {"role": "system",
              "content": "You are a senior commercial-contracts associate. "
                         "Produce ultra-dense but realistic MSAs; avoid tautology."}

def generate_chunk(call_no: int, start_pg: int, end_pg: int, tail: str) -> str:
    user_prompt = (
        f"{MASTER_PROMPT}\n\n"
        f"# Emit pages {start_pg}–{end_pg} (inclusive) of {PAGES_TOTAL}. "
        "Insert delimiter <<<PAGE>>> between pages."
    )
    messages = [SYSTEM_MSG,
                {"role": "assistant", "content": tail},
                {"role": "user", "content": user_prompt}]

    est_tokens = (end_pg - start_pg + 1) * 2000
    max_tokens = min(32768, int(est_tokens * 1.3))
    logging.info(f"[Call {call_no}] pages {start_pg}-{end_pg} "
                 f"tail={len(enc.encode(tail))}t  max={max_tokens}")

    resp = client.chat.completions.create(
        model=DEPLOYMENT, messages=messages,
        max_tokens=max_tokens, temperature=0.7)
    logging.info(f"[Call {call_no}] ✅ {resp.usage.completion_tokens} tokens")
    return resp.choices[0].message.content.strip()

# ╔══════════════════════════════════════════════════════════════════════════╗
# MAIN
# ╚══════════════════════════════════════════════════════════════════════════╝
calls_per_contract = math.ceil(PAGES_TOTAL / PAGES_PER_CALL)

for c_idx in tqdm(range(N_CONTRACTS), desc="Contracts"):
    txt = OUT_TXT / f"contract_{c_idx+1:02d}.txt"
    pdf = OUT_PDF / f"contract_{c_idx+1:02d}.pdf"
    if txt.exists() and pdf.exists():
        continue

    pages, tail = [], ""
    bar = tqdm(range(calls_per_contract),
               desc=f"Contract {c_idx+1} calls", leave=False)
    for i in bar:
        s, e = i * PAGES_PER_CALL + 1, min((i+1)*PAGES_PER_CALL, PAGES_TOTAL)
        chunk = generate_chunk(i+1, s, e, tail)
        tail = last_tokens(chunk)
        pages.extend(contaminate(pg.strip()) for pg in chunk.split("<<<PAGE>>>"))
        time.sleep(0.8)

    full = patch_crossrefs("[PAGE BREAK]".join(pages))
    txt.write_text(full, encoding="utf-8")
    build_pdf(full, pdf)
    if RASTERISE:
        pdf_to_pngs(pdf)

logging.info("✅ Completed. TXT, PDF (and PNG) generated.")
