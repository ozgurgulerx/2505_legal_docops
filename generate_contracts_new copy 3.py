#!/usr/bin/env python
"""
generate_contracts.py — hierarchy + light-friction (robust TOC)
───────────────────────────────────────────────────────────────
Creates 6-page customer MSAs that look professional but annoy rule-based
clause extractors.

Needs: python-dotenv, reportlab, pdf2image, tqdm, tiktoken, openai≥1.3
"""

# ── stdlib ────────────────────────────────────────────────────────────────
import os, math, time, random, re, logging, datetime as _dt
from pathlib import Path
from typing import List

# ── 3rd-party ─────────────────────────────────────────────────────────────
from dotenv import load_dotenv
from tqdm import tqdm
import tiktoken
from openai import AzureOpenAI
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (SimpleDocTemplate, Paragraph, PageBreak, Image,
                                Spacer)
# TOC can live in a sub-module or be absent on minimal builds
try:
    from reportlab.platypus.tableofcontents import TableOfContents
except ImportError:          # fall back: disable TOC
    TableOfContents = None

from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ╔═ logging ══════════════════════════════════════════════════════════════╗
logging.basicConfig(level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")

# ╔═ font registration ════════════════════════════════════════════════════╗
for fp in (
    "./fonts/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/Library/Fonts/DejaVuSans.ttf",
    "C:\\Windows\\Fonts\\DejaVuSans.ttf",
):
    if Path(fp).is_file():
        pdfmetrics.registerFont(TTFont("DejaVuSans", fp)); BASE_FONT="DejaVuSans"; break
else:
    logging.warning("DejaVuSans not found – fallback to Helvetica/Times")
    BASE_FONT="Helvetica"

# ╔═ configuration ════════════════════════════════════════════════════════╗
load_dotenv(override=True)
client = AzureOpenAI(
    api_key        = os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version    = os.getenv("AZURE_OPENAI_API_VERSION") or "2024-12-01-preview",
)
DEPLOY = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME") or "gpt-4.1"
logging.info(f"Using Azure OpenAI deployment «{DEPLOY}»")

PROMPT_FILE     = "prompt.txt"
N_CONTRACTS     = 10
PAGES_TOTAL     = 6
PAGES_PER_CALL  = 3
PNG_SNAPSHOTS   = True

OUT_TXT = Path("contracts_txt"); OUT_TXT.mkdir(exist_ok=True)
OUT_PDF = Path("contracts_pdf"); OUT_PDF.mkdir(exist_ok=True)
OUT_PNG = Path("contracts_png"); OUT_PNG.mkdir(exist_ok=True)

# contamination settings
ZW_CHARS=['\u200b','\u200d','\u00ad']; ZW_EVERY=170
LIG_RATIO=0.015; HOMO_RATIO=0.01
HOMO_MAP=str.maketrans({'a':'а','e':'е','i':'і','o':'о','p':'р','c':'с'})
TAIL_TOKENS=2_000

TABLE_IMAGES={
    "[TABLE_EX_A]":"fee_schedule.png",
    "[TABLE_EX_B]":"uptime_targets.png",
    "[TABLE_EX_C]":"escalation_matrix.png",
}

enc = tiktoken.encoding_for_model("gpt-4")

# ── contamination helpers ────────────────────────────────────────────────
def contaminate(txt:str)->str:
    out=[]
    for idx,w in enumerate(txt.split()):
        if random.random()<LIG_RATIO and any(d in w for d in ("fi","fl","ffi")):
            w=(w.replace("ffi","ﬃ").replace("fi","ﬁ").replace("fl","ﬂ"))
        elif w.islower() and random.random()<HOMO_RATIO:
            w=w.translate(HOMO_MAP)
        out.append(w)
        if (idx+1)%ZW_EVERY==0: out.append(random.choice(ZW_CHARS))
    return " ".join(out)

def last_tokens(txt:str,limit:int=TAIL_TOKENS)->str:
    ids=enc.encode(txt)
    return txt if len(ids)<=limit else enc.decode(ids[-limit:])

# ── heading fixer (allow one duplicate) ───────────────────────────────────
HEAD_RE=re.compile(r'^(\d+)\.\s')
def fix_duplicates(pages:List[str])->List[str]:
    seen=set(); dup=0; new=[]; maxnum=0
    for pg in pages:
        lines=[]
        for ln in pg.splitlines():
            m=HEAD_RE.match(ln.strip())
            if m:
                num=int(m.group(1))
                if num in seen:
                    dup+=1
                    if dup>1:
                        num=maxnum+1
                        ln=HEAD_RE.sub(f"{num}. ",ln,1)
                seen.add(num); maxnum=max(maxnum,num)
            lines.append(ln)
        new.append("\n".join(lines))
    return new

# ── x-ref patcher (keep one bad ref) ──────────────────────────────────────
def patch_refs(text:str)->str:
    refs=re.findall(r'§(\d+\.\d+)',text)
    existing=set(re.findall(r'\((\d+\.\d+)[\)\s]',text))
    kept=False
    for bad in refs:
        if bad not in existing:
            if not kept: kept=True; continue
            if existing: text=text.replace(f'§{bad}',f'§{random.choice(list(existing))}')
    return text

# ── ReportLab styles & PDF builder ───────────────────────────────────────
_STY=getSampleStyleSheet()
_STY.add(ParagraphStyle("H1",fontName="Helvetica-Bold",fontSize=12,spaceAfter=6,leading=14))
_STY.add(ParagraphStyle("H2",fontName="Helvetica-Bold",fontSize=11,leftIndent=18,spaceAfter=6,leading=13))
_STY.add(ParagraphStyle("BODY",fontName="Times-Roman",fontSize=11,leftIndent=36,spaceAfter=6,leading=13))

def style_key(line:str)->str:
    if re.match(r'^\d+\.\s',line): return "H1"
    if re.match(r'^\d+\.\d+\s',line): return "H2"
    return "BODY"

def build_pdf(pgs:List[str], pdf:Path):
    doc=SimpleDocTemplate(str(pdf),pagesize=letter,
        leftMargin=72,rightMargin=72,topMargin=72,bottomMargin=72)
    story=[]
    # cover
    story.append(Paragraph(pgs[0],_STY["BODY"])); story.append(PageBreak())
    # TOC page (only if class available)
    if TableOfContents:
        toc=TableOfContents(); toc.levelStyles=[_STY["H1"]]
        story+= [Paragraph("TABLE OF CONTENTS",_STY["H1"]), Spacer(1,12), toc, PageBreak()]
    # body & annexes
    decoy=False
    for pg in pgs[1:]:
        for raw in pg.splitlines():
            ln=raw.strip()
            if ln.startswith("[Sig-Block-α]"):
                if decoy: continue
                decoy=True
            story.append(Paragraph(ln or "&nbsp;", _STY[style_key(ln)]))
        story.append(PageBreak())
    today=_dt.date.today().isoformat()
    def footer(cv,doc):
        cv.setFont("Times-Roman",8)
        cv.drawCentredString(letter[0]/2,40,
            f"CONFIDEN\u00ADTIAL – DRAFT {today} – Page {doc.page}")
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    if PNG_SNAPSHOTS: raster(pdf)

def raster(pdf:Path):
    from pdf2image import convert_from_path
    for i,img in enumerate(convert_from_path(str(pdf)),1):
        img.save(OUT_PNG/f"{pdf.stem}_p{i:02d}.png","PNG")

# ── OpenAI prompt helper ─────────────────────────────────────────────────
MASTER_PROMPT=Path(PROMPT_FILE).read_text()
SYSTEM={"role":"system",
        "content":"You are a senior commercial-contracts associate. "
                  "Draft dense but realistic MSAs; avoid tautology."}

def chunk_call(i,start,end,tail):
    prompt=(f"{MASTER_PROMPT}\n\n"
            f"# Emit pages {start}–{end} of {PAGES_TOTAL}. "
            "Insert delimiter <<<PAGE>>> between pages.")
    msgs=[SYSTEM,{"role":"assistant","content":tail},
                  {"role":"user","content":prompt}]
    est=(end-start+1)*2000
    resp=client.chat.completions.create(
        model=DEPLOY,messages=msgs,
        max_tokens=min(32768,int(est*1.3)),temperature=0.7)
    logging.info(f"[Call {i}] {resp.usage.completion_tokens} tokens")
    return resp.choices[0].message.content.strip()

# ── main loop ────────────────────────────────────────────────────────────
CALLS = math.ceil(PAGES_TOTAL/PAGES_PER_CALL)

for c in tqdm(range(N_CONTRACTS),desc="Contracts"):
    txt=OUT_TXT/f"contract_{c+1:02d}.txt"
    pdf=OUT_PDF/f"contract_{c+1:02d}.pdf"
    if txt.exists() and pdf.exists(): continue
    pages=[]; tail=""
    for j in tqdm(range(CALLS),desc=f"API calls {c+1}",leave=False):
        s=j*PAGES_PER_CALL+1; e=min((j+1)*PAGES_PER_CALL,PAGES_TOTAL)
        chunk=chunk_call(j+1,s,e,tail); tail=last_tokens(chunk)
        pages+= [contaminate(p.strip()) for p in chunk.split("<<<PAGE>>>")]
        time.sleep(0.8)
    pages=fix_duplicates(pages)
    full=patch_refs("[PAGE BREAK]".join(pages))
    txt.write_text(full,encoding="utf-8")
    build_pdf(full.split("[PAGE BREAK]"), pdf)

logging.info("✅ Done: contracts in TXT, PDF" + (" & PNG" if PNG_SNAPSHOTS else ""))
