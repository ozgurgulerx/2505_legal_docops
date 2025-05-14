#!/usr/bin/env python
"""
generate_contracts.py  – 12-page MSA generator
• hierarchy + targeted friction + span logging
• NO Table-of-Contents page
• dynamic footer date pulled from cover
• 3-4 hard clauses per contract with higher noise
"""

# ── stdlib ───────────────────────────────────────────────────────────────
import os, math, time, random, re, json, logging, datetime as _dt
from pathlib import Path
from typing import List

# ── 3rd-party ────────────────────────────────────────────────────────────
from dotenv import load_dotenv
from tqdm import tqdm
import tiktoken
from openai import AzureOpenAI
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, PageBreak
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
        pdfmetrics.registerFont(TTFont("DejaVuSans", fp)); break

# ╔═ config & paths ═══════════════════════════════════════════════════════╗
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
PAGES_TOTAL     = 12
PAGES_PER_CALL  = 4               # → 3 calls per contract
OUT_TXT  = Path("contracts_txt"); OUT_TXT.mkdir(exist_ok=True)
OUT_PDF  = Path("contracts_pdf"); OUT_PDF.mkdir(exist_ok=True)
ANNOT_DIR= Path("annotations");   ANNOT_DIR.mkdir(exist_ok=True)

# ╔═ contamination settings ═══════════════════════════════════════════════╗
ZW_EVERY        = 250      # default
LIG_RATIO       = 0.008
ZW_EVERY_HARD   = 80       # inside hard clause
LIG_RATIO_HARD  = 0.03
HOMO_RATIO      = 0.01
HOMO_MAP        = str.maketrans({'a':'а','e':'е','i':'і','o':'о','p':'р','c':'с'})
TAIL_TOKENS     = 6_000

enc = tiktoken.encoding_for_model("gpt-4")

# ╔═ contamination helpers ════════════════════════════════════════════════╗
def contaminate(text:str, hard=False)->str:
    zw  = ZW_EVERY_HARD if hard else ZW_EVERY
    lig = LIG_RATIO_HARD if hard else LIG_RATIO
    out=[]
    for i,w in enumerate(text.split()):
        if random.random()<lig and any(d in w for d in ("fi","fl","ffi")):
            w=w.replace("ffi","ﬃ").replace("fi","ﬁ").replace("fl","ﬂ")
        elif w.islower() and random.random()<HOMO_RATIO:
            w=w.translate(HOMO_MAP)
        out.append(w)
        if (i+1)%zw==0: out.append(random.choice(['\u200b','\u200d','\u00ad']))
    return " ".join(out)

def strip_log_hard_tags(raw:str,fid:str):
    s_tag, e_tag = "«HARD_CLAUSE_START»", "«HARD_CLAUSE_END»"
    spans=[]; off=0
    while True:
        s=raw.find(s_tag,off)
        if s==-1: break
        e=raw.find(e_tag,s)
        spans.append({"label":"HardClause","start":s,"end":e-len(s_tag)})
        raw=raw[:s]+raw[s+len(s_tag):e]+raw[e+len(e_tag):]; off=s
    (ANNOT_DIR/f"{fid}.json").write_text(json.dumps({"file":fid,"spans":spans}))
    return raw, spans

# ╔═ numbering / x-ref fixers ═════════════════════════════════════════════╗
H1_RE = re.compile(r'^(\d+)\.\s', re.M)
def fix_duplicates(pgs:List[str]):
    seen=set(); dup=0; out=[]; mx=0
    for pg in pgs:
        lines=[]
        for ln in pg.splitlines():
            m=H1_RE.match(ln)
            if m:
                n=int(m[1])
                if n in seen:
                    dup+=1
                    if dup>1: ln=H1_RE.sub(f"{mx+1}. ",ln,1); n=mx+1
                seen.add(n); mx=max(mx,n)
            lines.append(ln)
        out.append("\n".join(lines))
    return out

def patch_refs(txt:str):
    refs=re.findall(r'§(\d+\.\d+)',txt)
    existing=set(re.findall(r'\((\d+\.\d+)[\)\s]',txt))
    kept=False
    for r in refs:
        if r not in existing:
            if not kept: kept=True; continue
            if existing: txt=txt.replace(f'§{r}',f'§{random.choice(list(existing))}')
    return txt

# ╔═ PDF builder ══════════════════════════════════════════════════════════╗
_STY=getSampleStyleSheet()
_STY.add(ParagraphStyle("H1",fontName="Helvetica-Bold",fontSize=12,spaceAfter=6,leading=14))
_STY.add(ParagraphStyle("H2",fontName="Helvetica-Bold",fontSize=11,leftIndent=18,spaceAfter=6,leading=13))
_STY.add(ParagraphStyle("BODY",fontName="Times-Roman",fontSize=11,leftIndent=36,spaceAfter=6,leading=13))
def style_for(line:str):
    if re.match(r'^\d+\.\s',line): return "H1"
    if re.match(r'^\d+\.\d+\s',line): return "H2"
    return "BODY"

def build_pdf(pages:List[str], pdf:Path):
    doc=SimpleDocTemplate(str(pdf),pagesize=letter,
        leftMargin=72,rightMargin=72,topMargin=72,bottomMargin=72)
    story=[]
    # cover
    story.append(Paragraph(pages[0],_STY["BODY"])); story.append(PageBreak())
    # body & annexes
    decoy=False
    for pg in pages[1:]:
        for ln in pg.splitlines():
            if ln.startswith("[Sig-Block-α]"):
                if decoy: continue
                decoy=True
            story.append(Paragraph(ln or "&nbsp;", _STY[style_for(ln)]))
        story.append(PageBreak())
    # dynamic footer date from cover
    m=re.search(r'Effective Date:\s*([A-Za-z0-9, \-]+)', pages[0])
    draft_date = m.group(1) if m else _dt.date.today().isoformat()
    def footer(cvs,doc):
        cvs.setFont("Times-Roman",8)
        cvs.drawCentredString(letter[0]/2,40,
           f"CONFIDEN\u00ADTIAL – DRAFT {draft_date} – Page {doc.page}")
    doc.build(story,onFirstPage=footer,onLaterPages=footer)

# ╔═ OpenAI helper ════════════════════════════════════════════════════════╗
SYSTEM={"role":"system",
        "content":"You are a senior commercial-contracts associate. "
                  "Draft dense but realistic MSAs; avoid tautology."}
PROMPT=Path(PROMPT_FILE).read_text()
def call(idx,start,end,tail):
    prompt=(f"{PROMPT}\n\n"
            f"# Emit pages {start}–{end} of {PAGES_TOTAL}. "
            "Insert delimiter <<<PAGE>>> between pages.")
    msgs=[SYSTEM,{"role":"assistant","content":tail},
                  {"role":"user","content":prompt}]
    max_tok=min(32768,int((end-start+1)*2000*1.3))
    r=client.chat.completions.create(model=DEPLOY,messages=msgs,
                                     max_tokens=max_tok,temperature=0.7)
    logging.info(f"[call {idx}] {r.usage.completion_tokens} tokens")
    return r.choices[0].message.content.strip()

# ╔═ main loop ════════════════════════════════════════════════════════════╗
CALLS=math.ceil(PAGES_TOTAL/PAGES_PER_CALL)
for c in tqdm(range(N_CONTRACTS),desc="contracts"):
    fid=f"contract_{c+1:02d}"
    if (OUT_PDF/f"{fid}.pdf").exists(): continue
    pages_raw=[]; tail=""
    for j in range(CALLS):
        s=j*PAGES_PER_CALL+1; e=min((j+1)*PAGES_PER_CALL,PAGES_TOTAL)
        chunk=call(j+1,s,e,tail); tail=chunk[-TAIL_TOKENS:]
        pages_raw+=chunk.split("<<<PAGE>>>"); time.sleep(0.8)
    # join → strip hard tags → span log
    joined="<<PB>>".join(p.strip() for p in pages_raw)
    joined, spans = strip_log_hard_tags(joined,fid)
    # contaminate page-by-page
    pages=[]; cursor=0
    for pg in joined.split("<<PB>>"):
        hard=any(s["start"]<=cursor<s["end"] for s in spans)
        pages.append(contaminate(pg, hard)); cursor+=len(pg)+4
    # numbering / refs
    pages=fix_duplicates(pages); full=patch_refs("[PAGE BREAK]".join(pages))
    # write outputs
    (OUT_TXT/f"{fid}.txt").write_text(full,encoding="utf-8")
    build_pdf(full.split("[PAGE BREAK]"), OUT_PDF/f"{fid}.pdf")

logging.info("✅ 12-page contracts generated, hard-clause spans logged.")
