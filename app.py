# app.py
"""
StudyBuddy.ai — Ultra Premium (flan-t5-base local + optional HF Router Inference API)
Features:
- Upload PDFs, extract text (pdfplumber)
- Chunking + embeddings (sentence-transformers)
- FAISS index for retrieval + reranking
- Local generator (flan-t5-base) with deterministic beams + fallbacks
- Optional remote generator using Hugging Face Router Inference API (fast, production-ready)
- Robust MCQ generation (one-by-one + retries) with strict JSON output
- Auto-retry for truncated Q&A answers
- Ultra-premium UI + debug toggles

To use Hugging Face Router API: set HF_API_KEY env var or paste token into app UI when enabling remote mode.
"""

import os
import json
import re
import time
from typing import List, Dict, Any, Tuple

import streamlit as st
import numpy as np
import pandas as pd
import pdfplumber
from sentence_transformers import SentenceTransformer
import faiss
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import requests

# ------------------------
# CONFIG
# ------------------------
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GEN_MODEL_NAME_LOCAL = "google/flan-t5-base"   # local model
GEN_MODEL_NAME_HF = "google/flan-t5-base"      # remote model name on HF Router
INDEX_DIM = 384
TOP_K_DEFAULT = 6
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 300
PROGRESS_FILE = os.path.join("data", "user_progress.json")
os.makedirs("data", exist_ok=True)

# ------------------------
# CACHED MODEL LOADERS
# ------------------------
@st.cache_resource(show_spinner="🔁 Loading embedding model...")
def load_embedder():
    return SentenceTransformer(EMBED_MODEL_NAME)

@st.cache_resource(show_spinner="🤖 Loading local generator model (flan-t5-base)...")
def load_generator_local():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_NAME_LOCAL, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL_NAME_LOCAL)
    model.to(device)
    return tokenizer, model, device

# ------------------------
# UTIL: PDF + chunking + FAISS
# ------------------------
def read_pdf_bytes(uploaded_file) -> str:
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            texts = [page.extract_text() or "" for page in pdf.pages]
        return "\n".join(texts)
    except Exception:
        return ""

def clean_text(text: str) -> str:
    text = text.replace("\u00a0", " ").replace("\t", " ")
    return " ".join(text.split())

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    text = clean_text(text)
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end]
        # end near sentence boundary if possible
        if end < n:
            pos = text.rfind(".", start, end)
            if pos != -1 and pos > start + int(0.4 * chunk_size):
                end = pos + 1
                chunk = text[start:end]
        chunks.append(chunk.strip())
        start = max(end - overlap, end) if end < n else end
        if start >= n:
            break
    return [c for c in chunks if len(c) > 50]

def build_faiss_index(embedder: SentenceTransformer, docs: List[Dict[str, Any]]):
    """
    Build FAISS index and return (index, embeddings)
    Also stores embeddings in session state for reranking.
    """
    texts = [d["text"] for d in docs]
    embeddings = embedder.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings

def retrieve_chunks(query: str, embedder: SentenceTransformer, index, docs: List[Dict[str, Any]],
                    top_k: int = TOP_K_DEFAULT, re_rank_top: int = 24) -> List[Dict[str, Any]]:
    """
    1) Use FAISS to get top `re_rank_top` candidates
    2) Re-rank them by exact cosine with stored embeddings for stability
    3) Return top_k final results
    """
    if index is None:
        return []

    q_emb = embedder.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    rr = max(top_k, re_rank_top)
    scores, idxs = index.search(q_emb, rr)
    idxs = idxs[0]
    # Build candidate list
    cand = []
    for i in idxs:
        if i < 0 or i >= len(docs):
            continue
        cand.append(i)

    # Re-rank using stored embeddings if available
    emb_matrix = st.session_state.get("embeddings", None)
    ranked = []
    if emb_matrix is not None:
        cand_embs = emb_matrix[cand]  # shape (n, dim)
        dot = np.dot(q_emb, cand_embs.T)[0]  # similarity scores
        ranked = sorted(zip(cand, dot.tolist()), key=lambda x: x[1], reverse=True)
    else:
        # fallback: compute individually
        temp = []
        for i in cand:
            emb = embedder.encode([docs[i]["text"]], convert_to_numpy=True)
            faiss.normalize_L2(emb)
            score = float(np.dot(q_emb, emb.T))
            temp.append((i, score))
        ranked = sorted(temp, key=lambda x: x[1], reverse=True)

    results = []
    added = set()
    for i, score in ranked[:top_k]:
        key = (docs[i]["doc_id"], docs[i]["chunk_id"])
        if key in added:
            continue
        added.add(key)
        item = docs[i].copy()
        item["score"] = float(score)
        results.append(item)
    return results

# ------------------------
# GENERATION: local & HF Router
# ------------------------
def run_generator_local(prompt: str, tokenizer, model, device,
                        max_tokens: int = 256, num_beams: int = 4, do_sample: bool = False,
                        temperature: float = 0.0, retries: int = 1) -> str:
    """
    Local generation with beams + fallback sampling retry logic.
    """
    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    except Exception:
        # Tokenizer errors -> return empty
        return ""
    gen_kwargs = dict(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=max_tokens,
        num_beams=num_beams,
        do_sample=do_sample,
        temperature=temperature,
        early_stopping=True,
        no_repeat_ngram_size=3,
    )
    try:
        with torch.no_grad():
            out = model.generate(**gen_kwargs)
        text = tokenizer.decode(out[0], skip_special_tokens=True).strip()
    except Exception:
        # fallback: sampling
        try:
            with torch.no_grad():
                out = model.generate(**{**gen_kwargs, "do_sample": True, "top_p": 0.95, "temperature": 0.9})
            text = tokenizer.decode(out[0], skip_special_tokens=True).strip()
        except Exception:
            text = ""

    # Sanity check: if too short, retry with sampling/longer tokens
    if retries > 0:
        words = text.split()
        if len(words) < 6:
            try:
                with torch.no_grad():
                    out = model.generate(**{**gen_kwargs, "do_sample": True, "top_p": 0.95, "temperature": 0.85, "max_new_tokens": max(512, max_tokens*2)})
                text2 = tokenizer.decode(out[0], skip_special_tokens=True).strip()
                if len(text2.split()) > len(words):
                    text = text2
            except Exception:
                pass
    return text

def run_generator_hf(prompt: str, hf_api_key: str, model_name: str = GEN_MODEL_NAME_HF, max_tokens: int = 256,
                     parameters: dict = None) -> str:
    """
    Use Hugging Face Router Inference API.
    Requires HF_API_KEY.
    """
    if not hf_api_key:
        return ""

    # New Router endpoint (required)
    url = f"https://router.huggingface.co/hf-inference/models/{model_name}"

    headers = {
        "Authorization": f"Bearer {hf_api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "inputs": prompt,
        "parameters": parameters or {
            "max_new_tokens": max_tokens,
            "temperature": 0.0,
            "top_p": 0.95,
            "do_sample": False,
            "num_beams": 4
        }
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=120)
        data = resp.json()

        # HF Router may return various shapes; handle common cases
        if isinstance(data, dict) and "generated_text" in data:
            return data["generated_text"].strip()
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict) and "generated_text" in data[0]:
            return data[0]["generated_text"].strip()

        # return structured error if unexpected
        return f"[HF ERROR {resp.status_code}] {data}"

    except Exception as e:
        return f"[HF EXCEPTION] {str(e)}"

def run_generator(prompt: str, mode: str, tokenizer=None, model=None, device=None, hf_api_key: str = None,
                  max_tokens: int = 256, num_beams: int = 4, do_sample: bool = False, temperature: float = 0.0, retries: int = 1):
    """
    Wrapper: choose remote HF Router inference or local generation.
    mode: 'hf' or 'local'
    """
    if mode == "hf":
        return run_generator_hf(prompt, hf_api_key=hf_api_key, model_name=GEN_MODEL_NAME_HF, max_tokens=max_tokens, parameters={"max_new_tokens": max_tokens, "temperature": temperature, "num_beams": num_beams, "do_sample": do_sample})
    else:
        return run_generator_local(prompt, tokenizer, model, device, max_tokens=max_tokens, num_beams=num_beams, do_sample=do_sample, temperature=temperature, retries=retries)

# ------------------------
# MCQ generation (one-by-one + retries)
# ------------------------
def generate_mcqs_from_chunk(chunk: str, num_questions: int, gen_mode: str, tokenizer=None, model=None, device=None, hf_api_key: str = None) -> List[Dict[str, Any]]:
    """
    Generate MCQs one-by-one and validate JSON. Adaptive: uses local or HF generator depending on gen_mode.
    """
    def gen_one_question(material: str, q_index: int) -> Dict[str, Any]:
        single_example = """
{
  "question": "What is the main goal of hypothesis testing?",
  "options": ["Estimate a mean", "Test a claim about a population", "Visualize data", "Create a survey"],
  "answer_index": 1,
  "explanation": "Hypothesis testing uses sample data to decide whether to reject a population-level claim."
}
"""
        system_prompt = (
            "You are an expert tutor. Produce ONE multiple-choice question in VALID JSON ONLY. "
            "The JSON must be an object with keys: question, options (array of 4), answer_index (0-based), explanation."
        )
        material_short = material[:2200]
        prompt = system_prompt + "\n\nEXAMPLE:\n" + single_example + "\n\nMATERIAL:\n" + material_short + f"\n\nNow produce question number {q_index+1} as a JSON object."

        attempts = 0
        while attempts < 3:
            raw = run_generator(prompt, gen_mode, tokenizer=tokenizer, model=model, device=device, hf_api_key=hf_api_key,
                                max_tokens=320, num_beams=6, do_sample=False, temperature=0.0, retries=1)
            # extract first {...}
            m = re.search(r"(\{.*?\})", raw, flags=re.S)
            json_text = m.group(1) if m else raw
            try:
                obj = json.loads(json_text)
            except Exception:
                import ast
                try:
                    obj = ast.literal_eval(json_text)
                except Exception:
                    obj = None
            if obj:
                try:
                    q = str(obj["question"]).strip()
                    options = [str(x).strip() for x in obj["options"]][:4]
                    if len(options) < 4:
                        raise ValueError("not 4 options")
                    ans = int(obj["answer_index"])
                    if not (0 <= ans < 4):
                        raise ValueError("answer index out of range")
                    exp = str(obj.get("explanation", "")).strip()
                    return {"question": q, "options": options, "answer_index": ans, "explanation": exp}
                except Exception:
                    obj = None
            # fallback: try sampling run
            attempts += 1
            prompt = prompt + "\n\nIf your previous JSON was invalid, try again and ensure valid JSON."
        return None

    material = chunk.strip()
    mcqs = []
    for i in range(num_questions):
        qobj = gen_one_question(material, i)
        if qobj:
            mcqs.append(qobj)
        else:
            break
    return mcqs

# ------------------------
# Q&A prompt + auto-extend retry
# ------------------------
def answer_from_context(question: str, retrieved_chunks: List[Dict[str, Any]],
                        gen_mode: str, tokenizer=None, model=None, device=None, hf_api_key: str = None,
                        max_answer_tokens: int = 420) -> Tuple[str, List[Dict[str, Any]], str]:
    """
    Build prompt to require citations. If truncated/short, auto-retry with more tokens and sampling.
    Returns (answer_text, citation_list, raw_output).
    """
    context_parts = []
    for r in retrieved_chunks[:10]:
        ctx = f"[{r['doc_id']}::chunk_{r['chunk_id']}]\n{r['text'][:1000]}"
        context_parts.append(ctx)
    context_text = "\n\n".join(context_parts)

    prompt = (
        "You are a precise, concise tutor. Answer the QUESTION USING ONLY the STUDY MATERIAL below. "
        "If the material does not contain the answer, respond exactly: "
        "\"I don't know based on the provided material.\" "
        "Provide: (A) Short answer (1-3 sentences), (B) brief rationale, (C) a 'CITATIONS:' line listing sources as [doc::chunk_id]. "
        "Do NOT hallucinate."
        f"\n\nSTUDY MATERIAL:\n{context_text}\n\nQUESTION: {question}\n\nAnswer now."
    )

    raw = run_generator(prompt, gen_mode, tokenizer=tokenizer, model=model, device=device, hf_api_key=hf_api_key,
                        max_tokens=max_answer_tokens, num_beams=4, do_sample=False, temperature=0.0, retries=1)

    def looks_truncated(s: str) -> bool:
        s = s.strip()
        if not s:
            return True
        if len(s.split()) < 20 and not re.search(r"[.?!]$", s):
            return True
        # detect mid-phrase endings (common)
        trailing = s.split()[-6:] if len(s.split()) >= 6 else s.split()
        trail_join = " ".join(trailing).lower()
        if re.search(r"\b(each layer|can be thought of|such as|for example|i\.e\.|e\.g\.)\b", trail_join):
            return True
        return False

    if looks_truncated(raw):
        # retry with expanded request
        raw_retry = run_generator(prompt + "\n\nIf the previous answer looks short or incomplete, please expand the answer in 2 short paragraphs and include the CITATIONS line again.", gen_mode, tokenizer=tokenizer, model=model, device=device, hf_api_key=hf_api_key, max_tokens=max(600, max_answer_tokens*2), num_beams=3, do_sample=True, temperature=0.7, retries=0)
        if len(raw_retry.split()) > len(raw.split()):
            raw = raw_retry

    # extract citations
    cit_pattern = re.compile(r"\[([^\]]+::chunk_[0-9]+)\]")
    citations = cit_pattern.findall(raw)
    seen = set()
    cit_list = []
    for c in citations:
        if c not in seen:
            seen.add(c)
            doc, chunk_id = c.split("::chunk_")
            try:
                cit_list.append({"doc_id": doc, "chunk_id": int(chunk_id)})
            except Exception:
                continue

    answer_text = raw.strip()
    if len(answer_text) > 4000:
        answer_text = answer_text[:4000] + "..."
    return answer_text, cit_list, raw

# ------------------------
# Progress tracking
# ------------------------
def load_progress() -> Dict[str, Any]:
    if not os.path.exists(PROGRESS_FILE):
        return {}
    try:
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_progress(data: Dict[str, Any]):
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def update_mastery(progress: Dict[str, Any], topic_id: str, correct: bool):
    if topic_id not in progress:
        progress[topic_id] = {"correct": 0, "total": 0}
    progress[topic_id]["total"] += 1
    if correct:
        progress[topic_id]["correct"] += 1
    save_progress(progress)

def compute_mastery_score(stats: Dict[str, int]) -> float:
    if stats["total"] == 0:
        return 0.0
    return (stats["correct"] + 1) / (stats["total"] + 2)

# ------------------------
# Ultra-premium CSS + UI helpers
# ------------------------
def set_ultra_premium_theme():
    st.set_page_config(page_title="StudyBuddy.ai — Ultra Premium", page_icon="📘", layout="wide")
    css = """
    <style>
    :root{
        --bg:#040616; --card:#071020; --muted:#9fb2c9; --accent:#7ee7b5; --glass: rgba(255,255,255,0.03);
    }
    body { background: linear-gradient(180deg,#020416 0%, #071020 70%); color: #e8f2fb; }
    .block-container{ max-width:1200px; margin:0 auto; padding:1.6rem 2rem;}
    .brand { display:flex; align-items:center; gap:12px; margin-bottom:8px; }
    .brand .logo { width:48px; height:48px; border-radius:12px; background:linear-gradient(135deg,#0ea5a7,#5eead4); display:flex; align-items:center; justify-content:center; font-weight:700; color:#02111a; font-size:20px; }
    .title { font-size:20px; font-weight:800; letter-spacing:-0.4px; }
    .subtitle { color:var(--muted); font-size:13px; margin-top:-4px; }
    .card { background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); border-radius:14px; padding:16px; border:1px solid rgba(255,255,255,0.04); box-shadow:0 10px 30px rgba(0,0,0,0.6); }
    .kpis { display:flex; gap:14px; margin-bottom:14px; }
    .kpi { padding:12px 16px; border-radius:12px; background: rgba(255,255,255,0.02); min-width:170px; }
    .kpi .num { font-size:20px; font-weight:800; }
    .muted { color:var(--muted); }
    .mcq-card { background: linear-gradient(180deg, rgba(255,255,255,0.015), rgba(255,255,255,0.01)); padding:14px; border-radius:12px; border:1px solid rgba(255,255,255,0.03); margin-bottom:10px; }
    .citation-badge { display:inline-block; padding:6px 10px; background:rgba(255,255,255,0.03); border-radius:999px; margin-right:8px; font-size:12px; color:var(--muted); }
    .searchbar { display:flex; gap:10px; align-items:center; }
    .right-muted { color:var(--muted); font-size:12px; }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

def sidebar_panel():
    st.sidebar.markdown("### 📚 StudyBuddy.ai — Ultra Premium")
    st.sidebar.markdown("Upload PDFs, create index, ask accurate Q&A, generate MCQs, and follow a personalized study plan.")
    st.sidebar.caption("Tip: Use searchable PDFs (not photos). For best RAG results, upload chapter-wise PDFs.")

# ------------------------
# APP: main UI
# ------------------------
def main():
    set_ultra_premium_theme()
    sidebar_panel()

    # Header
    st.markdown("""
    <div class='brand'>
      <div class='logo'>SB</div>
      <div>
        <div class='title'>StudyBuddy.ai</div>
        <div class='subtitle'>Ultra-premium study assistant — reliable answers, exam-quality MCQs, and study plans.</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # session init
    if "docs" not in st.session_state:
        st.session_state.docs = []
    if "index" not in st.session_state:
        st.session_state.index = None
    if "embeddings" not in st.session_state:
        st.session_state.embeddings = None
    if "embedder" not in st.session_state:
        st.session_state.embedder = load_embedder()
    if "use_hf" not in st.session_state:
        st.session_state.use_hf = False
    if "hf_key" not in st.session_state:
        st.session_state.hf_key = os.environ.get("HF_API_KEY", "")
    # lazy load local generator only if needed
    if "tokenizer" not in st.session_state and not st.session_state.use_hf:
        try:
            tok, mod, dev = load_generator_local()
            st.session_state.tokenizer = tok
            st.session_state.gen_model = mod
            st.session_state.device = dev
        except Exception:
            st.session_state.tokenizer = None
            st.session_state.gen_model = None
            st.session_state.device = None
    if "mcq_history" not in st.session_state:
        st.session_state.mcq_history = []
    if "raw_last" not in st.session_state:
        st.session_state.raw_last = ""

    # HF config – small UI control
    hf_col, _ = st.columns([3, 1])
    with hf_col:
        use_hf = st.checkbox("Use remote inference (Hugging Face Router) — recommended for production speed & quality", value=False)
        st.session_state.use_hf = use_hf
        if use_hf:
            key_input = st.text_input("Hugging Face API Key (or set HF_API_KEY env var)", value=st.session_state.hf_key, type="password")
            st.session_state.hf_key = key_input.strip()
            if not st.session_state.hf_key:
                st.warning("Remote inference enabled but no HF API key provided. Provide a key to use remote generation.")

    # KPI row
    progress = load_progress()
    total_chunks = len(st.session_state.docs)
    total_quizzes = sum([v["total"] for v in progress.values()]) if progress else 0
    mastery_scores = [compute_mastery_score(v) for v in progress.values()] if progress else []
    avg_mastery = round(float(np.mean(mastery_scores)) if mastery_scores else 0.0, 3)

    c1, c2, c3, c4 = st.columns([1.2, 1, 1, 1])
    with c1:
        st.markdown(f"<div class='kpi card'><div class='muted'>Indexed chunks</div><div class='num'>{total_chunks}</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='kpi card'><div class='muted'>MCQs generated</div><div class='num'>{len(st.session_state.mcq_history)}</div></div>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div class='kpi card'><div class='muted'>Quiz attempts</div><div class='num'>{total_quizzes}</div></div>", unsafe_allow_html=True)
    with c4:
        st.markdown(f"<div class='kpi card'><div class='muted'>Avg mastery</div><div class='num'>{avg_mastery:.2f}</div></div>", unsafe_allow_html=True)

    # Tabs
    tab_upload, tab_qa, tab_mcq, tab_plan = st.tabs(["📂 Upload & Index", "🔎 Ask Questions", "📝 Practice MCQs", "📅 Study Plan & Progress"])

    # === Upload & Index ===
    with tab_upload:
        st.subheader("Upload PDFs & Build Index")
        uploaded = st.file_uploader("Upload PDFs (multiple allowed)", type=["pdf"], accept_multiple_files=True)
        left, right = st.columns([3,1])
        with left:
            st.markdown("<div class='card'><b>Pro Tips</b><ul class='muted'><li>Searchable PDFs give best results</li><li>Split large books into chapters for fast indexing</li><li>Use debug toggles if results look off</li></ul></div>", unsafe_allow_html=True)
        with right:
            if st.button("🚀 Build / Refresh Index", use_container_width=True):
                if not uploaded:
                    st.warning("Upload at least one PDF.")
                else:
                    docs = []
                    for f in uploaded:
                        with st.spinner(f"Extracting from {f.name}..."):
                            txt = read_pdf_bytes(f)
                            if not txt.strip():
                                st.warning(f"No text found in {f.name}. Try OCR.")
                                continue
                            chs = chunk_text(txt)
                            for i, ch in enumerate(chs):
                                docs.append({"doc_id": f.name, "chunk_id": i, "text": ch})
                    if not docs:
                        st.error("No extractable text found.")
                    else:
                        with st.spinner("Computing embeddings & building FAISS index..."):
                            try:
                                index, embs = build_faiss_index(st.session_state.embedder, docs)
                                st.session_state.docs = docs
                                st.session_state.index = index
                                st.session_state.embeddings = embs
                                st.success(f"Index built — {len(docs)} chunks.")
                            except Exception as e:
                                st.exception(e)
                                st.error("Failed to build index.")
        if st.session_state.docs:
            st.markdown("Indexed documents (preview)")
            meta = pd.DataFrame(st.session_state.docs)[["doc_id", "chunk_id"]]
            st.dataframe(meta.head(50), use_container_width=True, height=220)

    # === Ask Questions ===
    with tab_qa:
        st.subheader("Ask questions — get concise, cited answers")
        if st.session_state.index is None:
            st.info("Build an index first in Upload tab.")
        else:
            q_col, opt_col = st.columns([3,1])
            with q_col:
                query = st.text_area("Type your question here (conceptual or problem)", height=140)
            with opt_col:
                top_k = st.slider("Citations (top-k chunks)", 3, 12, TOP_K_DEFAULT)
                debug = st.checkbox("Show raw model output", key="qa_debug")
                btn = st.button("🔍 Get Answer", use_container_width=True)
            if btn:
                if not query.strip():
                    st.warning("Please enter a question.")
                else:
                    # ensure local generator loaded if needed
                    if not st.session_state.use_hf and (st.session_state.tokenizer is None or st.session_state.gen_model is None):
                        try:
                            tok, mod, dev = load_generator_local()
                            st.session_state.tokenizer = tok
                            st.session_state.gen_model = mod
                            st.session_state.device = dev
                        except Exception as e:
                            st.error("Failed to load local generator. Try enabling remote HF inference or check logs.")
                            st.exception(e)
                    gen_mode = "hf" if st.session_state.use_hf else "local"
                    tokenizer = st.session_state.tokenizer
                    gen_model = st.session_state.gen_model
                    device = st.session_state.device
                    hf_key = st.session_state.hf_key or os.environ.get("HF_API_KEY", "")

                    with st.spinner("Retrieving context & generating answer..."):
                        retrieved = retrieve_chunks(query, st.session_state.embedder, st.session_state.index, st.session_state.docs, top_k=top_k)
                        answer_text, citations, raw = answer_from_context(query, retrieved, gen_mode, tokenizer=tokenizer, model=gen_model, device=device, hf_api_key=hf_key, max_answer_tokens=420)
                        st.session_state.raw_last = raw

                    st.markdown("<div class='card'><b>Answer</b></div>", unsafe_allow_html=True)
                    st.markdown(answer_text)
                    if citations:
                        st.markdown("<div style='margin-top:8px'><b>Citations used:</b></div>", unsafe_allow_html=True)
                        for c in citations:
                            st.markdown(f"<span class='citation-badge'>{c['doc_id']} • chunk {c['chunk_id']}</span>", unsafe_allow_html=True)
                    else:
                        st.markdown("<div class='muted'>No citations (model did not reference retrieved chunks).</div>", unsafe_allow_html=True)
                    if debug:
                        st.markdown("#### Raw model output")
                        st.code(st.session_state.raw_last)

    # === MCQ Generation & Practice ===
    with tab_mcq:
        st.subheader("Generate high-quality MCQs and practice")
        if st.session_state.index is None:
            st.info("Build an index first in Upload tab.")
        else:
            docs = st.session_state.docs
            unique_docs = sorted(list({d["doc_id"] for d in docs}))
            doc_choice = st.selectbox("Select document / chapter", unique_docs)
            doc_chunks = [d for d in docs if d["doc_id"] == doc_choice]
            st.caption(f"{len(doc_chunks)} chunks in {doc_choice}")

            left, right = st.columns([3,1])
            with left:
                num_q = st.slider("Number of MCQs", 3, 10, 5)
            with right:
                debug_mcq = st.checkbox("Show raw MCQ output", key="mcq_debug")

            if st.button("🎯 Generate MCQs", use_container_width=True):
                # ensure local generator loaded if needed
                if not st.session_state.use_hf and (st.session_state.tokenizer is None or st.session_state.gen_model is None):
                    try:
                        tok, mod, dev = load_generator_local()
                        st.session_state.tokenizer = tok
                        st.session_state.gen_model = mod
                        st.session_state.device = dev
                    except Exception as e:
                        st.error("Failed to load local generator. Try enabling remote HF inference or check logs.")
                        st.exception(e)
                gen_mode = "hf" if st.session_state.use_hf else "local"
                tokenizer = st.session_state.tokenizer
                gen_model = st.session_state.gen_model
                device = st.session_state.device
                hf_key = st.session_state.hf_key or os.environ.get("HF_API_KEY", "")

                candidate_text = " ".join([c["text"] for c in doc_chunks[:3]])[:4000]
                with st.spinner("Generating MCQs (may take 20-60s depending on mode)..."):
                    mcqs = generate_mcqs_from_chunk(candidate_text, num_q, gen_mode, tokenizer=tokenizer, model=gen_model, device=device, hf_api_key=hf_key)
                    st.session_state.mcq_history = mcqs
                if not mcqs:
                    st.error("MCQ generation failed or returned invalid format. Try again or reduce questions.")
                else:
                    st.success(f"{len(mcqs)} MCQs generated.")

            mcqs = st.session_state.mcq_history
            progress = load_progress()
            if mcqs:
                st.markdown("<div class='card'><b>Your MCQ Quiz</b></div>", unsafe_allow_html=True)
                for i, q in enumerate(mcqs):
                    topic_id = f"{doc_choice}::gen_q{i}"
                    st.markdown(f"<div class='mcq-card'><b>Q{i+1}. {q['question']}</b></div>", unsafe_allow_html=True)
                    c1, c2 = st.columns([3,1])
                    with c1:
                        choice = st.radio("", q["options"], key=f"mcq_choice_{i}")
                    with c2:
                        if st.button("Check", key=f"mcq_check_{i}"):
                            correct = q["options"][q["answer_index"]]
                            if choice == correct:
                                st.success("Correct ✅")
                                update_mastery(progress, topic_id, True)
                            else:
                                st.error(f"Incorrect — correct: **{correct}**")
                                update_mastery(progress, topic_id, False)
                            st.markdown(f"**Explanation:** {q.get('explanation','No explanation provided.')}")
                if debug_mcq and st.session_state.raw_last:
                    st.markdown("#### Raw MCQ generator output")
                    st.code(st.session_state.raw_last)

    # === Study Plan & Progress ===
    with tab_plan:
        st.subheader("Personalized Study Plan & Progress")
        docs = st.session_state.docs
        if not docs:
            st.info("Build index and generate MCQs to collect progress.")
        else:
            progress = load_progress()
            if progress:
                rows = []
                for key, stats in progress.items():
                    rows.append({"topic_id": key, "correct": stats["correct"], "total": stats["total"], "mastery": round(compute_mastery_score(stats), 3)})
                df = pd.DataFrame(rows).sort_values("mastery")
                st.markdown("<div class='card'><b>Mastery Overview</b></div>", unsafe_allow_html=True)
                st.dataframe(df, use_container_width=True, height=260)
            else:
                st.info("No quiz history yet.")

            days = st.slider("Plan length (days)", 3, 14, 7)
            if st.button("📅 Generate Study Plan", use_container_width=True):
                with st.spinner("Creating personalized study plan..."):
                    plan = generate_study_plan(progress, docs, days=days)
                st.success("Plan ready ✅")
                for day, items in plan.items():
                    if not items:
                        continue
                    st.markdown(f"### Day {day}")
                    for it in items:
                        st.markdown(f"- **{it['doc_id']} — chunk {it['chunk_id']}** (mastery: {it.get('mastery_score', 0):.2f})")

# ------------------------
# Study plan helper
# ------------------------
def generate_study_plan(progress: Dict[str, Any], docs: List[Dict[str, Any]], days: int = 7, max_per_day: int = 6) -> Dict[int, List[Dict[str, Any]]]:
    topic_map = {}
    for d in docs:
        key = f"{d['doc_id']}::chunk_{d['chunk_id']}"
        topic_map[key] = d
    all_keys = list(topic_map.keys())
    full_progress = {}
    for k in all_keys:
        full_progress[k] = progress.get(k, {"correct": 0, "total": 0})
    scored = []
    for k, stats in full_progress.items():
        scored.append((k, compute_mastery_score(stats)))
    scored = sorted(scored, key=lambda x: x[1])  # weaker first
    plan = {day: [] for day in range(1, days + 1)}
    i = 0
    for day in range(1, days + 1):
        for _ in range(max_per_day):
            if i >= len(scored):
                break
            key, score = scored[i]
            item = topic_map[key].copy()
            item["mastery_score"] = round(score, 3)
            plan[day].append(item)
            i += 1
        if i >= len(scored):
            break
    return plan

# ------------------------
# Entrypoint
# ------------------------
if __name__ == "__main__":
    main()