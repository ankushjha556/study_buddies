# app.py
"""
StudyBuddy.ai – Ultra Premium Streamlit App (flan-t5-base)
- Upload PDFs, extract text
- Chunking + semantic retrieval via SentenceTransformers + FAISS
- Robust Q&A with RAG (flan-t5-base)
- Reliable MCQ generation (few-shot JSON + beam decoding)
- Local progress tracking
- Ultra-premium dark UI (cards, KPIs, badges, debug modes)
"""

import os
import json
import re
from typing import List, Dict, Any, Tuple

import streamlit as st
import numpy as np
import pandas as pd
import pdfplumber
from sentence_transformers import SentenceTransformer
import faiss
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ------------------------
# CONFIG
# ------------------------
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GEN_MODEL_NAME = "google/flan-t5-base"         # swapped to base for better quality
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

@st.cache_resource(show_spinner="🤖 Loading generator model (flan-t5-base)...")
def load_generator():
    # auto-detect device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_NAME, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL_NAME)
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
    texts = [d["text"] for d in docs]
    emb = embedder.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    faiss.normalize_L2(emb)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    return index, emb

def retrieve_chunks(query: str, embedder: SentenceTransformer, index, docs: List[Dict[str, Any]], top_k: int = TOP_K_DEFAULT):
    q_emb = embedder.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    scores, idxs = index.search(q_emb, top_k)
    idxs = idxs[0]
    scores = scores[0]
    results = []
    for i, score in zip(idxs, scores):
        if i < 0 or i >= len(docs):
            continue
        r = docs[i].copy()
        r["score"] = float(score)
        results.append(r)
    # dedupe by (doc_id, chunk_id)
    seen = {}
    for r in results:
        key = (r["doc_id"], r["chunk_id"])
        if key not in seen or r["score"] > seen[key]["score"]:
            seen[key] = r
    results = sorted(seen.values(), key=lambda x: x["score"], reverse=True)
    return results

# ------------------------
# GENERATOR: deterministic beams, safe decoding
# ------------------------
def run_generator(prompt: str, tokenizer, model, device, max_tokens: int = 256,
                  num_beams: int = 4, do_sample: bool = False, temperature: float = 0.0):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
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
    with torch.no_grad():
        outputs = model.generate(**gen_kwargs)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text.strip()

# ------------------------
# MCQ generation (few-shot + JSON)
# ------------------------
def generate_mcqs_from_chunk(chunk: str, num_questions: int, tokenizer, model, device) -> List[Dict[str, Any]]:
    example_json = """
[
  {
    "question": "What is the primary purpose of hypothesis testing?",
    "options": [
      "To estimate the population mean",
      "To test a claim about a population using sample data",
      "To compute descriptive statistics",
      "To visualize data distribution"
    ],
    "answer_index": 1,
    "explanation": "Hypothesis testing evaluates evidence from a sample to decide whether to reject a claim about the population."
  }
]
"""
    system_prompt = (
        "You are an expert tutor. Output VALID JSON ONLY. "
        "From the STUDY MATERIAL generate EXACTLY %d multiple-choice questions. "
        "Each item must be an object with: "
        "\"question\" (string), \"options\" (array of 4 short strings), "
        "\"answer_index\" (0-based index), \"explanation\" (brief). "
        "Return a JSON array only, no commentary."
    ) % num_questions

    material = chunk.strip()
    if len(material) > 2400:
        material = material[:2400]

    prompt = (
        system_prompt + "\n\n"
        "EXAMPLE_OUTPUT:\n" + example_json + "\n\n"
        "STUDY MATERIAL:\n" + material + "\n\n"
        "Output the JSON array now."
    )

    raw = run_generator(prompt, tokenizer, model, device, max_tokens=512, num_beams=6, do_sample=False, temperature=0.0)

    # attempt to extract JSON array
    m = re.search(r"(\[.*\])", raw, flags=re.S)
    json_text = m.group(1) if m else raw
    try:
        data = json.loads(json_text)
    except Exception:
        import ast
        try:
            data = ast.literal_eval(json_text)
        except Exception:
            return []  # fail gracefully

    mcqs = []
    for item in data:
        try:
            q = str(item["question"]).strip()
            options = [str(x).strip() for x in item["options"]][:4]
            if len(options) < 4:
                continue
            ans_idx = int(item["answer_index"])
            if not (0 <= ans_idx < 4):
                continue
            explanation = str(item.get("explanation", "")).strip()
            mcqs.append({"question": q, "options": options, "answer_index": ans_idx, "explanation": explanation})
        except Exception:
            continue
    return mcqs

# ------------------------
# Q&A prompt & citation extraction
# ------------------------
def answer_from_context(question: str, retrieved_chunks: List[Dict[str, Any]], tokenizer, model, device, max_answer_tokens: int = 320) -> Tuple[str, List[Dict[str, Any]], str]:
    context_parts = []
    for r in retrieved_chunks[:8]:
        ctx = f"[{r['doc_id']}::chunk_{r['chunk_id']}]\n{r['text'][:900]}"
        context_parts.append(ctx)
    context_text = "\n\n".join(context_parts)

    prompt = (
        "You are a precise, concise tutor. Answer the QUESTION USING ONLY the STUDY MATERIAL below. "
        "If the material doesn't contain the answer, reply exactly: "
        "\"I don't know based on the provided material.\" "
        "Write: (1) Short answer (1-3 sentences), (2) short rationale, (3) a 'CITATIONS:' line listing sources as [doc::chunk_id]. "
        "Do NOT hallucinate."
        f"\n\nSTUDY MATERIAL:\n{context_text}\n\nQUESTION: {question}\n\nAnswer now."
    )

    raw = run_generator(prompt, tokenizer, model, device, max_tokens=max_answer_tokens, num_beams=4, do_sample=False, temperature=0.0)

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
# Premium CSS
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
    .brand {
        display:flex; align-items:center; gap:12px; margin-bottom:8px;
    }
    .brand .logo {
        width:48px; height:48px; border-radius:12px; background:linear-gradient(135deg,#0ea5a7,#5eead4);
        display:flex; align-items:center; justify-content:center; font-weight:700; color:#02111a; font-size:20px;
    }
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

# ------------------------
# App: UI
# ------------------------
def sidebar_panel():
    st.sidebar.markdown("### 📚 StudyBuddy.ai — Ultra Premium")
    st.sidebar.markdown("Upload PDFs, create index, ask accurate Q&A, generate MCQs, and follow a personalized study plan.")
    st.sidebar.caption("Tip: Use searchable PDFs (no photos). For best RAG results, upload chapter-wise PDFs.")

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
    if "embedder" not in st.session_state:
        st.session_state.embedder = load_embedder()
    if "tokenizer" not in st.session_state:
        tok, mod, dev = load_generator()
        st.session_state.tokenizer = tok
        st.session_state.gen_model = mod
        st.session_state.device = dev
    if "mcq_history" not in st.session_state:
        st.session_state.mcq_history = []
    if "raw_last" not in st.session_state:
        st.session_state.raw_last = ""

    # KPI row
    col1, col2, col3, col4 = st.columns([1.2, 1, 1, 1])
    progress = load_progress()
    total_chunks = len(st.session_state.docs)
    total_quizzes = sum([v["total"] for v in progress.values()]) if progress else 0
    mastery_scores = [compute_mastery_score(v) for v in progress.values()] if progress else []
    avg_mastery = round(float(np.mean(mastery_scores)) if mastery_scores else 0.0, 3)
    with col1:
        st.markdown(f"<div class='kpi card'><div class='muted'>Indexed chunks</div><div class='num'>{total_chunks}</div></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='kpi card'><div class='muted'>MCQs generated</div><div class='num'>{len(st.session_state.mcq_history)}</div></div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='kpi card'><div class='muted'>Quiz attempts</div><div class='num'>{total_quizzes}</div></div>", unsafe_allow_html=True)
    with col4:
        st.markdown(f"<div class='kpi card'><div class='muted'>Avg mastery</div><div class='num'>{avg_mastery:.2f}</div></div>", unsafe_allow_html=True)

    # Tabs
    tab_upload, tab_qa, tab_mcq, tab_plan = st.tabs(["📂 Upload & Index", "🔎 Ask Questions", "📝 Practice MCQs", "📅 Study Plan & Progress"])

    # === Upload & Index ===
    with tab_upload:
        st.subheader("Upload PDFs & Build Index")
        uploaded = st.file_uploader("Upload PDFs (multiple allowed)", type=["pdf"], accept_multiple_files=True)
        left, right = st.columns([3,1])
        with left:
            st.markdown("<div class='card'><b>Pro Tips</b><ul class='muted'><li>Searchable PDFs give best results</li><li>Split large books into chapters for fast indexing</li><li>Use the debug toggle if results look off</li></ul></div>", unsafe_allow_html=True)
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
                                index, _ = build_faiss_index(st.session_state.embedder, docs)
                                st.session_state.docs = docs
                                st.session_state.index = index
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
                debug = st.checkbox("Show raw output", key="qa_debug")
                btn = st.button("🔍 Get Answer", use_container_width=True)
            if btn:
                if not query.strip():
                    st.warning("Please enter a question.")
                else:
                    embedder = st.session_state.embedder
                    index = st.session_state.index
                    docs = st.session_state.docs
                    tokenizer = st.session_state.tokenizer
                    gen_model = st.session_state.gen_model
                    device = st.session_state.device

                    with st.spinner("Retrieving context & generating answer..."):
                        retrieved = retrieve_chunks(query, embedder, index, docs, top_k=top_k)
                        answer_text, citations, raw = answer_from_context(query, retrieved, tokenizer, gen_model, device, max_answer_tokens=420)
                        st.session_state.raw_last = raw

                    # Nicely formatted answer card
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
                tokenizer = st.session_state.tokenizer
                gen_model = st.session_state.gen_model
                device = st.session_state.device
                candidate_text = " ".join([c["text"] for c in doc_chunks[:3]])[:4000]
                with st.spinner("Generating MCQs (this may take 20-60s depending on CPU/GPU)..."):
                    mcqs = generate_mcqs_from_chunk(candidate_text, num_q, tokenizer, gen_model, device)
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
# Study plan helper (same algorithm)
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