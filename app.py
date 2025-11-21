import os
import json
import time
from typing import List, Dict, Any, Tuple

import streamlit as st
import numpy as np
import pandas as pd
from tqdm import tqdm

import pdfplumber
from sentence_transformers import SentenceTransformer
import faiss

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# ============================
#  CONFIG / CONSTANTS
# ============================

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GEN_MODEL_NAME = "google/flan-t5-small"  # light, free, works on CPU
INDEX_DIM = 384  # all-MiniLM-L6-v2 embedding size
TOP_K = 8        # retrieved chunks for RAG
CHUNK_SIZE = 2000   # characters per chunk
CHUNK_OVERLAP = 300

PROGRESS_FILE = os.path.join("data", "user_progress.json")
os.makedirs("data", exist_ok=True)


# ============================
#  UTILITIES
# ============================

@st.cache_resource(show_spinner="🔁 Loading embedding model...")
def load_embedder():
    model = SentenceTransformer(EMBED_MODEL_NAME)
    return model


@st.cache_resource(show_spinner="🤖 Loading FLAN-T5 model (first time only)...")
def load_generator():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL_NAME)
    model.to(device)
    return tokenizer, model, device


def read_pdf_bytes(uploaded_file) -> str:
    """Extract text from uploaded PDF using pdfplumber."""
    with pdfplumber.open(uploaded_file) as pdf:
        texts = []
        for page in pdf.pages:
            try:
                txt = page.extract_text()
            except Exception:
                txt = ""
            if txt:
                texts.append(txt)
    full_text = "\n".join(texts)
    return full_text


def clean_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = text.replace("\t", " ")
    return " ".join(text.split())


def chunk_text(text: str,
               chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Sliding-window character-based chunking (robust for PDFs)."""
    text = clean_text(text)
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        # avoid cutting sentences aggressively – extend to nearest period
        if end < len(text):
            period_pos = text.rfind(".", start, end)
            if period_pos != -1 and period_pos > start + 0.4 * chunk_size:
                end = period_pos + 1
                chunk = text[start:end]
        chunks.append(chunk.strip())
        start = end - overlap
        if start < 0:
            start = 0
        if start >= len(text):
            break
    return [c for c in chunks if len(c) > 50]


def build_faiss_index(
    embedder: SentenceTransformer,
    docs: List[Dict[str, Any]]
) -> Tuple[faiss.IndexFlatIP, np.ndarray]:
    """
    docs: list of {"doc_id": str, "chunk_id": int, "text": str}
    returns: (faiss_index, embedding_matrix)
    """
    texts = [d["text"] for d in docs]
    embeddings = embedder.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings


def retrieve_chunks(
    query: str,
    embedder: SentenceTransformer,
    index: faiss.IndexFlatIP,
    docs: List[Dict[str, Any]],
    top_k: int = TOP_K,
) -> List[Dict[str, Any]]:
    q_emb = embedder.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    scores, idxs = index.search(q_emb, top_k)
    idxs = idxs[0]
    scores = scores[0]
    results = []
    for i, score in zip(idxs, scores):
        if i < 0 or i >= len(docs):
            continue
        item = docs[i].copy()
        item["score"] = float(score)
        results.append(item)
    # remove duplicates / sort
    results = sorted({(r["doc_id"], r["chunk_id"]): r for r in results}.values(),
                     key=lambda x: x["score"],
                     reverse=True)
    return results


def run_generator(prompt: str,
                  tokenizer,
                  model,
                  device,
                  max_tokens: int = 256) -> str:
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
        )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text.strip()


# ============================
#  MCQ GENERATION
# ============================

def generate_mcqs_from_chunk(
    chunk: str,
    num_questions: int,
    tokenizer,
    model,
    device,
) -> List[Dict[str, Any]]:
    """
    Returns list of MCQs:
    {
        "question": str,
        "options": [optA, optB, optC, optD],
        "answer_index": int,
        "explanation": str
    }
    """
    system_instruction = (
        "You are an expert university tutor. "
        "From the following study material, generate high-quality multiple-choice questions. "
        f"Create exactly {num_questions} questions. "
        "Return them as JSON list. Each item must have keys: "
        "`question` (string), `options` (list of 4 short options), "
        "`answer_index` (0-based index of correct option), "
        "`explanation` (short explanation). "
        "Do not include backticks or extra text, only raw JSON."
    )

    prompt = system_instruction + "\n\nSTUDY MATERIAL:\n" + chunk[:2500]
    raw = run_generator(prompt, tokenizer, model, device, max_tokens=512)

    # Very defensive JSON parsing
    import re
    import ast
    # try to cut out JSON array
    match = re.search(r"\[.*\]", raw, re.S)
    if match:
        raw = match.group(0)
    try:
        data = json.loads(raw)
    except Exception:
        try:
            # last resort: ast.literal_eval
            data = ast.literal_eval(raw)
        except Exception:
            return []

    mcqs = []
    for item in data:
        try:
            q = str(item["question"]).strip()
            options = [str(o).strip() for o in item["options"]][:4]
            if len(options) < 4:
                continue
            ans_idx = int(item["answer_index"])
            if not (0 <= ans_idx < len(options)):
                continue
            expl = str(item.get("explanation", "")).strip()
            mcqs.append(
                {
                    "question": q,
                    "options": options,
                    "answer_index": ans_idx,
                    "explanation": expl,
                }
            )
        except Exception:
            continue
    return mcqs


# ============================
#  PROGRESS TRACKING
# ============================

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


def update_mastery(progress: Dict[str, Any],
                   topic_id: str,
                   correct: bool):
    """
    Simple mastery model: keep counts of correct/total + rolling score.
    """
    if topic_id not in progress:
        progress[topic_id] = {"correct": 0, "total": 0}
    progress[topic_id]["total"] += 1
    if correct:
        progress[topic_id]["correct"] += 1
    save_progress(progress)


def compute_mastery_score(stats: Dict[str, int]) -> float:
    if stats["total"] == 0:
        return 0.0
    # simple smoothed accuracy
    return (stats["correct"] + 1) / (stats["total"] + 2)


# ============================
#  STUDY PLAN GENERATOR
# ============================

def generate_study_plan(
    progress: Dict[str, Any],
    docs: List[Dict[str, Any]],
    days: int = 7,
    max_per_day: int = 6,
) -> Dict[int, List[Dict[str, Any]]]:
    """
    Very simple heuristic:
    - Map each chunk (doc_id, chunk_id) to mastery score
    - Sort ascending (weak topics first)
    - Fill days with up to max_per_day items
    """
    # build topic key -> chunk text meta
    topic_map = {}
    for d in docs:
        key = f"{d['doc_id']}::chunk_{d['chunk_id']}"
        topic_map[key] = d

    # ensure every topic has some stats
    all_keys = list(topic_map.keys())
    full_progress = {}
    for k in all_keys:
        full_progress[k] = progress.get(k, {"correct": 0, "total": 0})

    scored = []
    for k, stats in full_progress.items():
        scored.append((k, compute_mastery_score(stats)))

    scored = sorted(scored, key=lambda x: x[1])  # low mastery first

    plan: Dict[int, List[Dict[str, Any]]] = {day: [] for day in range(1, days + 1)}
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


# ============================
#  STREAMLIT UI
# ============================

def set_dark_theme():
    st.set_page_config(
        page_title="StudyBuddy.ai – Smart Study Assistant",
        page_icon="📘",
        layout="wide",
    )
    dark_css = """
    <style>
        body { background-color: #050816; color: #f5f5f5; }
        .main { background-color: #050816; }
        .stApp { background: radial-gradient(circle at top, #1f2937 0, #020617 55%); }
        .block-container { padding-top: 1rem; }
        h1, h2, h3, h4, h5, h6 { color: #e5e7eb !important; }
        .metric-label { color: #cbd5f5 !important; }
        .mcq-card {
            background: linear-gradient(145deg, #020617, #0f172a);
            padding: 1rem 1.2rem;
            border-radius: 1rem;
            border: 1px solid rgba(148,163,184,0.35);
            margin-bottom: 0.8rem;
        }
        .info-card {
            background: linear-gradient(145deg, #020617, #111827);
            padding: 1.2rem 1.4rem;
            border-radius: 1rem;
            border: 1px solid rgba(55,65,81,0.8);
        }
    </style>
    """
    st.markdown(dark_css, unsafe_allow_html=True)


def sidebar_header():
    st.sidebar.markdown(
        "### 📚 StudyBuddy.ai\n"
        "AI-powered study assistant for your PDFs.\n\n"
        "**Steps:**\n"
        "1. Upload 1–3 PDFs (books/notes)\n"
        "2. Generate index (1-2 min)\n"
        "3. Ask questions / create MCQs\n"
        "4. Track weak topics & get a 7-day plan\n"
    )


def main():
    set_dark_theme()
    sidebar_header()

    st.title("📘 StudyBuddy.ai – Smart Study & MCQ Assistant")

    # --- Session state init ---
    if "docs" not in st.session_state:
        st.session_state.docs: List[Dict[str, Any]] = []
    if "index" not in st.session_state:
        st.session_state.index = None
    if "embedder" not in st.session_state:
        st.session_state.embedder = load_embedder()
    if "tokenizer" not in st.session_state:
        tok, model, device = load_generator()
        st.session_state.tokenizer = tok
        st.session_state.gen_model = model
        st.session_state.device = device
    if "mcq_history" not in st.session_state:
        st.session_state.mcq_history = []

    # --- Layout ---
    tab_upload, tab_qa, tab_mcq, tab_progress = st.tabs(
        ["📂 Upload & Index", "🔎 Ask Questions", "📝 Practice MCQs", "📊 Progress & Plan"]
    )

    # ============================
    #  TAB 1: UPLOAD & INDEX
    # ============================
    with tab_upload:
        st.subheader("Step 1 – Upload your PDFs & build knowledge index")

        uploaded_files = st.file_uploader(
            "Upload textbooks / notes (PDF). Multiple allowed.",
            type=["pdf"],
            accept_multiple_files=True,
        )

        build_col1, build_col2 = st.columns([2, 1])

        with build_col1:
            st.markdown(
                "<div class='info-card'>"
                "<b>Tips for best results:</b><br>"
                "• Prefer searchable PDFs (not photos).<br>"
                "• Upload 1–3 chapters at a time for faster indexing.<br>"
                "• You can re-index anytime if you add new PDFs."
                "</div>",
                unsafe_allow_html=True,
            )

        with build_col2:
            if st.button("🚀 Build / Refresh Index", use_container_width=True):
                if not uploaded_files:
                    st.warning("Please upload at least one PDF.")
                else:
                    docs = []
                    for file in uploaded_files:
                        with st.spinner(f"Extracting text from **{file.name}**..."):
                            text = read_pdf_bytes(file)
                            if not text.strip():
                                st.warning(f"No text found in {file.name} (maybe scanned only).")
                                continue
                            chunks = chunk_text(text)
                            for idx, ch in enumerate(chunks):
                                docs.append(
                                    {
                                        "doc_id": file.name,
                                        "chunk_id": idx,
                                        "text": ch,
                                    }
                                )
                    if not docs:
                        st.error("No valid text extracted from any PDF.")
                    else:
                        with st.spinner("🔍 Computing embeddings & building FAISS index..."):
                            index, _ = build_faiss_index(st.session_state.embedder, docs)
                        st.session_state.docs = docs
                        st.session_state.index = index
                        st.success(f"Index built ✅  | Total chunks: {len(docs)}")

        if st.session_state.docs:
            st.markdown("#### Indexed documents")
            meta = pd.DataFrame(st.session_state.docs)[["doc_id", "chunk_id"]]
            st.dataframe(meta.head(20), use_container_width=True, height=260)

    # ============================
    #  TAB 2: Q&A
    # ============================
    with tab_qa:
        st.subheader("Step 2 – Ask questions from your PDFs")

        if st.session_state.index is None:
            st.info("Please build the index first in the **Upload & Index** tab.")
        else:
            query = st.text_area(
                "🧠 Ask any conceptual / numerical question...",
                placeholder="e.g., What is the definition of conditional probability? "
                            "Explain with a simple example.",
                height=120,
            )
            colA, colB = st.columns([2, 1])
            with colA:
                top_k = st.slider("Number of supporting chunks (citations)", 3, 12, TOP_K)
            with colB:
                if st.button("🔍 Answer from my PDFs", use_container_width=True):
                    if not query.strip():
                        st.warning("Please type a question.")
                    else:
                        embedder = st.session_state.embedder
                        index = st.session_state.index
                        docs = st.session_state.docs
                        tokenizer = st.session_state.tokenizer
                        gen_model = st.session_state.gen_model
                        device = st.session_state.device

                        with st.spinner("Retrieving relevant context + generating answer..."):
                            retrieved = retrieve_chunks(
                                query, embedder, index, docs, top_k=top_k
                            )
                            context_text = "\n\n".join(
                                [f"[{r['doc_id']} – chunk {r['chunk_id']}]\n{r['text']}"
                                 for r in retrieved]
                            )[:5000]

                            prompt = (
                                "You are an expert statistics tutor for university students. "
                                "Answer the student's question *only* using the provided material. "
                                "Explain clearly but concisely. If the answer is not in the "
                                "material, honestly say you are not sure.\n\n"
                                f"STUDY MATERIAL:\n{context_text}\n\n"
                                f"QUESTION: {query}\n\n"
                                "ANSWER:"
                            )

                            answer = run_generator(
                                prompt,
                                tokenizer,
                                gen_model,
                                device,
                                max_tokens=320,
                            )

                        st.markdown("### ✅ Answer")
                        st.markdown(answer)

                        with st.expander("🔗 View retrieved context & citations"):
                            for r in retrieved:
                                st.markdown(
                                    f"**{r['doc_id']} – chunk {r['chunk_id']} "
                                    f"(score={r['score']:.3f})**"
                                )
                                st.write(r["text"][:800] + ("..." if len(r["text"]) > 800 else ""))
                                st.markdown("---")

    # ============================
    #  TAB 3: MCQ PRACTICE
    # ============================
    with tab_mcq:
        st.subheader("Step 3 – Generate & practice MCQs")

        if st.session_state.index is None:
            st.info("Please build the index first in the **Upload & Index** tab.")
        else:
            docs = st.session_state.docs
            unique_docs = sorted(list({d["doc_id"] for d in docs}))
            doc_choice = st.selectbox("Choose document / chapter", unique_docs)

            # Filter chunks for selected doc
            doc_chunks = [d for d in docs if d["doc_id"] == doc_choice]
            st.caption(f"{len(doc_chunks)} chunks available from {doc_choice}")

            num_q = st.slider("Number of MCQs to generate", 3, 10, 5)

            if st.button("🎯 Generate MCQs", use_container_width=True):
                tokenizer = st.session_state.tokenizer
                gen_model = st.session_state.gen_model
                device = st.session_state.device

                # choose a representative chunk (or concatenate 2)
                selected = " ".join([c["text"] for c in doc_chunks[:3]])[:4000]

                with st.spinner("Generating exam-style MCQs (may take ~30–60s)..."):
                    mcqs = generate_mcqs_from_chunk(
                        selected, num_questions=num_q,
                        tokenizer=tokenizer,
                        model=gen_model,
                        device=device,
                    )

                if not mcqs:
                    st.error("MCQ generation failed. Try again or reduce number of questions.")
                else:
                    st.success(f"Generated {len(mcqs)} questions ✅")
                    st.session_state.mcq_history = mcqs

        # --- Display MCQs & capture answers ---
        mcqs = st.session_state.get("mcq_history", [])
        progress = load_progress()
        if mcqs:
            st.markdown("### 🧪 Your MCQ Quiz")
            score = 0
            total = len(mcqs)
            for i, q in enumerate(mcqs):
                topic_id = f"{doc_choice}::gen_q{i}"
                with st.container():
                    st.markdown(
                        f"<div class='mcq-card'><b>Q{i+1}. {q['question']}</b></div>",
                        unsafe_allow_html=True,
                    )
                    choice = st.radio(
                        f"Select answer for Q{i+1}",
                        q["options"],
                        key=f"mcq_{i}",
                    )
                    correct_option = q["options"][q["answer_index"]]
                    is_correct = choice == correct_option
                    if st.button(f"Check Q{i+1}", key=f"check_{i}"):
                        if is_correct:
                            st.success(f"✅ Correct!  (Answer: {correct_option})")
                            score += 1
                        else:
                            st.error(f"❌ Incorrect. Correct answer: **{correct_option}**")
                        if q.get("explanation"):
                            st.info(f"📘 Explanation: {q['explanation']}")
                        # update mastery
                        update_mastery(progress, topic_id, is_correct)

            if total > 0:
                st.markdown("---")
                st.markdown(f"**Session score (manual):** check each Q to see correctness.**")

    # ============================
    #  TAB 4: PROGRESS & STUDY PLAN
    # ============================
    with tab_progress:
        st.subheader("Step 4 – Track weak topics & get a 7-day plan")

        docs = st.session_state.docs
        if not docs:
            st.info("Index is empty. Upload PDFs and build index first.")
        else:
            progress = load_progress()

            # Show simple mastery table
            rows = []
            for key, stats in progress.items():
                rows.append(
                    {
                        "topic_id": key,
                        "correct": stats["correct"],
                        "total": stats["total"],
                        "mastery": round(compute_mastery_score(stats), 3),
                    }
                )
            if rows:
                df = pd.DataFrame(rows).sort_values("mastery")
                st.markdown("#### Mastery overview (lower = weaker topics)")
                st.dataframe(df, use_container_width=True, height=260)
            else:
                st.info("No quiz history yet. Solve some MCQs to build progress.")

            days = st.slider("Study plan length (days)", 3, 14, 7)
            if st.button("📅 Generate personalized study plan", use_container_width=True):
                with st.spinner("Building your plan..."):
                    plan = generate_study_plan(progress, docs, days=days)
                st.success("Study plan ready ✅")

                for day, items in plan.items():
                    if not items:
                        continue
                    st.markdown(f"### Day {day}")
                    for it in items:
                        key = f"{it['doc_id']}::chunk_{it['chunk_id']}"
                        st.markdown(
                            f"- **{it['doc_id']} – chunk {it['chunk_id']}**  "
                            f"(mastery: {it.get('mastery_score', 0):.2f})"
                        )


if __name__ == "__main__":
    main()