"""
=============================================================
  Agentic RAG — Streamlit UI  (app.py)
=============================================================
  Run:  streamlit run app.py
=============================================================
"""

import os
import tempfile
import streamlit as st
from agentic_rag import build_graph, prepare_vector_db, run_query

# ── Page config ──────────────────────────────────────────
st.set_page_config(
    page_title="Agentic RAG Demo",
    page_icon="🤖",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────
st.markdown("""
<style>
/* Overall background */
.stApp { background: #0f1117; }

/* Cards */
.rag-card {
    background: #1a1d27;
    border: 1px solid #2e3147;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
}

/* Step log items */
.step-item {
    font-family: 'Fira Code', monospace;
    font-size: 0.85rem;
    color: #a0aec0;
    padding: 0.25rem 0;
    border-left: 3px solid #4f6ef7;
    padding-left: 0.75rem;
    margin-bottom: 0.4rem;
}

/* Citation block */
.citation-block {
    background: #12151f;
    border-left: 4px solid #38b2ac;
    border-radius: 6px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.6rem;
    font-size: 0.85rem;
    color: #cbd5e0;
}

.citation-label {
    font-weight: 700;
    color: #38b2ac;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Answer box */
.answer-box {
    background: linear-gradient(135deg, #1a1d27 0%, #141720 100%);
    border: 1px solid #4f6ef7;
    border-radius: 12px;
    padding: 1.5rem;
    color: #e2e8f0;
    font-size: 1rem;
    line-height: 1.75;
}
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────
if "graph"        not in st.session_state: st.session_state.graph        = None
if "num_chunks"   not in st.session_state: st.session_state.num_chunks   = 0
if "history"      not in st.session_state: st.session_state.history      = []
if "api_key_set"  not in st.session_state: st.session_state.api_key_set  = False


# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Configuration")

    st.divider()

    # Upload documents
    st.subheader("📄 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="All uploaded PDFs are indexed into a shared vector store.",
    )

    if uploaded_files:
        if st.button("🚀 Build Vector Store", use_container_width=True):
            with st.spinner("Indexing documents …"):
                tmp_paths = []
                original_names = []
                for uf in uploaded_files:
                    suffix = os.path.splitext(uf.name)[1]
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(uf.read())
                        tmp_paths.append(tmp.name)          # just the path string
                        original_names.append(uf.name)      # original name separately

                try:
                    retriever, n_chunks = prepare_vector_db(tmp_paths, original_names)  # pass both
                    st.session_state.graph      = build_graph(retriever)
                    st.session_state.num_chunks = n_chunks
                    st.success(f"✅ Indexed **{n_chunks}** chunks from {len(uploaded_files)} file(s)")
                except Exception as e:
                    st.error(f"Error building index: {e}")
                finally:
                    for p in tmp_paths:
                        os.unlink(p)          # p is a plain string now, unlink works fine
    elif uploaded_files and not st.session_state.api_key_set:
        st.warning("Enter your OpenAI API key first.")

    st.divider()

    # Stats
    if st.session_state.graph:
        st.metric("Chunks in store", st.session_state.num_chunks)
        st.metric("Queries answered", len(st.session_state.history))

    # Architecture diagram
    with st.expander("🗺️ Agent Architecture"):
        st.markdown("""
```
         ┌──────────┐
    ──►  │ RETRIEVE │
         └────┬─────┘
              │
         ┌────▼────────┐
         │  GRADE DOCS │
         └────┬────────┘
        no    │   yes
    ┌─────────┤
    │         │
    ▼     ┌───▼──────┐
REWRITE   │ GENERATE │
    │     └───┬──────┘
    └──►       │
         ┌─────▼──────────┐
         │  HALLUCINATION │
         │    CHECK       │
         └─────┬──────────┘
        fail   │  pass
          ├────┘
          │
         END
```
        """)


# ── Main area ─────────────────────────────────────────────
st.title("🤖 Agentic RAG System")
st.caption("Upload PDFs → Ask questions → Watch the agent reason step-by-step")

if not st.session_state.graph:
    st.info("👈 Upload a PDF and build the vector store in the sidebar to get started.")
    st.stop()

# Query input
question = st.text_input(
    "Ask a question about your documents",
    placeholder="e.g. What are the main findings of this report?",
)

col_run, col_clear = st.columns([3, 1])
with col_run:
    run_btn = st.button("🔎 Run Agentic RAG", use_container_width=True, type="primary")
with col_clear:
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.history = []

# ── Run the agent ─────────────────────────────────────────
if run_btn and question:
    with st.spinner("Agent is reasoning …"):
        result = run_query(st.session_state.graph, question)
        st.session_state.history.append({
            "question"  : question,
            "answer"    : result["answer"],
            "citations" : result["citations"],
            "steps"     : result["steps"],
        })

# ── Display history (newest first) ───────────────────────
for entry in reversed(st.session_state.history):
    st.markdown("---")

    # Question
    st.markdown(f"### 💬 {entry['question']}")

    # Three columns: steps | answer | citations
    col_steps, col_answer, col_cites = st.columns([1.2, 2, 1.5])

    with col_steps:
        st.markdown("**🧠 Agent Trace**")
        for step in entry["steps"]:
            st.markdown(f'<div class="step-item">{step}</div>', unsafe_allow_html=True)

    with col_answer:
        st.markdown("**📝 Answer**")
        st.markdown(
            f'<div class="answer-box">{entry["answer"]}</div>',
            unsafe_allow_html=True,
        )

    with col_cites:
        st.markdown("**📚 Citations**")
        if entry["citations"]:
            for i, cite in enumerate(entry["citations"], 1):
                st.markdown(
                    f"""<div class="citation-block">
                        <div class="citation-label">📄 [{i}] {cite['source']} — p.{cite['page']}</div>
                        <div style="margin-top:0.4rem">{cite['snippet']}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
        else:
            st.info("No citations available.")