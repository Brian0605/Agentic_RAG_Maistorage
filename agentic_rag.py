"""
=============================================================
  Agentic RAG — Core Engine
=============================================================
  Architecture (LangGraph):
    retrieve → grade_documents ──(relevant)──► generate → hallucination_check ──(grounded)──► END
                     │                                              │
                  (not relevant)                             (not grounded)
                     ▼                                             ▼
               rewrite_query → retrieve                      generate (retry)

  Key Features:
    ✅ Correct chunk retrieval (RecursiveCharacterTextSplitter)
    ✅ Document relevance grading (LLM-as-judge)
    ✅ Query rewriting when retrieval fails
    ✅ Hallucination / faithfulness check before output
    ✅ Citation tracking (chunk source + page)
    ✅ Retry limit to prevent infinite loops
    ✅ Full observable step log for Streamlit UI
"""

import os
from typing import List, TypedDict, Literal

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, StateGraph

# ─────────────────────────────────────────────
# 0.  Configuration
# ─────────────────────────────────────────────
CHUNK_SIZE       = 500   # characters per chunk
CHUNK_OVERLAP    = 75    # overlap to preserve context across chunk boundaries
MAX_RETRIES      = 2     # max query-rewrite attempts before giving up
TOP_K_RETRIEVAL  = 5     # number of chunks retrieved per query

# ─────────────────────────────────────────────
# 1.  Document ingestion & vector store
# ─────────────────────────────────────────────

def prepare_vector_db(file_paths: list[str], original_names: list[str] = None) -> object:
    """
    Load PDF(s), split into chunks, embed, and store in ChromaDB.

    Chunking strategy:
      - RecursiveCharacterTextSplitter tries to split on paragraphs → sentences → words
        in that order, so chunks stay semantically coherent.
      - chunk_overlap retains a small window at chunk boundaries so context is
        never abruptly lost.

    Returns a LangChain retriever.
    """
    all_docs: list[Document] = []

    for i, path in enumerate(file_paths):
        # Use original name if provided, otherwise fall back to temp filename
        display_name = original_names[i] if original_names else os.path.basename(path)

        loader = PyPDFLoader(path)
        pages  = loader.load()
        for page in pages:
            page.metadata["source"] = display_name   # ← correct original name
            print(f"Page metadata: {page.metadata}")  # remove this after testing
        all_docs.extend(pages)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size    = CHUNK_SIZE,
        chunk_overlap = CHUNK_OVERLAP,
        separators    = ["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(all_docs)

    embeddings  = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": TOP_K_RETRIEVAL, "fetch_k": TOP_K_RETRIEVAL * 3},
    )
    return retriever, len(chunks)


# ─────────────────────────────────────────────
# 2.  Shared LLM (swap model/provider here)
# ─────────────────────────────────────────────

def get_llm(temperature: float = 0.0) -> ChatOllama:
    """
    Central LLM factory.  Change this one function to swap providers.
    Examples:
      • Anthropic : from langchain_anthropic import ChatAnthropic; return ChatAnthropic(model="claude-3-5-sonnet-20241022")
      • Groq      : from langchain_groq import ChatGroq; return ChatGroq(model="llama3-70b-8192")
      • Ollama    : from langchain_ollama import ChatOllama; return ChatOllama(model="llama3")
    """
    return ChatOllama(model="llama3.2", temperature=temperature)


# ─────────────────────────────────────────────
# 3.  Graph state
# ─────────────────────────────────────────────

class GraphState(TypedDict):
    question    : str
    documents   : List[Document]   # retrieved chunks
    generation  : str              # final answer
    retry_count : int              # query-rewrite attempts so far
    steps       : List[str]        # observable log for UI
    citations   : List[dict]       # [{source, page, snippet}, …]


# ─────────────────────────────────────────────
# 4.  Node: Retrieve
# ─────────────────────────────────────────────

def make_retrieve_node(retriever):
    def retrieve(state: GraphState) -> GraphState:
        state["steps"].append("🔍 **Retrieve** — searching vector store …")
        docs = retriever.invoke(state["question"])
        state["steps"].append(f"   Found **{len(docs)}** chunks")
        return {**state, "documents": docs}
    return retrieve


# ─────────────────────────────────────────────
# 5.  Node: Grade Documents (LLM-as-judge)
# ─────────────────────────────────────────────

GRADE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a relevance grader. Answer ONLY 'yes' or 'no'.\n"
     "'yes' = the document chunk contains information useful to answer the question.\n"
     "'no'  = the chunk is off-topic or irrelevant."),
    ("human", "Question: {question}\n\nDocument chunk:\n{document}"),
])

def make_grade_node():
    llm   = get_llm()
    chain = GRADE_PROMPT | llm | StrOutputParser()

    def grade_documents(state: GraphState) -> GraphState:
        state["steps"].append("⚖️  **Grade** — checking document relevance …")
        relevant = []
        for doc in state["documents"]:
            score = chain.invoke({
                "question" : state["question"],
                "document" : doc.page_content,
            }).strip().lower()
            if score == "yes":
                relevant.append(doc)

        kept = len(relevant)
        total = len(state["documents"])
        state["steps"].append(f"   Kept **{kept}/{total}** relevant chunks")
        return {**state, "documents": relevant}
    return grade_documents


# ─────────────────────────────────────────────
# 6.  Edge: decide after grading
# ─────────────────────────────────────────────

def decide_after_grade(state: GraphState) -> Literal["generate", "rewrite_query", "end_no_docs"]:
    if state["documents"]:
        return "generate"
    if state["retry_count"] < MAX_RETRIES:
        return "rewrite_query"
    return "end_no_docs"


# ─────────────────────────────────────────────
# 7.  Node: Rewrite Query
# ─────────────────────────────────────────────

REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a query optimizer for a RAG system. "
     "The original query failed to retrieve relevant documents. "
     "Rewrite it to be more specific, use different keywords, or decompose it. "
     "Return ONLY the rewritten query, nothing else."),
    ("human", "Original query: {question}"),
])

def make_rewrite_node():
    llm   = get_llm()
    chain = REWRITE_PROMPT | llm | StrOutputParser()

    def rewrite_query(state: GraphState) -> GraphState:
        new_q = chain.invoke({"question": state["question"]}).strip()
        state["steps"].append(f"✏️  **Rewrite** — new query: *{new_q}*")
        return {
            **state,
            "question"    : new_q,
            "retry_count" : state["retry_count"] + 1,
        }
    return rewrite_query


# ─────────────────────────────────────────────
# 8.  Node: Generate Answer
# ─────────────────────────────────────────────

GENERATE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant. Answer the question using ONLY the provided context.\n"
     "If the context does not contain enough information, say so honestly.\n"
     "Be concise and factual. Use bullet points where helpful."),
    ("human",
     "Context:\n{context}\n\n"
     "Question: {question}"),
])

def make_generate_node():
    llm   = get_llm(temperature=0.2)
    chain = GENERATE_PROMPT | llm | StrOutputParser()

    def generate(state: GraphState) -> GraphState:
        state["steps"].append("💡 **Generate** — producing answer …")
        context = "\n\n---\n\n".join(d.page_content for d in state["documents"])
        answer  = chain.invoke({"context": context, "question": state["question"]})

        # Build citations from document metadata
        citations = []
        for doc in state["documents"]:
            citations.append({
                "source"  : doc.metadata.get("source", "unknown"),
                "page"    : doc.metadata.get("page", 0) + 1,
                "snippet" : doc.page_content[:200] + "…" if len(doc.page_content) > 200 else doc.page_content,
            })

        return {**state, "generation": answer, "citations": citations}
    return generate


# ─────────────────────────────────────────────
# 9.  Node: Hallucination Check (faithfulness)
# ─────────────────────────────────────────────

HALLUCINATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a faithfulness checker. "
     "Determine if the answer is FULLY grounded in the provided context.\n"
     "Answer ONLY 'yes' (grounded) or 'no' (contains hallucinations / unsupported claims)."),
    ("human",
     "Context:\n{context}\n\n"
     "Answer:\n{generation}"),
])

def make_hallucination_node():
    llm   = get_llm()
    chain = HALLUCINATION_PROMPT | llm | StrOutputParser()

    def check_hallucination(state: GraphState) -> GraphState:
        state["steps"].append("🔬 **Hallucination Check** — verifying answer faithfulness …")
        context = "\n\n".join(d.page_content for d in state["documents"])
        verdict = chain.invoke({
            "context"    : context,
            "generation" : state["generation"],
        }).strip().lower()
        grounded = verdict == "yes"
        state["steps"].append(f"   Faithfulness: **{'✅ Grounded' if grounded else '❌ Hallucination detected'}**")
        # Store verdict in generation field metadata trick — we pass it via a flag
        state["_grounded"] = grounded  # type: ignore[index]
        return state
    return check_hallucination


def decide_after_hallucination(state: GraphState) -> Literal["end_answer", "generate"]:
    grounded     = state.get("_grounded", True)        # type: ignore[call-overload]
    retry_count  = state.get("retry_count", 0)
    if grounded or retry_count >= MAX_RETRIES:
        return "end_answer"
    # Force regeneration (increment retry so we don't loop forever)
    state["retry_count"] = retry_count + 1             # type: ignore[index]
    return "generate"


# ─────────────────────────────────────────────
# 10. Node: Graceful no-doc fallback
# ─────────────────────────────────────────────

def no_docs_fallback(state: GraphState) -> GraphState:
    state["steps"].append("⚠️  **No relevant documents found** after retries.")
    return {
        **state,
        "generation" : "I could not find relevant information in the uploaded documents to answer your question.",
        "citations"  : [],
    }


# ─────────────────────────────────────────────
# 11. Build the graph
# ─────────────────────────────────────────────

def build_graph(retriever) -> StateGraph:
    """
    Compile and return the runnable LangGraph agentic workflow.

    Flow:
      retrieve → grade_documents
        ├─(relevant)────────────────────► generate → hallucination_check
        │                                                   ├─(grounded)──► END
        │                                                   └─(not grounded)─► generate (retry)
        ├─(not relevant + retries left)─► rewrite_query → retrieve
        └─(not relevant + no retries)──► no_docs_fallback → END
    """
    wf = StateGraph(GraphState)

    wf.add_node("retrieve",            make_retrieve_node(retriever))
    wf.add_node("grade_documents",     make_grade_node())
    wf.add_node("rewrite_query",       make_rewrite_node())
    wf.add_node("generate",            make_generate_node())
    wf.add_node("hallucination_check", make_hallucination_node())
    wf.add_node("no_docs_fallback",    no_docs_fallback)

    wf.set_entry_point("retrieve")
    wf.add_edge("retrieve",        "grade_documents")
    wf.add_edge("rewrite_query",   "retrieve")

    wf.add_conditional_edges(
        "grade_documents",
        decide_after_grade,
        {
            "generate"      : "generate",
            "rewrite_query" : "rewrite_query",
            "end_no_docs"   : "no_docs_fallback",
        },
    )

    wf.add_edge("generate", "hallucination_check")

    wf.add_conditional_edges(
        "hallucination_check",
        decide_after_hallucination,
        {
            "end_answer" : END,
            "generate"   : "generate",
        },
    )

    wf.add_edge("no_docs_fallback", END)

    return wf.compile()


# ─────────────────────────────────────────────
# 12. Public helper — run a query end-to-end
# ─────────────────────────────────────────────

def run_query(graph, question: str) -> dict:
    """
    Invoke the compiled graph and return a clean result dict.

    Returns:
        {
            "answer"    : str,
            "citations" : list[dict],
            "steps"     : list[str],
        }
    """
    initial_state: GraphState = {
        "question"    : question,
        "documents"   : [],
        "generation"  : "",
        "retry_count" : 0,
        "steps"       : [],
        "citations"   : [],
    }
    final = graph.invoke(initial_state)
    return {
        "answer"    : final["generation"],
        "citations" : final.get("citations", []),
        "steps"     : final.get("steps", []),
    }