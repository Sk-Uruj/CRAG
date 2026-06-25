from __future__ import annotations

import os
import streamlit as st

from backend.config import settings, ensure_directories
from backend.ingestion import ingest_directory
from backend.chunking import chunk_documents
from backend.embeddings import GeminiEmbedder
from backend.vectorstore import ChromaVectorStore
from backend.retriever import ChromaRetriever
from backend.evaluator import RetrievalEvaluator
from backend.generator import ResponseGenerator
from backend.orchestrator import CRAGOrchestrator
from backend.models import RetrievalLabel

st.set_page_config(
    page_title="ThoughtsTracer AI - CRAG Dashboard",
    page_icon="🚦",
    layout="wide",
)

ensure_directories()

st.title("🚦 ThoughtsTracer AI")
st.subheader("Corrective Retrieval-Augmented Generation (CRAG) Engine")
st.caption("A real Chroma-backed retrieval pipeline with Gemini-powered quality-control routing.")
st.markdown("---")

if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
    st.warning("⚠️ GOOGLE_API_KEY (or GEMINI_API_KEY) not found. Embeddings/evaluation will fail until configured.")


@st.cache_resource
def init_engine():
    embedder = GeminiEmbedder()
    vectorstore = ChromaVectorStore()
    retriever = ChromaRetriever(embedder=embedder, vectorstore=vectorstore, top_k=settings.TOP_K)
    evaluator = RetrievalEvaluator()
    generator = ResponseGenerator()
    orchestrator = CRAGOrchestrator(retriever, evaluator, generator)
    return embedder, vectorstore, orchestrator


embedder, vectorstore, orchestrator = init_engine()

left, right = st.columns([1, 1])

with left:
    st.markdown("## 1) Ingest Documents")
    st.write(f"Raw docs folder: `{settings.RAW_DIR}`")
    uploaded = st.file_uploader("Upload .txt, .md, or .docx files", type=["txt", "md", "docx"], accept_multiple_files=True)

    if st.button("📥 Index Documents"):
        if uploaded:
            for file in uploaded:
                save_path = settings.RAW_DIR / file.name
                save_path.write_bytes(file.getbuffer())

        docs = ingest_directory(settings.RAW_DIR)
        if not docs:
            st.error("No documents found in data/raw.")
        else:
            chunks = chunk_documents(docs, chunk_size=settings.CHUNK_SIZE, overlap=settings.CHUNK_OVERLAP)
            texts = [c.text for c in chunks]
            embeddings = embedder.embed_texts(texts)
            vectorstore.reset()
            vectorstore.add_chunks(chunks, embeddings)
            st.success(f"Indexed {len(chunks)} chunks from {len(docs)} documents.")
            st.info(f"Chroma collection count: {vectorstore.count()}")

with right:
    st.markdown("## 2) Ask a Question")
    st.write("Try a query like: **What is the company leave policy?**")
    query = st.text_input("Enter your question", placeholder="Ask about the uploaded docs...")

    if st.button("🚀 Run CRAG Pipeline") and query:
        result = orchestrator.run(query)
        eval_data = result["evaluation"]
        status = eval_data.get("label", "AMBIGUOUS")
        confidence = eval_data.get("confidence", 0.0)

        if isinstance(status, RetrievalLabel):
            status = status.value
        elif isinstance(status, str) and "RetrievalLabel." in status:
            status = status.replace("RetrievalLabel.", "")

        st.markdown("### 🚦 Routing Status")
        if status == "CORRECT":
            st.success(f"🟢 GATE VERDICT: {status} (Confidence: {confidence})")
        elif status == "INCORRECT":
            st.error(f"🔴 GATE VERDICT: {status} (Confidence: {confidence})")
        else:
            st.warning(f"🟡 GATE VERDICT: {status} (Confidence: {confidence})")

        with st.expander("🔍 Evaluator Reasons", expanded=True):
            for reason in eval_data.get("reasons", []):
                st.write(f"- {reason}")

        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Retrieved Chunks")
            for chunk in result.get("retrieved_chunks", []):
                st.info(
                    f"📄 **Source:** {chunk.get('source')} | **Chunk:** {chunk.get('chunk_id')}

"
                    f"{chunk.get('text', '')}"
                )

            if status in ["INCORRECT", "AMBIGUOUS"]:
                st.markdown("---")
                st.markdown("**Corrective Query:**")
                st.code(eval_data.get("rewritten_query", "(none)"))

                st.markdown("**Corrective Web Context:**")
                for w_chunk in result.get("web_chunks", []):
                    st.warning(w_chunk.get("text", ""))

        with col2:
            st.markdown("### Final Grounded Output")
            st.text_area(
                "Answer Trace",
                value=result.get("generation", {}).get("answer", ""),
                height=220,
                disabled=True,
            )
            st.markdown("**Provenance:**")
            st.json(result.get("generation", {}).get("sources", []))

            with st.expander("JSON Payload"):
                st.json(result)
