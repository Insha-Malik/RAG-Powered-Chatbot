# task4_streamlit_gemini.py
import streamlit as st
import google.generativeai as genai
import os
import tempfile
from typing import List

# Page config
st.set_page_config(page_title="Gemini RAG AI Chat", page_icon="🤖", layout="wide")

# Dark mode CSS
st.markdown("""
<style>
/* Main content area */
[data-testid="stAppViewContainer"] {
    background-color: #000000;  /* black */
    color: #ffffff;             /* white text */
    padding: 20px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #000000;  /* black */
    color: #ffffff;
    padding: 20px;
}

/* Sidebar headers and text */
[data-testid="stSidebar"] h1, 
[data-testid="stSidebar"] h2, 
[data-testid="stSidebar"] h3, 
[data-testid="stSidebar"] label {
    color: #ffffff;
}

/* Chat messages */
.user {
    background: #111111;        /* dark grey for messages */
    color: #ffffff;
    padding:12px;
    border-radius:15px;
    margin-bottom:10px;
    box-shadow: none;
    max-width:70%;
}

.ai {
    background: #222222;        /* slightly lighter dark grey */
    color: #ffffff;
    padding:12px;
    border-radius:15px;
    margin-bottom:10px;
    box-shadow: none;
    max-width:70%;
}

/* Align messages */
div.stMarkdown div.user {
    margin-left: auto;
}
div.stMarkdown div.ai {
    margin-right: auto;
}

/* Buttons */
.stButton>button {
    background-color: #111111 !important;
    color: #ffffff !important;
    border: 1px solid #444444 !important;
}

/* Slider (temperature) */
.css-1aumxhk {  /* slider track */
    background-color: #555555 !important;
}
.css-1aumxhk .stSlider>div>div>div>div { 
    background-color: #888888 !important; /* handle */
}
</style>
""", unsafe_allow_html=True)

# Session state init
if "docstore" not in st.session_state:
    st.session_state.docstore: List[str] = []

if "messages" not in st.session_state:
    st.session_state.messages: List[dict] = []

# Sidebar: Settings + KB Upload
with st.sidebar:
    st.header("⚙ Settings")
    api_key = st.text_input(" Gemini API Key", type="password")
    model_name = st.selectbox(
        "Model",
        [
            "models/gemini-2.5-flash",
            "models/gemini-flash-latest"
        ]
    )
    temperature = st.slider("🌡 Temperature", 0.0, 1.0, 0.7, 0.1)
    max_tokens = st.number_input(" Max Tokens", min_value=50, max_value=2000, value=500, step=50)

    st.markdown("---")
    st.subheader("Knowledge Base")
    uploaded_files = st.file_uploader("Upload PDF / TXT", accept_multiple_files=True)
    if st.button(" Add files"):
        if uploaded_files:
            for uf in uploaded_files:
                fname = uf.name.lower()
                with tempfile.NamedTemporaryFile(delete=False) as tmp:
                    tmp.write(uf.getvalue())
                    tmp_path = tmp.name
                text = ""
                if fname.endswith(".pdf"):
                    from pypdf import PdfReader
                    reader = PdfReader(tmp_path)
                    text = "\n".join([p.extract_text() or "" for p in reader.pages])
                else:
                    with open(tmp_path, "r", encoding="utf-8", errors="ignore") as f:
                        text = f.read()
                if text.strip():
                    st.session_state.docstore.append(text)
                os.remove(tmp_path)
            st.success(f" Added {len(uploaded_files)} file(s) to KB.")
        else:
            st.warning("No files uploaded.")

    if st.button(" Clear KB"):
        st.session_state.docstore = []
        st.success("Knowledge base cleared.")

    st.markdown("---")
    if st.button(" Clear Chat"):
        st.session_state.messages = []
        st.success("Chat cleared.")

# API key missing
if not api_key:
    st.warning("Please enter your Gemini API key in the sidebar.")
    st.stop()

# Configure Gemini API
genai.configure(api_key=api_key)
llm = genai.GenerativeModel(model_name)

# Simple Semantic Search
def semantic_search(query, top_k=3):
    if not st.session_state.docstore:
        return []
    query_words = query.lower().split()
    scored_docs = []
    for doc in st.session_state.docstore:
        score = sum(doc.lower().count(word) for word in query_words)
        scored_docs.append((score, doc))
    scored_docs.sort(reverse=True, key=lambda x: x[0])
    top_docs = [doc for score, doc in scored_docs if score > 0]
    return top_docs[:top_k] if top_docs else ["No relevant info found."]

# RAG Pipeline
def rag_pipeline(question):
    context_docs = semantic_search(question)
    context = "\n".join(context_docs)
    prompt = f"""
You are a helpful AI assistant. Use the following context to answer the question.

Context: {context}

Question: {question}

Answer in a clear and human-like way:
"""
    response = llm.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=int(max_tokens)
        )
    )
    return response.text

# Main Chat UI
st.title("🤖 Gemini RAG AI Chat")

user_input = st.text_area(" Ask your question:", height=120)
if st.button("Ask") and user_input:
    answer = rag_pipeline(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.messages.append({"role": "assistant", "content": answer})

# Display chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"<div class='user'> <b>You:</b> {msg['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='ai'> <b>AI:</b> {msg['content']}</div>", unsafe_allow_html=True)