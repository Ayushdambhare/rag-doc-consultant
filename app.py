import os
import streamlit as st
from dotenv import load_dotenv

from ingest_data import load_documents, scrape_web_page
from rag_core import RAGCore

# --- Environment Setup ---
load_dotenv()
if "HUGGINGFACEHUB_API_TOKEN" not in os.environ:
    st.error("Hugging Face API token is not set! Please create a .env file with HUGGINGFACEHUB_API_TOKEN='your_token'.")
    st.stop()
    
# --- Page Configuration ---
st.set_page_config(page_title="AI Tech Doc Consultant", page_icon="🤖", layout="wide")

# --- Session State Initialization ---
def initialize_session_state():
    if 'rag_core' not in st.session_state:
        st.session_state.rag_core = RAGCore()
    
    if 'chain_ready' not in st.session_state:
        # Check if vector store exists and try to load it
        if st.session_state.rag_core.load_existing_vectorstore():
            st.session_state.rag_core.setup_qa_chain()
            st.session_state.chain_ready = True
        else:
            st.session_state.chain_ready = False

    if 'messages' not in st.session_state:
        st.session_state.messages = []

initialize_session_state()

# --- UI Rendering ---
st.title("Advanced RAG-Powered AI Consultant 🤖")
st.markdown("Welcome! This AI assistant can answer questions about technical documents. Ingest data to begin.")

# --- Sidebar for Data Ingestion ---
with st.sidebar:
    st.header("Ingest Your Data")
    
    uploaded_files = st.file_uploader(
        "Upload PDF, TXT, or MD files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True
    )
    url_to_scrape = st.text_input("Or scrape a technical documentation website")

    if st.button("Ingest Documents"):
        if not uploaded_files and not url_to_scrape:
            st.warning("Please upload files or provide a URL to ingest.")
        else:
            with st.spinner("Ingesting documents... This may take a while."):
                all_docs = []
                
                if uploaded_files:
                    temp_dir = "temp_uploaded"
                    if not os.path.exists(temp_dir):
                        os.makedirs(temp_dir)
                    for uploaded_file in uploaded_files:
                        file_path = os.path.join(temp_dir, uploaded_file.name)
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                    all_docs.extend(load_documents(temp_dir))

                if url_to_scrape:
                    all_docs.extend(scrape_web_page(url_to_scrape))
                
                st.session_state.rag_core.ingest_documents(all_docs)
                st.session_state.rag_core.setup_qa_chain()
                st.session_state.chain_ready = True
                st.success("Ingestion complete! You can now ask questions.")

    st.divider()
    if st.session_state.chain_ready:
        st.success("Data has been ingested. The QA system is ready.")
    else:
        st.warning("No data ingested yet. Please upload documents.")

# --- Main Chat Interface ---
st.header("Ask Your Questions")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about your documents"):
    if not st.session_state.chain_ready:
        st.warning("Please ingest documents before asking questions.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.rag_core.ask_question(prompt)
                answer = response.get('answer', 'Sorry, I could not find an answer.')
                st.markdown(answer)
                
                if 'source_documents' in response and response['source_documents']:
                    with st.expander("View Sources"):
                        for doc in response['source_documents']:
                            st.markdown(f"**Source:** `{doc.metadata.get('source', 'Unknown')}`")
                            st.markdown(f"> {doc.page_content.strip()}")
                            st.divider()

        st.session_state.messages.append({"role": "assistant", "content": answer})