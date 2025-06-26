import os
from typing import List, Dict, Any

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain.memory import ConversationBufferMemory

# --- NEW/UPDATED IMPORTS ---
from langchain_huggingface import ChatHuggingFace 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- Constants ---
LLM_REPO_ID = "mistralai/Mixtral-8x7B-Instruct-v0.1" 
VECTORSTORE_DIR = "vectorstore"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# --- RAG Prompt Template for a CHAT model ---
RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an expert technical assistant. Use the provided context to answer the user's question. If you don't know the answer, state that you don't know. Do not try to make up an answer."),
        ("human", "CONTEXT:\n{context}\n\nCHAT HISTORY:\n{chat_history}\n\nQUESTION:\n{question}\n\nANSWER:"),
    ]
)

class RAGCore:
    def __init__(self):
        """Initializes the RAG Core components."""
        self.vector_store = None
        self.qa_chain = None
        self.retriever = None
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        
        # 1. Define the base endpoint LLM
        llm_endpoint = HuggingFaceEndpoint(
            repo_id=LLM_REPO_ID,
            huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
            temperature=0.5,
            max_new_tokens=512,
        )

        # 2. Wrap the endpoint in the ChatHuggingFace class

        self.llm = ChatHuggingFace(llm=llm_endpoint)

    def load_existing_vectorstore(self):
        """Loads the vector store and initializes the retriever."""
        if os.path.exists(VECTORSTORE_DIR):
            print("Loading existing vector store...")
            self.vector_store = Chroma(persist_directory=VECTORSTORE_DIR, embedding_function=self.embedding_function)
            self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 4})
            print("Vector store loaded and retriever is ready.")
            return True
        print("Vector store not found.")
        return False

    def ingest_documents(self, source_documents: List[Dict[str, Any]]):
        """Processes and ingests documents into a new vector store."""
        if not source_documents: return
        docs_to_process = [Document(page_content=doc['content'], metadata={'source': doc['source']}) for doc in source_documents]
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunked_docs = text_splitter.split_documents(docs_to_process)
        self.vector_store = Chroma.from_documents(documents=chunked_docs, embedding=self.embedding_function, persist_directory=VECTORSTORE_DIR)
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 4})
        print("Ingestion complete. Retriever is ready.")

    def _format_docs(self, docs: List[Document]) -> str:
        """Helper function to format retrieved documents into a single string."""
        return "\n\n".join(f"Source: {doc.metadata.get('source', 'Unknown')}\nContent: {doc.page_content}" for doc in docs)

    def setup_qa_chain(self):
        """Sets up the conversational QA chain using LCEL."""
        if not self.retriever:
            print("Retriever is not available. Cannot set up QA chain.")
            return

        def get_chat_history(inputs):
            return self.memory.load_memory_variables(inputs).get("chat_history", [])

        # The new LCEL chain definition
        self.qa_chain = (
            {
                "context": lambda x: self._format_docs(self.retriever.invoke(x["question"])),
                "question": lambda x: x["question"],
                "chat_history": get_chat_history,
            }
            | RAG_PROMPT
            | self.llm
            | StrOutputParser()
        )
        print("LCEL QA Chain is ready.")

    def ask_question(self, question: str) -> Dict[str, Any]:
        """Asks a question to the QA chain, manages memory, and returns the response."""
        if not self.qa_chain:
            return {"answer": "The QA system is not initialized."}

        relevant_docs = self.retriever.invoke(question)
        
        # The chain now correctly manages the history via the lambda function
        answer = self.qa_chain.invoke({"question": question})
        
        # Manually save context to memory
        self.memory.save_context({"question": question}, {"answer": answer})
        
        return {"answer": answer, "source_documents": relevant_docs}