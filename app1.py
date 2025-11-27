import streamlit as st
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(page_title="RAG Chatbot", layout="wide")

# -------------------------
# LOAD API KEY
# -------------------------
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# -------------------------
# SESSION STATE INIT
# -------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

# -------------------------
# CACHE RESOURCES
# -------------------------
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource
def build_vector_db(_docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(_docs)

    embeddings = load_embeddings()
    return FAISS.from_documents(chunks, embeddings)


# -------------------------
# UI
# -------------------------
st.title("📘 Continuous RAG Chatbot")

uploaded_files = st.file_uploader(
    "Upload PDF/TXT files",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

# -------------------------
# PROCESS UPLOADED FILES
# -------------------------
if uploaded_files and st.session_state.vector_db is None:
    documents = []

    with st.spinner("⏳ Loading & processing documents..."):
        for f in uploaded_files:
            path = f"./temp_{f.name}"
            with open(path, "wb") as file:
                file.write(f.read())

            if f.name.endswith(".txt"):
                loader = TextLoader(path, encoding="utf-8")
            else:
                loader = PyPDFLoader(path)

            documents.extend(loader.load())

        st.session_state.vector_db = build_vector_db(documents)

    st.success("Documents loaded successfully! You can start chatting now.")


# -------------------------
# CHAT INTERFACE
# -------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


# -------------------------
# USER INPUT
# -------------------------
user_input = st.chat_input("Ask a question about your documents...")

if user_input:
    # Show user message
    st.chat_message("user").write(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    if st.session_state.vector_db is None:
        st.warning("Please upload documents first.")
    else:
        retriever = st.session_state.vector_db.as_retriever(search_kwargs={"k": 3})

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-001",
            temperature=0.2
        )

        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""
You are an AI assistant. Answer the question ONLY using the provided context.

Context:
{context}

Question:
{question}

If answer is not in the context, say:
"Information not available in the provided context."

Answer:
"""
        )
        parser = StrOutputParser()

        rag_chain = (
            {
                "context": retriever | (lambda docs: "\n\n".join([d.page_content for d in docs])),
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
            | parser
        )

        with st.chat_message("assistant"):
            with st.spinner("🤖 Thinking..."):
                answer = rag_chain.invoke(user_input)
                st.write(answer)

        # Save assistant reply
        st.session_state.messages.append({"role": "assistant", "content": answer})
