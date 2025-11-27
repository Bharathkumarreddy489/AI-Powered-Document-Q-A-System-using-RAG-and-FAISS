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
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Fast RAG Chatbot", layout="wide")

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")


# -----------------------------
# CACHING (MAJOR SPEED BOOST)
# -----------------------------

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource
def build_vector_db(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embeddings = load_embeddings()
    return FAISS.from_documents(chunks, embeddings)


# -----------------------------
# Streamlit UI
# -----------------------------

st.title("⚡ Fast RAG Chatbot (Gemini + FAISS)")

uploaded_files = st.file_uploader(
    "Upload PDF/TXT files",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

documents = []

if uploaded_files:
    for f in uploaded_files:
        path = f"./temp_{f.name}"
        with open(path, "wb") as file:
            file.write(f.read())

        if f.name.endswith(".txt"):
            loader = TextLoader(path, encoding="utf-8")
        else:
            loader = PyPDFLoader(path)

        documents.extend(loader.load())

    with st.spinner("🔍 Building Vector Database... (only first time)"):
        vector_db = build_vector_db(documents)

    retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    # Gemini Model
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001", temperature=0.2)

    # Prompt
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
Answer the question ONLY using the context.

Context:
{context}

Question:
{question}

If answer is not found, say:
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

    question = st.text_input("Ask something:")

    if question:
        with st.spinner("🤖 Thinking..."):
            answer = rag_chain.invoke(question)
        st.success(answer)

else:
    st.info("Upload documents to begin.")
