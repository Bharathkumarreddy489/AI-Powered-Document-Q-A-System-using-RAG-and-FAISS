import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore")

#API Key
load_dotenv()  # Loads variables from .env file

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Google API Key not found. Please check your .env file.")

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

FOLDER_PATH = r"C:\Users\BHARATH KUMAR REDDY\Desktop\Merit Assign\LangChain"   # put all txt/pdf files inside this folder

#load documents
documents = []

for file_name in os.listdir(FOLDER_PATH):
    file_path = os.path.join(FOLDER_PATH, file_name)

    if file_name.endswith(".txt"):
        loader = TextLoader(file_path, encoding="utf-8")
        documents.extend(loader.load())

    elif file_name.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
        documents.extend(loader.load())


#split the documuents
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(documents)

#hugging face embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# FAISS VECTOR DATABASE
vector_db = FAISS.from_documents(chunks, embeddings)
retriever = vector_db.as_retriever(search_kwargs={"k": 3})


# Google Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-001",
    temperature=0.2
)

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an expert AI assistant specialized in financial technology and business platforms. 
Answer the user’s question ONLY using the information provided in the context. 

Context:
{context}

Question:
{question}

Instructions:
- Provide concise, clear, and accurate answers.
- If the answer is not in the context, respond: "Information not available in the provided context."
- Use bullet points if listing multiple items.

Answer:
"""
)
parser = StrOutputParser()


#LCEL RAG CHAIN
rag_chain = (
    {
        "context": retriever | (lambda docs: "\n\n".join([d.page_content for d in docs])),
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | parser
)

# TEST QUERY
query = "What is the purpose of the Intelligent Decisioning Engine (IDE)?"
print(rag_chain.invoke(query))
