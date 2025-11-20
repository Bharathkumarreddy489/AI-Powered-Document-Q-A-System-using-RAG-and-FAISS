# AI-Powered Document Q&A System using RAG and FAISS

This project demonstrates a **Retrieval-Augmented Generation (RAG)** pipeline using **FAISS** for vector search and **Google Gemini** as the language model. It allows you to query multiple documents and get accurate answers based on their content. The system is designed for **efficient knowledge retrieval from large document collections**, making it ideal for internal knowledge bases, research papers, or corporate documentation.

---

## Features

- Load multiple `.txt` and `.pdf` documents.
- Split documents into chunks for better context handling.
- Generate embeddings using **HuggingFace sentence-transformers**.
- Store embeddings in **FAISS** vector database for fast retrieval.
- Answer questions using **Google Gemini LLM** with context from documents.
- Uses **LCEL (LangChain Expression Language)** for a modern RAG workflow.
- Supports multi-document querying and context-aware responses.

---

## Requirements

- Python 3.9+
- `langchain`, `langchain-google-genai`, `langchain-huggingface`, `langchain-community`
- `sentence-transformers`
- `faiss-cpu` or `faiss-gpu`
- `python-dotenv` (for API key management)

---

## Setup

1. **Clone this repository**  
   `git clone https://github.com/Bharathkumarreddy489/AI-Powered-Document-Q-A-System-using-RAG-and-FAISS.git`  

2. **Install the required Python packages**  
   `pip install langchain langchain-google-genai langchain-huggingface langchain-community sentence-transformers faiss-cpu python-dotenv`

3. **Create a `.env` file** in the project root and add your Google API key:  
   `GOOGLE_API_KEY=YOUR_GOOGLE_API_KEY`

4. **Place your `.txt` and `.pdf` documents** inside a folder named `docs`.

5. **Open and run** the notebook or script:  
   `AI-Powered Document Q&A System using RAG and FAISS.ipynb` or `main.py`

---

## Usage

Once the setup is complete:

- Load and split your documents into chunks.
- Generate embeddings using HuggingFace models.
- Store embeddings in FAISS vector database.
- Use the RAG chain with Google Gemini LLM to ask questions and get answers based on document content.
- Ideal for **internal documentation search, research, or AI-assisted knowledge retrieval**.

---

## Project Value

This project simplifies querying multiple documents by combining **vector search** with **state-of-the-art generative AI**, enabling users to:

- Quickly find information in large collections of text or PDF files.
- Get context-aware answers instead of plain keyword search results.
- Build AI-driven knowledge assistants or internal helpdesk systems.

---

## License

This project is open-source and available under the MIT License.
