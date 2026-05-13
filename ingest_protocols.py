import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# CONFIG
PDF_PATH = "Assets/protocols.pdf" # Make sure your PDF is named this
DB_PATH = "protocol_db"

def ingest():
    print("🚑 Starting Protocol Ingestion...")
    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()
    
    # Split the 435 pages into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)
    
    # Create searchable 'embeddings' (This is the 'brain' part)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Build and save the local database
    vector_db = FAISS.from_documents(chunks, embeddings)
    vector_db.save_local(DB_PATH)
    print(f"✅ Protocol DB saved to {DB_PATH}")

if __name__ == "__main__":
    ingest()
    