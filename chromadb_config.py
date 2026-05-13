from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from keybert import KeyBERT
import datetime
import os

from extract_text import extract_and_clean_pdf_text

def generate_tags(text_sample: str) -> str:
    kw_model = KeyBERT(model="all-MiniLM-L6-v2")
    keywords = kw_model.extract_keywords(
        text_sample[:2000],
        keyphrase_ngram_range=(1, 2),
        stop_words="english",
        top_n=8,
    )
    tags = ", ".join([kw for kw, _score in keywords])
    return tags

def create_chunks(pdf_path):
    text = extract_and_clean_pdf_text(pdf_path)

    filename = os.path.basename(pdf_path)
    title = os.path.splitext(filename)[0].replace("_", " ").title()
    timestamp = datetime.datetime.now().isoformat()
    tags = generate_tags(text)
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        separators=["\n\n", "\n", " "]
    )
    raw_chunks = splitter.split_text(text)
    
    documents = []
    for i, chunk in enumerate(raw_chunks):
        doc = Document(
            page_content=chunk,
            metadata={
                "title": title,
                "source": pdf_path,
                "tags": tags,
                "page": i,
                "timestamp": timestamp,
            }
        )
        documents.append(doc)

    print(f"Number of chunks: {len(documents)}")
    return documents

def store_in_database(documents):
    embedding_model = OllamaEmbeddings(model="nomic-embed-text")

    db = Chroma.from_documents(
        documents, 
        embedding=embedding_model,
        persist_directory="./chroma_db",
        collection_metadata={"hnsw:space": "cosine"}
    )

    total_docs = db._collection.count()
    return total_docs