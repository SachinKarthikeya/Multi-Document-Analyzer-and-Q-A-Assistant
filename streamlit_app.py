from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
import streamlit as st

from chromadb_config import create_chunks, store_in_database

PROMPT_TEMPLATE = """
You are a professional document analyzer and Q&A assistant. 
Based on the context from the document chunks, answer the question as accurately as possible. 
Base your response solely on the provided context from the document chunks. DO NOT use any external information or make assumptions.

Document Chunks:
{chunks}

Question: {question}

Guidelines while answering:
1. Provide a concise and accurate answer based on the document chunks.
2. If the context directly answers the question, provide that answer with key details.
3. If the context has partial or related information, summarize the relevant points and note the limitations 
(e.g., "Based on the documents, here's what relates to your query...").
4. Strictly say "I don't know" if the document chunks contain ZERO relevant information to answer the question.
5. Refuse sensitive/personal/confidential information politely. 
"""

def hybrid_retrieval(documents, top_k=5):
    embedding_model = OllamaEmbeddings(model="nomic-embed-text")

    db = Chroma(
        persist_directory="./chroma_db", 
        embedding_function=embedding_model
    )
    vector_retriever = db.as_retriever(search_kwargs={"k": top_k})
    
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = top_k

    ensemble_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever], 
        weights=[0.5, 0.5]
    )
    return ensemble_retriever

def retrieve_relevant_chunks(question, documents, top_k=5):
    retriever = hybrid_retrieval(documents, top_k)
    retrieved_docs = retriever.invoke(question)

    context_text = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(chunks=context_text, question=question)

    llm = OllamaLLM(model="llama3.2:1b", temperature=0)
    response = llm.invoke(prompt)

    sources = list({doc.metadata.get("source", "Unknown") for doc in retrieved_docs})
    titles = list({doc.metadata.get("title", "Unknown") for doc in retrieved_docs})
    tags = list({doc.metadata.get("tags", "") for doc in retrieved_docs})

    formatted_response = (
        f"{response}\n\n"
        f"**Sources**: {', '.join(sources)}\n"
        f"**Titles**: {', '.join(titles)}\n"
        f"**Tags**: {', '.join(tags)}\n"
        f"**Chunks Retrieved**: {len(retrieved_docs)}"
    )
    return formatted_response

if __name__ == "__main__":
    st.title("Multi-Document Analyzer and Q&A Assistant")
    st.write("Upload your PDF document path and ask questions based on its content.")

    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

    if "documents" not in st.session_state:
        st.session_state.documents = None

    if "db_created" not in st.session_state:
        st.session_state.db_created = False

    if uploaded_file is not None:
        if not st.session_state.db_created:
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
        
            with st.spinner("Processing the document..."):
                documents = create_chunks("temp.pdf")
                total_docs = store_in_database(documents)

                st.session_state.documents = documents
                st.session_state.db_created = True

                st.success("Document processed successfully!")

        query = st.text_input("Enter your question:")
        if query:
            with st.spinner("Retrieving answer..."):
                answer = retrieve_relevant_chunks(query, st.session_state.documents)
                st.markdown(answer)