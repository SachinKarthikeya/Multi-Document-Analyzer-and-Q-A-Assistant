# 📄 Multi-Document Analyzer and Q&A Assistant

An LLM-powered document analysis and Q&A system that lets user upload any type of PDF document (research, medical, financial, legal, etc)
and answer his questions related to the uploaded document. This project is heavily relied on **RAG Pipeline** with **Hybrid Retrieval** 
method for efficient question-answering based conversation.

## 🚀 Features

- **PyMuPDF4LLM**: Extracts raw text for cleaning and pre-processing into structured text.
- **LangChain**: Primary framework for managing project workflow
- **RecursiveCharacterTextSplitter**: Responsible for splitting text into chunks based on size and overlapping
- **KeyBERT**: Transformer model to generate tags by analyzing the text
- **Metadata**: Features added along with chunks for efficient retrieval.
- **Nomic-Embed-Text**: Converts the chunks and metadata into vector embeddings
- **ChromaDB**: Vector database to store the embeddings 
- **Hybrid Retrieval**: Combination of BM25(keyword-based) + Vector retrieval to fetch top related chunks 
- **Llama3.2:1b**: Combines System Prompt + Retrieved Chunks to generate a contextual answer
- **Streamlit**: For an interactive dashboard for users

## 📄 Workflow

- User uploads a PDF document which he wants to analyze and ask his queries.
- PyMuPDF4LLM extracts the raw text and cleans and pre-processes using Regex into structured text.
- LangChain's sub-module RecursiveCharacterTextSplitter splits the text into certain chunk size and controls overlapping between
  chunks to preserve context.
- KeyBERT model analyzes the text to generate relevant tags for database storage
- Metadata like Title, Source, PageNo and TimeStamp are created and added along with tags and chunks to convert them into vector
  embeddings.
- The embeddings are then stored into ChromaDB.
- After this process is complete, user is allowed to ask his questions.
- After he enters his question, the question is also converted into vector embeddings.
- Using these embeddings, Vector retriever along with BM25 retriever search for top-k relevant chunks from the database.
- Ensemble Retriever combines the searched chunks of both retrievers by equally dividing the context of searched chunks.
- Llama3.2:1b combines the system prompt, retrieved chunks along with filtering metadata to generate a contextual answer for the
  user's question.
- The generated answer is displayed to the user via Streamlit UI along with title, tags, source document of the answer and No.of
  chunks retrieved for the answer.

## 🧰 Tech Stack

- **Frontend**: Streamlit
- **Backend**: LangChain, PyMuPDF4LLM, Regex
- **LLM**: Llama3.2:1b
- **Models**: Nomic-Embed-Text, KeyBERT
- **Database**: ChromaDB

## 📁 Versions

For this project, there are two versions which work separately:

**Version 1**: Initial version which works on local machine's Terminal

**Version 2**: Upgraded version for which Streamlit UI is built for users

## 📢 Future Enhancements

- Automatically clears the database to avoid memory and retrieval issues
- Connecting to external tools such as Regular Databases, Web, APIs for in-depth answering  
- Integrating a summarizer model if the user wants to summarize the whole document at once. 
