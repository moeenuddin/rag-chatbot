
import streamlit as st
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import pipeline
import tempfile

st.title("📚 RAG Chatbot")

uploaded_file = st.file_uploader("Upload a .txt file", type="txt")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
        tmp.write(uploaded_file.read())
        filepath = tmp.name

    loader = TextLoader(filepath)
    docs = loader.load()
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs_split = splitter.split_documents(docs)

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs_split, embedding_model)

    generator = pipeline("text2text-generation", model="google/flan-t5-small")

    query = st.text_input("Ask a question based on the document")

    if query:
        docs_relevant = vectorstore.similarity_search(query, k=2)
        context = " ".join([doc.page_content for doc in docs_relevant])
        prompt = f"Answer the question based on the context:\n\nContext: {context}\n\nQuestion: {query}"
        response = generator(prompt, max_new_tokens=200)[0]["generated_text"]

        st.markdown("### 📖 Answer")
        st.write(response)
