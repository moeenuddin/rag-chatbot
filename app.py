
import streamlit as st
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import pipeline
import tempfile
import os

st.title("📚 RAG Chatbot")


st.subheader("📄 Download a Sample File (or Upload Your Own)")

books_dir = "books"
if not os.path.exists(books_dir):
    st.error("❌ 'books/' directory not found. Please make sure it exists and contains .txt files.")
else:
    # List .txt files in books directory
    txt_files = [f for f in os.listdir(books_dir) if f.endswith(".txt")]

    if txt_files:
        for file in txt_files:
            file_path = os.path.join(books_dir, file)
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                st.download_button(label=f"⬇️ Download {file}", data=content, file_name=file)
    else:
        st.info("No .txt files found in the 'books/' directory.")
        
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
        docs_relevant = vectorstore.similarity_search(query, k=10)
        context = " ".join([doc.page_content for doc in docs_relevant])
        prompt = f"Answer the question based on the context:\n\nContext: {context}\n\nQuestion: {query}"
        response = generator(prompt, max_new_tokens=200)[0]["generated_text"]

        st.markdown("### 📖 Answer")
        st.write(response)
