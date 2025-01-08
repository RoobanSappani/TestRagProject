import os
import streamlit as st
from langchain.vectorstores import FAISS

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

import google.generativeai as genai
from PIL import Image

genai.configure(api_key="AIzaSyCOXgeOwojSCvtbl3PN6dHkMpUYrrSuz-E")
model = genai.GenerativeModel("gemini-1.5-flash")

# Step 1: Load PDFs and Extract Text
def load_pdfs(pdf_dir):
	documents = []
	for file_name in os.listdir(pdf_dir):
		if file_name.endswith(".pdf"):
			loader = PyPDFLoader(os.path.join(pdf_dir, file_name))
			documents.extend(loader.load())
	return documents

# Step 2: Chunk the Documents
def chunk_documents(documents):
	text_splitter = RecursiveCharacterTextSplitter(
					chunk_size=1000,
					chunk_overlap=200)
	return text_splitter.split_documents(documents)

# Step 3: Create Vector Database
def create_vector_database(chunks, embeddings_model):
	vector_db = FAISS.from_documents(chunks, embeddings_model)
	return vector_db

# Step 4: Query Vector Database
def query_vector_database(vector_db, query, top_k=3):
	docs_with_scores = vector_db.similarity_search_with_score(query, k=top_k)
	return docs_with_scores

# Directory containing PDFs
pdf_directory = "./"

# Step 1: Load PDFs
print("Loading PDFs...")
raw_documents = load_pdfs(pdf_directory)

# Step 2: Chunk Documents
print("Chunking documents...")
chunks = chunk_documents(raw_documents)

# Step 3: Generate Vector Database
@st.cache_resource()  # This will cache the vector database
def get_vector_database():
    print("Creating vector database...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = create_vector_database(chunks, embeddings)
    return vector_db

vector_db = get_vector_database()

col1, col2 = st.columns(2)
image = Image.open('logo.png')
image = image.resize((50, 50))

# Streamlit UI
with col1:
	st.markdown("<h1 style='margin: 0;'>MeritoBuddy AI</h1>", unsafe_allow_html=True)
with col2:
	# st.image(image)

	st.markdown(
        """
        <div style="text-align: right;">
            <img src="https://raw.githubusercontent.com/RoobanSappani/TestRagProject/refs/heads/main/logo.png" alt="Logo" style="width: 50px; height: auto;">
        </div>
        """,
	    unsafe_allow_html=True,
	)

# Query input
query = st.text_input("Enter your query:")
if query:
	matches = query_vector_database(vector_db, query)
	
	context = ""	

	for i, (doc, score) in enumerate(matches, start=1):

		context += f"\n{i}. "
		context += doc.page_content
		
	prompt = f"You are an AI Assistant. Based on the given context, answer the given query. context: {context}, query: {query}"
	
	print(prompt)	

	response = model.generate_content(prompt)
	st.write("**Response:**")
	st.write(response.text)

