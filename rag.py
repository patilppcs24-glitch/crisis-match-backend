from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from firecrawl_loader import load_firecrawl_data

# Load embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# 🔥 Load data (runs once at startup)
documents = load_firecrawl_data()

# Create vector DB
vectorstore = FAISS.from_texts(documents, embeddings)

retriever = vectorstore.as_retriever()

def get_context(query):
    docs = retriever.get_relevant_documents(query)
    return "\n".join([doc.page_content[:1000] for doc in docs])
