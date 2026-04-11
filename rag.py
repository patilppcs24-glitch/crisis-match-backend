import json
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# ✅ Load embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-004")

# ✅ Load Firecrawl data
with open("firecrawl_data.json", "r") as f:
    documents = json.load(f)

if not documents:
    raise ValueError("No documents found ❌")

# 🔥 Split large text into chunks (VERY IMPORTANT)
def split_text(text, chunk_size=500):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

chunks = []
for doc in documents:
    chunks.extend(split_text(doc))

print(f"✅ Total chunks: {len(chunks)}")

# ✅ Create vector DB
vectorstore = FAISS.from_texts(chunks, embeddings)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ✅ Retrieve context
def get_context(query):
    docs = retriever.get_relevant_documents(query)
    return "\n".join([doc.page_content for doc in docs])
