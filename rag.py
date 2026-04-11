import json
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# ✅ Stable embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# Load data
with open("firecrawl_data.json", "r") as f:
    documents = json.load(f)

if not documents:
    raise ValueError("No documents found ❌")

# Split text
def split_text(text, chunk_size=500):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

chunks = []
for doc in documents:
    chunks.extend(split_text(doc))

print(f"✅ Total chunks: {len(chunks)}")

# Create vector DB
vectorstore = FAISS.from_texts(chunks, embeddings)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

def get_context(query):
    docs = retriever.get_relevant_documents(query)
    return "\n".join([doc.page_content for doc in docs])
