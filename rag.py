from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FastEmbedEmbeddings

# ✅ lightweight + accurate
embeddings = FastEmbedEmbeddings()

vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

def get_context(query):
    docs = retriever.get_relevant_documents(query)
    return "\n".join([doc.page_content for doc in docs])
