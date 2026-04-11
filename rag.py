from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FastEmbedEmbeddings

# ✅ embeddings
embeddings = FastEmbedEmbeddings()

vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

# 🔥 increase recall
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})


# 🧠 STEP 1 — detect category
def detect_query_category(query):
    q = query.lower()

    if any(k in q for k in ["fire", "burn", "smoke"]):
        return "fire"
    elif any(k in q for k in ["bleeding", "injury", "hurt", "doctor"]):
        return "medical"
    elif any(k in q for k in ["accident", "crash", "collision"]):
        return "accident"
    elif any(k in q for k in ["earthquake", "flood", "cyclone"]):
        return "natural_disaster"
    elif any(k in q for k in ["attack", "theft", "crime"]):
        return "crime"
    else:
        return "general"


# 🧠 STEP 2 — enhance query
def enhance_query(query, category):
    return f"{category} emergency situation: {query}. give safety steps"


# 🧠 STEP 3 — main retrieval
def get_context(query):
    category = detect_query_category(query)

    enhanced_query = enhance_query(query, category)

    docs = retriever.get_relevant_documents(enhanced_query)

    # 🔥 STEP 4 — filter by category
    filtered = [
        doc.page_content for doc in docs
        if f"[{category}]" in doc.page_content
    ]

    # fallback if empty
    if not filtered:
        filtered = [doc.page_content for doc in docs]

    # 🔥 STEP 5 — priority sorting (important lines first)
    if category == "medical":
        filtered = sorted(filtered, key=lambda x: "pressure" in x.lower(), reverse=True)
    elif category == "fire":
        filtered = sorted(filtered, key=lambda x: "leave" in x.lower(), reverse=True)

    # 🔥 STEP 6 — return best chunks only
    return "\n".join(filtered[:3])
