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

def rerank_docs(query, docs, llm):
    combined = "\n\n".join(docs)

    prompt = f"""
You are an expert emergency assistant.

User query:
{query}

Here are some safety guidelines:

{combined}

Select ONLY the most relevant 2–3 pieces of information.
Return clean, actionable steps only.
Do not include irrelevant info.
"""

    response = llm.invoke(prompt)

    return response.content if hasattr(response, "content") else str(response)

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
def get_context(query, llm):
    category = detect_query_category(query)

    enhanced_query = enhance_query(query, category)

    docs = retriever.get_relevant_documents(enhanced_query)

    # 🔥 STEP 4 — filter by category
    filtered = [
        doc.page_content for doc in docs
        if f"[{category}]" in doc.page_content
    ]

    if not filtered:
        filtered = [doc.page_content for doc in docs]

    # 🔥 STEP 5 — priority sorting
    if category == "medical":
        filtered = sorted(filtered, key=lambda x: "pressure" in x.lower(), reverse=True)
    elif category == "fire":
        filtered = sorted(filtered, key=lambda x: "leave" in x.lower(), reverse=True)

    # 🔥 STEP 6 — TAKE TOP 5 FOR RERANKING
    top_docs = filtered[:5]

    # ⚡ OPTIMIZATION (skip reranking if already clean)
    if len(top_docs) <= 2:
        return "\n".join(top_docs)

    # 🔥 STEP 7 — LLM RERANKING
    context = rerank_docs(query, top_docs, llm)

    return context
