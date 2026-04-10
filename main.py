from fastapi import FastAPI
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv
import json
import re

load_dotenv()

app = FastAPI()

# ✅ Load LLM (UPDATED MODEL)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# ✅ Memory store (per user)
memory_store = {}

def get_memory(user_id):
    if user_id not in memory_store:
        memory_store[user_id] = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
    return memory_store[user_id]

# ✅ Request format
class Input(BaseModel):
    message: str
    user_id: str

# ✅ Prompt (STRICT JSON OUTPUT)
prompt = PromptTemplate(
    input_variables=["message", "chat_history"],
    template="""
You are an intelligent emergency assistant.

Chat history:
{chat_history}

User message:
{message}

Tasks:
1. Extract intent
2. Extract occupation if mentioned
3. If not mentioned, infer carefully
4. If unclear, set is_occupation_provided = false

Rules:
- If user clearly mentions occupation → use it
- If unclear → do NOT guess randomly
- If not emergency → set is_valid_request = false

Return ONLY valid JSON.
Do NOT add explanation text.
Do NOT add markdown.
Only raw JSON.

Format:
{
 "intent": "",
 "occupation": "",
 "is_occupation_provided": true,
 "is_valid_request": true
}
"""
)

# ✅ Safe JSON extractor (prevents crashes)
def extract_json(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group()
    return "{}"

# ✅ Health check route
@app.get("/")
def home():
    return {"message": "Crisis Match Backend Running 🚀"}

# ✅ Main AI endpoint
@app.post("/agent")
def run_agent(input: Input):
    memory = get_memory(input.user_id)

    chain = LLMChain(
        llm=llm,
        prompt=prompt,
        memory=memory
    )

    result = chain.run(message=input.message)

    # ✅ Safe parsing
    try:
        json_text = extract_json(result)
        parsed = json.loads(json_text)
    except:
        parsed = {
            "intent": "emergency",
            "occupation": "",
            "is_occupation_provided": False,
            "is_valid_request": True
        }

    return parsed  # ✅ clean response for n8n

# ✅ Run for Render
import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
