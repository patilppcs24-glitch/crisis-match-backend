from rag import get_context
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv
import json
import re

load_dotenv()

app = FastAPI()

# =========================
# ✅ LLM (OPENAI FIXED)
# =========================
llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),  # ✅ FIXED (important)
    temperature=0.2,
    max_retries=2
)

# =========================
# ✅ MEMORY
# =========================
memory_store = {}

def get_memory(user_id):
    if user_id not in memory_store:
        memory_store[user_id] = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
    return memory_store[user_id]

# =========================
# ✅ ALLOWED OCCUPATIONS
# =========================
ALLOWED_OCCUPATIONS = [
    "doctor", "paramedic", "emergency_medical_technician (EMT)",
    "ambulance", "nurse", "firebrigade", "burn_specialist",
    "NDRF", "civil_engineer", "heavy_machinery_operator",
    "traffic_police", "tow_truck_operator", "police", "security"
]

# =========================
# ✅ FALLBACK MAPPING
# =========================
FALLBACK_MAPPING = {
    "firefighter": "firebrigade",
    "fireman": "firebrigade",
    "emt": "emergency_medical_technician (EMT)",
    "medic": "paramedic",
    "doctor": "doctor",
    "cop": "police",
    "policeman": "police",
    "ambulance driver": "ambulance",
    "rescue team": "NDRF"
}

def validate_occupation(occupation: str):
    if not occupation:
        return "paramedic"

    occupation = occupation.strip()

    if occupation in ALLOWED_OCCUPATIONS:
        return occupation

    lower = occupation.lower()
    if lower in FALLBACK_MAPPING:
        return FALLBACK_MAPPING[lower]

    for allowed in ALLOWED_OCCUPATIONS:
        if lower in allowed.lower():
            return allowed

    return "paramedic"

# =========================
# ✅ MODELS
# =========================
class Input(BaseModel):
    message: str
    user_id: str

class AssistInput(BaseModel):
    message: str
    user_id: str

# =========================
# ✅ AGENT PROMPT
# =========================
agent_prompt = PromptTemplate(
    input_variables=["message", "chat_history", "allowed_occupations"],
    template="""
You are an intelligent emergency assistant.

Chat history:
{chat_history}

User message:
{message}

Allowed occupations:
{allowed_occupations}

Tasks:
1. Determine intent (emergency / non_emergency)
2. Extract occupation ONLY if explicitly mentioned
3. Determine crisis_type (medical, fire, accident, crime, natural_disaster)

Rules:
- NEVER guess occupation
- Return ONLY valid JSON

Format:
{
 "intent": "",
 "occupation": "",
 "crisis_type": "",
 "is_occupation_provided": true,
 "is_valid_request": true
}
"""
)

# =========================
# ✅ ASSIST PROMPT
# =========================
assist_prompt = PromptTemplate(
    input_variables=["message", "chat_history", "context"],
    template="""
You are a calm emergency assistant.

Use ONLY:
{context}

Chat history:
{chat_history}

User message:
{message}

Instructions:
- 2–4 steps
- Clear & calm
- Life-saving priority

Respond in numbered steps only.
"""
)

# =========================
# ✅ JSON EXTRACTOR
# =========================
def extract_json(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return match.group() if match else "{}"

# =========================
# ✅ HEALTH CHECK
# =========================
@app.get("/")
def home():
    return {"message": "Crisis Match Backend Running 🚀"}

# =========================
# ✅ AGENT ENDPOINT
# =========================
@app.post("/agent")
def run_agent(input: Input):
    try:
        memory = get_memory(input.user_id)

        chain = agent_prompt | llm | StrOutputParser()

        chat_history = memory.load_memory_variables({}).get("chat_history", "")

        result = chain.invoke({
            "message": input.message,
            "chat_history": chat_history,
            "allowed_occupations": ALLOWED_OCCUPATIONS
        })

        memory.save_context(
            {"input": input.message},
            {"output": result}
        )

        json_text = extract_json(result)
        parsed = json.loads(json_text)

    except Exception as e:
        print("❌ AGENT ERROR:", str(e))
        parsed = {
            "intent": "emergency",
            "occupation": "",
            "crisis_type": "medical",
            "is_occupation_provided": False,
            "is_valid_request": True
        }

    occupation = parsed.get("occupation", "").strip()

    if occupation:
        parsed["occupation"] = validate_occupation(occupation)
        parsed["is_occupation_provided"] = True
    else:
        parsed["occupation"] = ""
        parsed["is_occupation_provided"] = False

    return parsed

# =========================
# ✅ ASSIST ENDPOINT
# =========================
@app.post("/assist")
def assist_user(input: AssistInput):
    try:
        memory = get_memory(input.user_id)

        chain = assist_prompt | llm | StrOutputParser()

        chat_history = memory.load_memory_variables({}).get("chat_history", "")

        context = get_context(input.message, llm)

        result = chain.invoke({
            "message": input.message,
            "chat_history": chat_history,
            "context": context
        })

        memory.save_context(
            {"input": input.message},
            {"output": result}
        )

        return {"assistant_message": result}

    except Exception as e:
        print("❌ ASSIST ERROR:", str(e))
        return {"assistant_message": "Stay calm. Help is on the way."}

# =========================
# 🚀 RUN FOR RENDER (CRITICAL)
# =========================
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
