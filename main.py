from rag import get_context
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
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
# ✅ LLM
# =========================
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")
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
    "doctor",
    "paramedic",
    "emergency_medical_technician (EMT)",
    "ambulance",
    "nurse",
    "firebrigade",
    "burn_specialist",
    "NDRF",
    "civil_engineer",
    "heavy_machinery_operator",
    "traffic_police",
    "tow_truck_operator",
    "police",
    "security"
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
        return "paramedic"  # safe default

    occupation = occupation.strip()

    # ✅ exact match
    if occupation in ALLOWED_OCCUPATIONS:
        return occupation

    # ✅ fallback mapping
    lower = occupation.lower()
    if lower in FALLBACK_MAPPING:
        return FALLBACK_MAPPING[lower]

    # ✅ fuzzy fallback (contains match)
    for allowed in ALLOWED_OCCUPATIONS:
        if lower in allowed.lower():
            return allowed

    # ✅ final fallback
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
1. Determine intent:
   - emergency OR non_emergency

2. Extract occupation ONLY from the allowed list above

3. If occupation not directly mentioned:
   - Choose BEST match from allowed list

4. Determine crisis_type:
   - medical → injury, illness, doctor
   - fire → fire, burning
   - accident → crash, collision
   - crime → theft, attack
   - natural_disaster → flood, earthquake

Rules:
- NEVER generate a new occupation
- ALWAYS return one occupation from allowed list
- If unsure → choose closest match
- If not emergency → is_valid_request = false

Return ONLY valid JSON.

Format:
{{
 "intent": "",
 "occupation": "",
 "crisis_type": "",
 "is_occupation_provided": true,
 "is_valid_request": true
}}
"""
)

# =========================
# ✅ ASSIST PROMPT
# =========================
assist_prompt = PromptTemplate(
    input_variables=["message", "chat_history", "context"],
    template="""
You are a calm emergency assistant.

Use ONLY the trusted safety guidelines below:
{context}

Chat history:
{chat_history}

User message:
{message}

Instructions:
- Give short, clear, step-by-step safety instructions
- Be calm and reassuring
- Prioritize life-saving steps
- Keep response 2–4 steps
- If context is missing, still give safe general advice

Do NOT:
- hallucinate
- give long paragraphs
- repeat instructions

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

    try:
        json_text = extract_json(result)
        parsed = json.loads(json_text)
    except:
        parsed = {
            "intent": "emergency",
            "occupation": "paramedic",
            "crisis_type": "medical",
            "is_occupation_provided": False,
            "is_valid_request": True
        }

    # ✅ enforce occupation restriction
    parsed["occupation"] = validate_occupation(parsed.get("occupation", ""))

    return parsed

# =========================
# ✅ ASSIST ENDPOINT
# =========================
@app.post("/assist")
def assist_user(input: AssistInput):
    memory = get_memory(input.user_id)

    chain = assist_prompt | llm | StrOutputParser()

    chat_history = memory.load_memory_variables({}).get("chat_history", "")

    # 🔥 RAG CONTEXT
    context = get_context(input.message, llm)

    print("🔍 CONTEXT:", context[:300])

    result = chain.invoke({
        "message": input.message,
        "chat_history": chat_history,
        "context": context
    })

    memory.save_context(
        {"input": input.message},
        {"output": result}
    )

    return {
        "assistant_message": result
    }
