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

# ✅ LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# ✅ Memory per user
memory_store = {}

def get_memory(user_id):
    if user_id not in memory_store:
        memory_store[user_id] = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
    return memory_store[user_id]

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
    input_variables=["message", "chat_history"],
    template="""
You are an intelligent emergency assistant.

Chat history:
{chat_history}

User message:
{message}

Tasks:
1. Determine intent:
   - emergency OR non_emergency
2. Extract occupation if clearly mentioned (doctor, police, firefighter, etc.)
3. Determine crisis_type:
   - medical → injury, illness, doctor, ambulance
   - fire → fire, burning
   - accident → crash, collision
   - crime → theft, attack, violence
   - natural_disaster → flood, earthquake, storm
4. If occupation not mentioned → infer carefully
5. If unclear → is_occupation_provided = false

Rules:
- If occupation is clearly mentioned → use it
- Always assign a valid crisis_type
- If not emergency → is_valid_request = false

Return ONLY valid JSON.
No explanation. No markdown.

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
    input_variables=["message", "chat_history"],
    template="""
You are a calm emergency assistant helping a victim.

Chat history:
{chat_history}

User message:
{message}

Tasks:
1. Understand the situation
2. Identify crisis type internally:
   - medical → injury, bleeding, illness
   - fire → fire, smoke
   - accident → crash, collision
   - crime → attack, danger
   - natural_disaster → flood, earthquake

3. Give immediate safety instructions

Rules:
- Be calm and reassuring
- Give 2–4 short actionable steps
- Focus on immediate safety
- Do NOT mention crisis type explicitly
- Do NOT ask unnecessary questions
- Do NOT give long explanations

Examples:
Medical → apply pressure, stop bleeding
Fire → exit building, avoid smoke
Accident → stay still, check injuries
Crime → move to safe place
Natural disaster → go to safe zone

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
        "chat_history": chat_history
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
            "occupation": "",
            "crisis_type": "medical",
            "is_occupation_provided": False,
            "is_valid_request": True
        }

    return parsed

# =========================
# ✅ ASSIST ENDPOINT
# =========================

@app.post("/assist")
def assist_user(input: AssistInput):
    memory = get_memory(input.user_id)

    chain = assist_prompt | llm | StrOutputParser()

    chat_history = memory.load_memory_variables({}).get("chat_history", "")

    result = chain.invoke({
        "message": input.message,
        "chat_history": chat_history
    })

    memory.save_context(
        {"input": input.message},
        {"output": result}
    )

    return {
        "assistant_message": result
    }
