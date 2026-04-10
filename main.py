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

# ✅ LLM (stable + fast)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
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

# ✅ Request model
class Input(BaseModel):
    message: str
    user_id: str

# ✅ Prompt (NOW INCLUDES crisis_type)
prompt = PromptTemplate(
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

# ✅ Safe JSON extractor
def extract_json(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return match.group() if match else "{}"

# ✅ Health route
@app.get("/")
def home():
    return {"message": "Crisis Match Backend Running 🚀"}

# ✅ Main endpoint
@app.post("/agent")
def run_agent(input: Input):
    memory = get_memory(input.user_id)

    # Build chain (new style, no deprecated LLMChain)
    chain = prompt | llm | StrOutputParser()

    # Inject memory manually
    chat_history = memory.load_memory_variables({}).get("chat_history", "")

    result = chain.invoke({
        "message": input.message,
        "chat_history": chat_history
    })

    # Save to memory
    memory.save_context(
        {"input": input.message},
        {"output": result}
    )

    # Parse JSON safely
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
