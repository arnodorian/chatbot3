from dotenv import load_dotenv
load_dotenv()

# =============================
# IMPORTS
# =============================

import os
import asyncio
from typing import Optional

from fastapi import FastAPI, Request
from twilio.rest import Client

from openai import OpenAI, APIConnectionError
from sqlalchemy import create_engine, text, bindparam
from sqlalchemy.dialects.postgresql import JSONB

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_postgres import PostgresChatMessageHistory

import redis
import psycopg
import uuid



# =============================
# ENV CONFIG
# =============================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_WHATSAPP_NUMBER = os.getenv("TWILIO_WHATSAPP_NUMBER")
SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL")
REDIS_URL = os.getenv("REDIS_URL")

BUFFER_DELAY = 7.0


# =============================
# INIT APP
# =============================

app = FastAPI()

twilio = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
TWILIO_FROM = TWILIO_WHATSAPP_NUMBER

openai_client = OpenAI(api_key=OPENAI_API_KEY)

# conn = psycopg.connect(SUPABASE_DB_URL)

# Still needed for RAG
# pg_engine = create_engine(
#     SUPABASE_DB_URL,
#     connect_args={"sslmode": "require"}
# )

pg_engine = create_engine(
    f"postgresql+psycopg://{SUPABASE_DB_URL.split('://')[1]}",
    connect_args={"sslmode": "require"}
)

llm = ChatGoogleGenerativeAI(
    # model="gemini-2.0-flash",
    model="gemini-2.5-flash-lite",
    temperature=0.1,
    api_key=GEMINI_API_KEY
)


# =============================
# REDIS (BUFFER + DEDUP)
# =============================

redis_client: Optional[redis.Redis] = None

if REDIS_URL:
    redis_client = redis.Redis.from_url(
        REDIS_URL,
        decode_responses=True
    )


# =============================
# SYSTEM PROMPT
# =============================

STATIC_SYSTEM_PROMPT = """
Role: You name is Rifat in English (Рифат in Russian).
You are the Senior Medical Consultant (AI WhatsApp Sales Assistant) for a premier men's health clinic called REFORMATOR CLINIC in Kazakhstan.
Your goal is to provide consultation to clients while maintaining a tone of "Professional Brotherhood."
The company name is REFORMATOR in English and Реформатор in Russian.
You primarily interact with the potential customers who speak either Russian, Kazakh or English.
Here is the description of the company:
Клиника "Реформатор" — это современное медицинское учреждение, предлагающее широкий
спектр услуг в области диагностики, лечения и профилактики заболеваний с фокусом на мужчинах.
Это первая клиника для мужчин в Казахстане, предоставляющая комплексный подход к мужскому здоровью.

Persona & Voice:
1. IMPORTANT! Locality: Use more local expressions, references, and cultural touchpoints relevant to Kazakhstan and Russian-speaking audiences.
2. Errors in texts are acceptable, inject more abbreviations, acronyms and emojis to make the conversation feel more natural.
1. The Persona: Imagine a seasoned specialist who is also a straight-shooting mentor. You are calm, authoritative, and reliable. You don't "sugarcoat," but you are never rude.
2. Directness: Use concise, punchy sentences. Avoid "fluff," excessive adjectives, or marketing jargon.
3. Simplicity: Translate complex medical terminology into plain, everyday language that a layman can understand. Never use high-level Latin terms unless followed by a simple explanation.
4. Masculine Professionalism: Avoid overly emotional or "soft" language. Use "You" (formal/respectful context) and focus on logic, results, and taking action.
5. Human-like Tone: Avoid robotic or overly formal phrasing. Replicate human-like texting in Russian.

Communication Rules:
1. No Diminutives: Never use "soft" versions of words (e.g., instead of "a little check-up," use "diagnostic exam").
2. Standard of Respect: Always address the user with professional respect. Do not use "bro" or slang, but stay grounded and relatable.
3. Action-Oriented: Every response should focus on the solution. If a user describes a problem, acknowledge it as a solvable technical issue, not a personal tragedy.
4. No Diagnosing: Explicitly state that while the symptoms are clear, a final conclusion requires a clinical visit. Frame the visit as "gathering data" or "getting a professional assessment."
5. Never mention AI, models, or automation
6. Don't address clients informally "ты".
7. Respond only once
8. Do not repeat or rephrase the same answer

You are authorized to:
1. Describe the company’s services and offerings
2. Answer pricing, availability, and process questions
3. Provide consultation on general medical questions within your expertise
4. Switch languages between Russian, Kazakh, and English as needed


You must not:
1. Share internal-only details

Hard constraint:
1. Responses must be under 5 sentences
2. Maximum 1550 characters

If you do not know an answer:
1. Use common knowledge to answer or say that you do not know the answer
2. Offer to connect the user with a human or book a call

Vocabulary Guidelines:
1. Avoid: "We are so sorry to hear that," "Please don't worry," "Tiny procedure," "Sweet," "Kind doctors."
2. Use: "This is a common issue," "It’s solvable," "Let's look at the facts," "Get back to 100%," "Standard protocol," "Professional approach."

Example Interactions:
1. User asks about a sensitive issue: "I understand. Many men face this, and there’s no reason to delay. The first step is a 20-minute consultation to identify the cause. Should we check the available slots for this week?"
2. User is hesitant: "Health is an investment in your quality of life. We don't do guesswork here—we use diagnostics and proven methods. It’s better to spend an hour at the clinic now than deal with complications later."
"""


# =============================
# POSTGRES MEMORY (LANGCHAIN)
# =============================

async def get_history(phone: str):
    # session_id = str(uuid.uuid4()) 
    session_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, phone))
    table_name = "message_store"
    conn = await psycopg.AsyncConnection.connect(
        SUPABASE_DB_URL,
        autocommit=True,
        prepare_threshold=None,  # critical for Supabase
        sslmode="require"
    )
    history = PostgresChatMessageHistory(
        table_name,
        session_id,
        async_connection=conn
    )

    # Inject system prompt only if new session
    if len(await history.aget_messages()) == 0:
        await history.aadd_message(SystemMessage(content=STATIC_SYSTEM_PROMPT))

    return history


# =============================
# RAG RETRIEVAL
# =============================
def retrieve_context(query: str, top_k: int = 2) -> str:
    try:
        embedding = openai_client.embeddings.create(
            model="text-embedding-3-large",
            input=query
        ).data[0].embedding
    except APIConnectionError:
        return ""

    sql = text("""
        SELECT *
        FROM public.match_document_chunks(
            cast(:query_embedding as vector),
            :match_count,
            :metadata_filter
        )
    """).bindparams(
        bindparam("metadata_filter", type_=JSONB)
    )

    with pg_engine.begin() as conn:
        rows = conn.execute(
            sql,
            {
                "query_embedding": embedding,
                "match_count": top_k,
                "metadata_filter": {}
            }
        ).fetchall()

    if not rows:
        return ""

    return "\n\n".join(f"- {row[4]}" for row in rows)


# =============================
# LLM HANDLER
# =============================

async def handle_llm_and_reply(phone: str, merged_user_input: str):
    context = retrieve_context(merged_user_input)

    if context:
        final_input = f"""
        Context (use only if relevant):
        {context}

        User message:
        {merged_user_input}
        """
    else:
        final_input = merged_user_input

    runnable = RunnableWithMessageHistory(
        runnable=llm,
        get_session_history=get_history,
        input_messages_key=None,
        history_messages_key=None,
    )

    # res = runnable.invoke(
    #     final_input,
    #     config={"configurable": {"session_id": phone}}
    # )
    res = await runnable.ainvoke(
        final_input,
        config={"configurable": {"session_id": phone}}
    )

    res_content = res.content.strip()

    if len(res_content) > 1500:
        res_content = res_content[:1480] + "…"

    twilio.messages.create(
        from_=TWILIO_FROM,
        to=phone,
        body=res_content
    )

# =============================
# BUFFER PROCESSOR
# =============================

async def process_buffer_after_delay(phone: str):
    await asyncio.sleep(BUFFER_DELAY)

    if not redis_client:
        return

    buffer_key = f"buffer:{phone}"

    # Get all buffered messages
    messages = redis_client.lrange(buffer_key, 0, -1)

    # Delete buffer AFTER reading
    redis_client.delete(buffer_key)

    if not messages:
        return

    merged_input = " ".join(messages)
    await handle_llm_and_reply(phone, merged_input)



# =============================
# WHATSAPP WEBHOOK
# =============================

@app.post("/whatsapp")
async def whatsapp_webhook(request: Request):
    form = await request.form()

    phone = form.get("From")
    incoming = form.get("Body", "").strip()
    message_sid = form.get("MessageSid")

    if not phone or not incoming:
        return "OK"
    if redis_client and message_sid:
    # Deduplicate Twilio retries (KEEP THIS)
        if not redis_client.set(
            f"twilio:{message_sid}",
            "1",
            nx=True,
            ex=60
        ):
            return "OK"

        buffer_key = f"buffer:{phone}"

        # Push message to Redis list
        redis_client.rpush(buffer_key, incoming)

        # Only schedule processor if this is the FIRST message
        if redis_client.llen(buffer_key) == 1:
            print("Scheduling buffer processor...")
            asyncio.create_task(process_buffer_after_delay(phone))

    # if redis_client and message_sid:
    #     # Deduplicate Twilio retries
    #     if not redis_client.set(
    #         f"twilio:{message_sid}",
    #         "1",
    #         nx=True,
    #         ex=60
    #     ):
    #         return "OK"

    #     buffer_key = f"buffer:{phone}"
    #     # lock_key = f"buffer:lock:{phone}"

    #     # redis_client.rpush(buffer_key, incoming)
    #     # redis_client.expire(buffer_key, int(BUFFER_DELAY))
    #     # if redis_client.set(lock_key, "1", nx=True, ex=int(BUFFER_DELAY)):
    #     #     print('TUT3')
    #     #     asyncio.create_task(process_buffer_after_delay(phone))
        
    #     redis_client.rpush(buffer_key, incoming)

    #     # Only schedule if this is first message
    #     if redis_client.llen(buffer_key) == 1:
    #         asyncio.create_task(process_buffer_after_delay(phone))

        # was_set = redis_client.set(lock_key, "1", nx=True, ex=int(BUFFER_DELAY))
        # if was_set:
        #     print("Creating asyncio task to process buffer...")
        #     asyncio.create_task(process_buffer_after_delay(phone))
        # else:
        #     print("Lock already exists, skipping task creation")


    return "OK"

