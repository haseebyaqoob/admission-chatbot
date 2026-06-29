"""
app.py — Chainlit web UI for the admissions chatbot.

Run:
    chainlit run app.py

Requires:
  - Ollama running with the configured model pulled
  - Index built: python -m ingestion.build
"""

import asyncio
from datetime import datetime

import chainlit as cl

from config_loader import cfg
from core.llm_handler import LLMHandler
from core.pipeline import RAGPipeline
from db.database import Database
from index.hybrid_searcher import HybridSearcher

# ─── One-time startup: load models + index ────────────────────────────────────

print("Loading index …")
_searcher = HybridSearcher()
_searcher.load(cfg["index_dir"])

print("Connecting to Ollama …")
_llm = LLMHandler()

_db       = Database()
_pipeline = RAGPipeline(searcher=_searcher, llm=_llm, db=_db)
print("Ready.\n")


# ─── Auth (local dev — change for production) ─────────────────────────────────

@cl.password_auth_callback
def auth(username: str, password: str):
    if username == "admin" and password == "admin123":
        return cl.User(identifier="admin", metadata={"role": "admin"})
    return None


# ─── Chat lifecycle ───────────────────────────────────────────────────────────

@cl.on_chat_start
async def on_start():
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    cl.user_session.set("session_id", session_id)

    await cl.Message(
        content=(
            "👋 **Welcome to NED University Admissions Assistant!**\n\n"
            "I can help with:\n"
            "- 📚 Degree programs & departments\n"
            "- ✅ Eligibility criteria (BE / MS / PhD)\n"
            "- 💰 Fee structures\n"
            "- 🏆 Scholarships & financial aid\n"
            "- 📅 Admission deadlines\n"
            "- 📄 Required documents\n"
            "- 🏠 Hostel & facilities\n\n"
            "Ask me anything about NED admissions!"
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    session_id = cl.user_session.get("session_id")

    # Show typing placeholder
    msg = cl.Message(content="")
    await msg.send()

    # Run sync pipeline in thread pool
    loop     = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None, _pipeline.process, message.content, session_id
    )

    msg.content = response
    await msg.update()
