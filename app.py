"""
app.py
──────
Chainlit web-chat interface for the NED Admissions Bot.

Authentication: hardcoded for local dev via @cl.password_auth_callback.
"""

import asyncio
import sys
from datetime import datetime

# ── Python 3.14+ / Windows compatibility ────────────────────────────
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

import chainlit as cl

from core.pipeline import get_pipeline


# ── Authentication ─────────────────────────────────────────────────────
_HARDCODED_USERS = {
    "admin": "admin123",
}


@cl.password_auth_callback
def auth_callback(username: str, password: str) -> cl.User | None:
    username = username.strip().lower()
    password = password.strip()
    if _HARDCODED_USERS.get(username) == password:
        return cl.User(
            identifier=username,
            metadata={"role": "user", "provider": "credentials"},
        )
    print(f"[auth] Login failed for '{username}'")
    return None


# ── Chat handlers ──────────────────────────────────────────────────────

@cl.on_chat_start
async def on_chat_start():
    """Send a welcome message with example queries."""
    pipeline = get_pipeline()
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    pipeline.set_session(session_id)
    cl.user_session.set("session_id", session_id)

    welcome_msg = (
        "**Welcome to the NED Admissions Assistant!**\n\n"
        "I can help you with NED University admissions:\n"
        "- Programs and degrees offered\n"
        "- Eligibility criteria\n"
        "- Fee information\n"
        "- Deadlines and schedule\n"
        "- Required documents\n"
        "- Hostel facilities\n"
        "- PhD supervisors\n\n"
        "**Try asking:**\n"
        '- "What programs does CS offer?"\n'
        '- "What is the eligibility for MS Data Science?"\n'
        '- "When is the last date for admission 2026?"\n'
        '- "Find PhD supervisors in AI"'
    )

    await cl.Message(content=welcome_msg, author="NED Admissions Bot").send()


@cl.on_message
async def on_message(message: cl.Message):
    """Handle user messages via the pipeline."""
    pipeline = get_pipeline()
    session_id = cl.user_session.get("session_id")
    if session_id:
        pipeline.set_session(session_id)

    async with cl.Step(name="Thinking...", show_input=False):
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, pipeline.process_query, message.content
        )

    await cl.Message(content=response, author="NED Admissions Bot").send()
