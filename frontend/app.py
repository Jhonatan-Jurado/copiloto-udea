import os

import chainlit as cl
import httpx

API_URL = os.getenv("API_URL", "http://localhost:8000")


@cl.on_message
async def main(message: cl.Message):
    payload = {"query": message.content}

    # Reuse the session_id to keep conversation memory across messages
    session_id = cl.user_session.get("session_id")
    if session_id:
        payload["session_id"] = session_id

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(f"{API_URL}/chat", json=payload)
        response.raise_for_status()
        data = response.json()

    cl.user_session.set("session_id", data.get("session_id"))

    await cl.Message(content=data["answer"]).send()
