"""The LangChain 1.0 agent (SPEC_02 §6).

Built with `create_agent`, the single retrieval tool, the `PostgresSaver`
checkpointer (shares the app pool), and a Spanish, strongly-grounded system
prompt. Run `scripts/setup_db.py` once before serving so the checkpointer
tables exist — do NOT call `.setup()` on every request.
"""
from langgraph.checkpoint.postgres import PostgresSaver

from langchain.agents import create_agent

from app.azure_clients import make_chat_model
from app.retrieval import buscar_reglamento
from app.db import pool

SYSTEM_PROMPT = """Eres un asistente experto en el reglamento estudiantil de pregrado y postgrado \
de la Universidad de Antioquia. Tu único objetivo es responder preguntas administrativas y \
normativas de la comunidad universitaria.

Reglas:
- Responde SIEMPRE en español, de forma clara, precisa y con tono institucional.
- Basa tu respuesta EXCLUSIVAMENTE en la información recuperada con la herramienta `buscar_reglamento`.
  Usa la herramienta al menos una vez antes de responder; nunca respondas de memoria.
- Puedes llamar la herramienta varias veces (reformulando la consulta) hasta un máximo de 10 veces.
- Cita las fuentes dentro de tu respuesta (p. ej. "según el Artículo 45 del Reglamento de Pregrado").
- Si tras buscar no encuentras la información, dilo explícitamente: "No encontré esta información en \
el reglamento" y sugiere consultar la dependencia correspondiente. No inventes artículos ni datos.
"""

# Checkpointer shares the app pool (autocommit + dict_row already configured in app/db.py).
checkpointer = PostgresSaver(pool)

agent = create_agent(
    model=make_chat_model(),
    tools=[buscar_reglamento],
    system_prompt=SYSTEM_PROMPT,
    checkpointer=checkpointer,
)
