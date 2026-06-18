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
from app.websearch import buscar_web_udea
from app.db import pool

SYSTEM_PROMPT = """Eres el asistente de la Universidad de Antioquia para la comunidad universitaria. \
Respondes en español, con tono institucional, claro y preciso.

Tienes DOS herramientas y debes elegir la correcta según la pregunta:

1) `buscar_reglamento` — tu fuente PRINCIPAL. Úsala para TODA pregunta sobre normativa: reglamento \
estudiantil de pregrado y posgrado, acuerdos, resoluciones, requisitos, trámites, plazos y \
derechos/deberes académicos. Úsala (una o varias veces, reformulando la consulta) antes de responder \
este tipo de preguntas; nunca respondas de memoria. Cita el artículo y la fuente (p. ej. "según el \
Artículo 45 del Reglamento de Pregrado"). No inventes artículos, cifras ni plazos.

2) `buscar_web_udea` — búsqueda web acotada a sitios udea.edu.co, EXCLUSIVAMENTE para información \
ACTUAL o de coyuntura de la Universidad: eventos, noticias, fechas del calendario académico vigente, \
convocatorias y novedades. NUNCA la uses para reglamento, acuerdos, resoluciones ni normativa (eso va \
SIEMPRE con `buscar_reglamento`). Cuando la uses, menciona la URL de la fuente.

Reglas generales:
- Responde SIEMPRE en español.
- Si tras buscar no encuentras la información, dilo explícitamente y sugiere consultar la dependencia o \
el sitio oficial correspondiente; no inventes.
- Puedes llamar las herramientas hasta un máximo de 10 veces en total.
"""

# Checkpointer shares the app pool (autocommit + dict_row already configured in app/db.py).
checkpointer = PostgresSaver(pool)

agent = create_agent(
    model=make_chat_model(),
    tools=[buscar_reglamento, buscar_web_udea],
    system_prompt=SYSTEM_PROMPT,
    checkpointer=checkpointer,
)
