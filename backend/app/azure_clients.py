"""Azure OpenAI client factories (SPEC_02 §3).

gpt-5-nano is a reasoning model (SPEC_00 §6.1): it rejects `temperature`,
`top_p`, `presence_penalty`, `frequency_penalty` and `max_tokens`. Do NOT pass
them. Control cost/latency with `reasoning_effort` instead.
"""
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

from app.config import settings


def make_chat_model() -> AzureChatOpenAI:
    # NOTE: do NOT pass temperature / max_tokens — gpt-5-nano (reasoning model) rejects them.
    # If your installed langchain-openai does not accept `reasoning_effort` as a direct
    # kwarg, pass it via model_kwargs={"reasoning_effort": settings.reasoning_effort} instead.
    return AzureChatOpenAI(
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key,
        api_version=settings.azure_openai_api_version,
        azure_deployment=settings.azure_openai_chat_deployment,
        reasoning_effort=settings.reasoning_effort,   # minimal -> fast & cheap, sequential tool calls
    )


def make_embeddings() -> AzureOpenAIEmbeddings:
    return AzureOpenAIEmbeddings(
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key,
        api_version=settings.azure_openai_api_version,
        azure_deployment=settings.azure_openai_embeddings_deployment,
        # text-embedding-3-small defaults to 1536 dims; do not override.
    )
