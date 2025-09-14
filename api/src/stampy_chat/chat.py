import re
import functools
import dataclasses
from typing import Any, Callable, Optional, cast

from frozendict import frozendict

from stampy_chat.callbacks import (
    BroadcastCallbackHandler,
    CallbackHandler,
    LoggerCallbackHandler,
)
from stampy_chat.settings import Settings
from stampy_chat.llms import query_llm, RETRIEVE_DOCS_TOOL
from stampy_chat.citations import retrieve_docs, Message
from stampy_chat.prompts import inject_guidance, inject_guidance_hyde
from stampy_chat.followups import search_followups, Followup


class ConversationContext:
    """Context object to track state across tool calls in a conversation."""
    def __init__(self):
        self.citation_id_offset = 0
        self.accumulated_citations = []

@functools.lru_cache(maxsize=128)
def generate_hyde(query: str, history: frozendict, settings: Settings) -> str:
    hyde_history = inject_guidance_hyde(query, list(history), settings)
    return cast(
        str,
        query_llm(
            hyde_history,
            settings,
            stream=False,
            max_tokens=settings.hyde_max_tokens,
            thinking_budget=0,
        ),
    )

@functools.lru_cache(maxsize=128)
def retrieve_docs_cached(query: str, settings: Settings):
    return retrieve_docs(query, settings)


def run_query(
    session_id: str,
    query: str,
    history: list[Message],
    settings: Settings,
    callback: Optional[Callable[[Any], None]] = None,
    followups=True,
) -> dict[str, str | list[Followup]]:
    """Execute the query.

    :param str query: the phrase that was input by the user
    :param list[Message] history: any previous interactions with the user
    :param Settings settings: the system settings
    :param Callable[[Any], None] callback: an optional callback that will be called at various key parts of the chain
    :returns: the result of the chain
    """
    # Create conversation context for tracking citations across tool calls
    context = ConversationContext()
    callbacks: list[CallbackHandler] = [
        LoggerCallbackHandler(session_id=session_id, query=query, history=history)
    ]
    if callback:
        callbacks += [BroadcastCallbackHandler(callback)]

    if settings.tool_mode:
        # With tool use, the LLM will retrieve documents automatically when needed
        # We still need to create the basic prompted history without documents
        prompted_history = inject_guidance(query, history, [], settings)
        for call in callbacks:
            call.on_prompt(prompted_history, query, history)
        # Set empty citations for logging purposes
        for call in callbacks:
            call.on_citations_retrieved([])
    else:
        # Use the traditional workflow with manual document retrieval
        # Convert history to frozendict for caching
        frozen_history = tuple(frozendict(m) for m in history)
        docs_settings = dataclasses.replace(settings, thinking_budget=0, max_response_tokens=settings.hyde_max_tokens)

        retrieval_query = query
        if settings.enable_hyde:
            retrieval_query = generate_hyde(query, frozen_history, docs_settings)
            for call in callbacks:
                call.on_hyde_done(retrieval_query)

        docs = retrieve_docs_cached(retrieval_query, docs_settings)
        for call in callbacks:
            call.on_citations_retrieved(docs)

        prompted_history = inject_guidance(query, history, docs, settings)
        for call in callbacks:
            call.on_prompt(prompted_history, query, history)

    for call in callbacks:
        call.on_llm_start()

    response = ""
    tools = [RETRIEVE_DOCS_TOOL] if settings.tool_mode else None
    for chunk in query_llm(prompted_history, settings, tools=tools, conversation_context=context, callbacks=callbacks):
        chunk_type, text = chunk.get("type"), chunk.get("text")
        if chunk_type == "thinking":
            for call in callbacks:
                call.on_thinking(text)
        elif chunk_type == "response":
            response += text
            for call in callbacks:
                call.on_response(text)

    for call in callbacks:
        call.on_llm_end(response)

    follows = []
    if followups:
        follows = search_followups(query, response, callbacks)

    print("result", response)

    if callback:
        callback({"state": "done"})
        callback(None)
    return {"response": response, "followups": follows}
