"""REPL logic for the assistant CLI command."""

from pathlib import Path
from typing import Any

from result_companion.core.analizers.factory_common import _build_llm_params
from result_companion.core.assistant.stream_chat import stream_chat
from result_companion.core.parsers.config import load_assistant_config


async def _stream_reply(messages: list[dict], llm_params: dict[str, Any]) -> str:
    """Streams reply tokens to stdout and returns the full reply.

    Args:
        messages: Conversation history.
        llm_params: LiteLLM parameters.

    Returns:
        Full assistant reply as a string.
    """
    chunks = []
    async for token in stream_chat(messages, llm_params):
        print(token, end="", flush=True)
        chunks.append(token)
    print()
    return "".join(chunks)


async def run_assistant(config_path: Path | None) -> None:
    """Runs the interactive assistant REPL.

    Args:
        config_path: Optional user config YAML to override defaults.
    """
    cfg = load_assistant_config(config_path)
    llm_params = _build_llm_params(cfg.llm_factory)
    messages: list[dict] = [{"role": "system", "content": cfg.system_prompt}]

    print("Result Companion Assistant (type 'exit' to quit)\n")
    while True:
        try:
            user_input = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user_input.lower() in ("exit", "quit"):
            break
        if not user_input:
            continue
        messages.append({"role": "user", "content": user_input})
        reply = await _stream_reply(messages, llm_params)
        messages.append({"role": "assistant", "content": reply})
