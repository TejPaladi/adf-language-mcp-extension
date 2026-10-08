"""Word-problem version of the ADF MCP calculator-agent loop.

This module adapts ``upstream_adf/code/mcp_agent.py::run_agent``. The MCP
tools, provider adapters, tool-message format, and maximum-step policy remain
the same. Only the agent's input changes from a symbolic equation to the
original SVAMP body and question.
"""

from __future__ import annotations

from typing import Any, Callable


LANGUAGE_SYSTEM_PROMPT = (
    "You are a calculator agent. Read the word problem and determine the "
    "arithmetic needed to answer it. Use the provided tools for every "
    "arithmetic operation, one operation per tool call. Do not calculate an "
    "operation yourself. When you have the final result, reply with ONLY the "
    "final numerical value, with no explanation or equation."
)


def word_problem_message(body: str, question: str) -> str:
    """Format original SVAMP text without adding a gold equation."""

    return f"WORD PROBLEM:\n{body.strip()}\n\n{question.strip()}\nANSWER:"


def run_language_agent(
    body: str,
    question: str,
    session: Any,
    chat: Callable[..., tuple[dict[str, Any], str, list[dict[str, Any]], dict[str, int]]],
    tool_message: Callable[[dict[str, Any], str], dict[str, Any]],
    tools: list[dict[str, Any]],
    max_steps: int = 6,
) -> tuple[str | None, dict[str, int]]:
    """Run the upstream-style MCP loop on a natural-language word problem.

    Returned statistics retain the fields used by the upstream agent loop,
    plus any provider-specific usage fields returned by ``chat``.
    """

    messages = [
        {"role": "system", "content": LANGUAGE_SYSTEM_PROMPT},
        {"role": "user", "content": word_problem_message(body, question)},
    ]
    stats: dict[str, int] = {
        "llm_turns": 0,
        "tool_calls": 0,
        "tool_errors": 0,
        "no_tool_answers": 0,
        "max_steps_exceeded": 0,
    }

    for _ in range(max_steps):
        try:
            assistant_message, content, tool_calls, turn_stats = chat(messages, tools)
        except Exception:
            # The runner records this as an API error, matching upstream
            # behaviour without placing provider response text in a result.
            return None, stats

        stats["llm_turns"] += 1
        for key, value in turn_stats.items():
            stats[key] = stats.get(key, 0) + int(value or 0)

        if not tool_calls:
            if stats["tool_calls"] == 0:
                stats["no_tool_answers"] = 1
            return content, stats

        messages.append(assistant_message)
        for call in tool_calls:
            stats["tool_calls"] += 1
            result = session.call_tool(call["name"], call["args"])
            if result.is_error:
                stats["tool_errors"] += 1
            messages.append(tool_message(call, result.content[0].text))

    stats["max_steps_exceeded"] = 1
    return None, stats
