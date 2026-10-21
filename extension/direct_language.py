"""Direct-LLM baselines for equation and word-problem inputs.

These baselines deliberately receive no MCP tools. They make it possible to
keep the upstream ADF distinction between models evaluated directly and models
evaluated as tool-using agents.
"""

from __future__ import annotations

from typing import Any, Callable


DIRECT_SYSTEM_PROMPT = (
    "You are a calculator assistant. The word problem may be written in any "
    "provided language. Determine the arithmetic needed to answer "
    "the request. Reply with ONLY the final numerical value, with no explanation "
    "or equation."
)


def direct_equation_message(equation: str) -> str:
    """Format the unchanged ADF equation for the no-tool direct baseline."""

    return f"QUESTION: {equation.strip()}\nANSWER:"


def direct_word_problem_message(body: str, question: str) -> str:
    """Format a language input without exposing its canonical ADF equation."""

    return f"WORD PROBLEM:\n{body.strip()}\n\n{question.strip()}\nANSWER:"


def run_direct(
    user_message: str,
    chat: Callable[[list[dict[str, str]]], tuple[str | None, dict[str, int]]],
) -> tuple[str | None, dict[str, int]]:
    """Run one direct model call using the same strict answer contract as ADF."""

    messages = [
        {"role": "system", "content": DIRECT_SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]
    return chat(messages)
