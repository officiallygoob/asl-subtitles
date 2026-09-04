"""Gloss → English stage (rule-based + optional local LLM hook)."""

from __future__ import annotations

import os
import re
from typing import Sequence

# Common ASL gloss → fluent-ish English snippets (limited domain).
GLOSS_PHRASES: dict[tuple[str, ...], str] = {
    ("HELLO",): "Hello.",
    ("THANKS",): "Thank you.",
    ("THANK", "YOU"): "Thank you.",
    ("YES",): "Yes.",
    ("NO",): "No.",
    ("PLEASE",): "Please.",
    ("HELP",): "Help.",
    ("HELP", "ME"): "Help me.",
    ("NAME",): "Name.",
    ("MY", "NAME"): "My name…",
    ("FRIEND",): "Friend.",
    ("LOVE",): "Love.",
    ("I", "LOVE", "YOU"): "I love you.",
    ("HOW",): "How?",
    ("HOW", "YOU"): "How are you?",
    ("YOU",): "You.",
    ("ME",): "Me.",
    ("GOOD",): "Good.",
    ("BAD",): "Bad.",
    ("MORE",): "More.",
    ("SORRY",): "Sorry.",
    ("BYE",): "Bye.",
    ("WHAT",): "What?",
    ("WHERE",): "Where?",
    ("OK",): "OK.",
    ("STOP",): "Stop.",
    ("UNDERSTAND",): "Understand.",
    ("AGAIN",): "Again.",
    ("SLOW",): "Slow please.",
    ("GOOD", "MORNING"): "Good morning.",
    ("SEE", "YOU", "LATER"): "See you later.",
}


def _norm(g: str) -> str:
    return re.sub(r"[^A-Z]", "", g.upper())


def gloss_to_english(gloss: Sequence[str], *, use_llm: bool = False) -> str:
    tokens = [_norm(g) for g in gloss if g and _norm(g)]
    if not tokens:
        return ""

    # Longest-phrase match first.
    for n in range(len(tokens), 0, -1):
        for i in range(0, len(tokens) - n + 1):
            key = tuple(tokens[i : i + n])
            if key in GLOSS_PHRASES:
                # If the whole sequence matches a phrase, return it.
                if n == len(tokens):
                    return GLOSS_PHRASES[key]
                # Otherwise stitch remaining tokens.
                left = gloss_to_english(tokens[:i], use_llm=False)
                mid = GLOSS_PHRASES[key]
                right = gloss_to_english(tokens[i + n :], use_llm=False)
                return " ".join(x for x in [left, mid, right] if x).strip()

    joined = " ".join(t.capitalize() for t in tokens)
    if use_llm:
        llm = _try_local_llm(tokens)
        if llm:
            return llm
    # Mild fluency: add terminal punctuation for interrogatives.
    if tokens[-1] in {"WHAT", "WHERE", "HOW", "WHO", "WHY"}:
        return joined + "?"
    return joined + "."


def _try_local_llm(tokens: list[str]) -> str | None:
    """Optional hook: set ASL_GLOSS_LLM_CMD to a local prompt command.

    Example: export ASL_GLOSS_LLM_CMD='ollama run llama3.2'
    We pass a short prompt on stdin; stdout is the English sentence.
    Disabled by default — no network calls, no API keys in-repo.
    """
    cmd = os.environ.get("ASL_GLOSS_LLM_CMD")
    if not cmd:
        return None
    import subprocess

    prompt = (
        "Convert these ASL gloss tokens into one short natural English sentence. "
        "Reply with only the sentence.\nGloss: " + " ".join(tokens)
    )
    try:
        result = subprocess.run(
            cmd,
            input=prompt,
            text=True,
            shell=True,
            capture_output=True,
            timeout=8,
            check=False,
        )
        text = (result.stdout or "").strip().splitlines()
        if text:
            return text[-1].strip()
    except Exception:
        return None
    return None
