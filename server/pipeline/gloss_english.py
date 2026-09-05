"""Gloss → English stage (rule-based + optional local LLM hook)."""

from __future__ import annotations

import os
import re
from typing import Sequence

# Common ASL gloss → fluent-ish English snippets (limited domain).
GLOSS_PHRASES: dict[tuple[str, ...], str] = {
    ("HELLO",): "Hello.",
    ("HI",): "Hi.",
    ("THANKS",): "Thank you.",
    ("THANK", "YOU"): "Thank you.",
    ("YES",): "Yes.",
    ("NO",): "No.",
    ("PLEASE",): "Please.",
    ("HELP",): "Help.",
    ("HELP", "ME"): "Help me.",
    ("NAME",): "Name.",
    ("MY", "NAME", "WHAT"): "What is your name?",
    ("MY", "NAME"): "My name…",
    ("YOUR", "NAME", "WHAT"): "What is your name?",
    ("FRIEND",): "Friend.",
    ("FAMILY",): "Family.",
    ("LOVE",): "Love.",
    ("I", "LOVE", "YOU"): "I love you.",
    ("ME", "LOVE", "YOU"): "I love you.",
    ("HOW",): "How?",
    ("HOW", "YOU"): "How are you?",
    ("HOW", "YOU", "FINE"): "How are you? I'm fine.",
    ("YOU", "FINE"): "Are you fine?",
    ("YOU", "GOOD"): "Are you good?",
    ("ME", "FINE"): "I'm fine.",
    ("ME", "GOOD"): "I'm good.",
    ("ME", "TIRED"): "I'm tired.",
    ("ME", "HUNGRY"): "I'm hungry.",
    ("ME", "WANT", "EAT"): "I want to eat.",
    ("ME", "WANT", "DRINK"): "I want a drink.",
    ("ME", "NEED", "HELP"): "I need help.",
    ("YOU",): "You.",
    ("ME",): "Me.",
    ("WE",): "We.",
    ("GOOD",): "Good.",
    ("BAD",): "Bad.",
    ("FINE",): "Fine.",
    ("GREAT",): "Great.",
    ("MORE",): "More.",
    ("SORRY",): "Sorry.",
    ("EXCUSE",): "Excuse me.",
    ("BYE",): "Bye.",
    ("SEE", "YOU", "LATER"): "See you later.",
    ("SEE", "LATER"): "See you later.",
    ("WHAT",): "What?",
    ("WHERE",): "Where?",
    ("WHEN",): "When?",
    ("WHO",): "Who?",
    ("WHY",): "Why?",
    ("WHICH",): "Which?",
    ("OK",): "OK.",
    ("MAYBE",): "Maybe.",
    ("STOP",): "Stop.",
    ("WAIT",): "Wait.",
    ("UNDERSTAND",): "I understand.",
    ("ME", "UNDERSTAND"): "I understand.",
    ("DONT-KNOW",): "I don't know.",
    ("ME", "DONT-KNOW"): "I don't know.",
    ("KNOW",): "I know.",
    ("AGAIN",): "Again.",
    ("SLOW",): "Slow please.",
    ("PLEASE", "SLOW"): "Please slow down.",
    ("GOOD", "MORNING"): "Good morning.",
    ("GOOD-MORNING",): "Good morning.",
    ("GOOD-NIGHT",): "Good night.",
    ("WANT",): "Want.",
    ("NEED",): "Need.",
    ("LIKE",): "Like.",
    ("GO",): "Go.",
    ("COME",): "Come.",
    ("HOME",): "Home.",
    ("WORK",): "Work.",
    ("SCHOOL",): "School.",
    ("TODAY",): "Today.",
    ("TOMORROW",): "Tomorrow.",
    ("HAPPY",): "Happy.",
    ("SAD",): "Sad.",
    ("HOT",): "Hot.",
    ("COLD",): "Cold.",
    ("LOOK",): "Look.",
    ("SPELL",): "Please spell that.",
    ("WRITE",): "Write.",
    ("TRUE",): "True.",
    ("FALSE",): "False.",
    ("SAME",): "Same.",
    ("DIFFERENT",): "Different.",
    ("WHAT", "YOU", "WANT"): "What do you want?",
    ("WHERE", "YOU", "GO"): "Where are you going?",
    ("YOU", "UNDERSTAND"): "Do you understand?",
    ("ME", "GO", "HOME"): "I'm going home.",
    ("NICE", "MEET", "YOU"): "Nice to meet you.",
    ("WELCOME",): "Welcome.",
    ("MOTHER",): "Mother.",
    ("FATHER",): "Father.",
    ("SISTER",): "Sister.",
    ("BROTHER",): "Brother.",
    ("BABY",): "Baby.",
    ("PERSON",): "Person.",
    ("DEAF",): "Deaf.",
    ("HEARING",): "Hearing.",
    ("RIGHT",): "Right.",
    ("WRONG",): "Wrong.",
    ("BIG",): "Big.",
    ("SMALL",): "Small.",
    ("NEW",): "New.",
    ("OLD",): "Old.",
    ("GIVE",): "Give.",
    ("TAKE",): "Take.",
    ("HAVE",): "Have.",
    ("MAKE",): "Make.",
    ("THINK",): "Think.",
    ("FEEL",): "Feel.",
    ("REMEMBER",): "Remember.",
    ("FORGET",): "Forget.",
    ("TELL",): "Tell.",
    ("ASK",): "Ask.",
    ("CALL",): "Call.",
    ("LEAVE",): "Leave.",
    ("STAY",): "Stay.",
    ("PLAY",): "Play.",
    ("READ",): "Read.",
    ("SIGN",): "Sign.",
    ("FOOD",): "Food.",
    ("WATER",): "Water.",
    ("COFFEE",): "Coffee.",
    ("THIRSTY",): "Thirsty.",
    ("ME", "THIRSTY"): "I'm thirsty.",
    ("STORE",): "Store.",
    ("HOSPITAL",): "Hospital.",
    ("BATHROOM",): "Bathroom.",
    ("OUTSIDE",): "Outside.",
    ("INSIDE",): "Inside.",
    ("HERE",): "Here.",
    ("THERE",): "There.",
    ("CITY",): "City.",
    ("YESTERDAY",): "Yesterday.",
    ("NOW",): "Now.",
    ("MORNING",): "Morning.",
    ("NIGHT",): "Night.",
    ("WEEK",): "Week.",
    ("MONTH",): "Month.",
    ("YEAR",): "Year.",
    ("ANGRY",): "Angry.",
    ("SCARED",): "Scared.",
    ("SICK",): "Sick.",
    ("HURT",): "Hurt.",
    ("ME", "SICK"): "I'm sick.",
    ("ME", "HURT"): "I'm hurt.",
    ("ONE",): "One.",
    ("TWO",): "Two.",
    ("THREE",): "Three.",
    ("FOUR",): "Four.",
    ("FIVE",): "Five.",
    ("SIX",): "Six.",
    ("SEVEN",): "Seven.",
    ("EIGHT",): "Eight.",
    ("NINE",): "Nine.",
    ("TEN",): "Ten.",
    ("PHONE",): "Phone.",
    ("TEXT",): "Text.",
    ("EMAIL",): "Email.",
    ("MONEY",): "Money.",
    ("BUY",): "Buy.",
    ("BUSY",): "Busy.",
    ("READY",): "Ready.",
    ("IMPORTANT",): "Important.",
    ("PROBLEM",): "Problem.",
    ("QUESTION",): "Question.",
    ("ANSWER",): "Answer.",
    ("IDEA",): "Idea.",
    ("CAR",): "Car.",
    ("BUS",): "Bus.",
    ("WALK",): "Walk.",
    ("SLEEP",): "Sleep.",
    ("BOOK",): "Book.",
    ("RED",): "Red.",
    ("BLUE",): "Blue.",
    ("GREEN",): "Green.",
    ("BLACK",): "Black.",
    ("WHITE",): "White.",
    ("RAIN",): "Rain.",
    ("SUN",): "Sun.",
    ("WEATHER",): "Weather.",
    ("MONDAY",): "Monday.",
    ("TUESDAY",): "Tuesday.",
    ("WEDNESDAY",): "Wednesday.",
    ("THURSDAY",): "Thursday.",
    ("FRIDAY",): "Friday.",
    ("SATURDAY",): "Saturday.",
    ("SUNDAY",): "Sunday.",
    ("ENGLISH",): "English.",
    ("ASL",): "ASL.",
    ("ME", "GO", "WORK"): "I'm going to work.",
    ("ME", "GO", "SCHOOL"): "I'm going to school.",
    ("YOU", "WANT", "DRINK"): "Do you want a drink?",
    ("WHERE", "BATHROOM"): "Where is the bathroom?",
    ("WHAT", "TIME"): "What time?",
}


def _norm(g: str) -> str:
    g = g.upper().strip()
    g = g.replace(" ", "-")
    return re.sub(r"[^A-Z\-]", "", g)


def _dedupe_consecutive(tokens: list[str]) -> list[str]:
    out: list[str] = []
    for t in tokens:
        if not out or out[-1] != t:
            out.append(t)
    return out


def gloss_to_english(
    gloss: Sequence[str],
    *,
    use_llm: bool = False,
    nmm: dict | None = None,
) -> str:
    tokens = _dedupe_consecutive([_norm(g) for g in gloss if g and _norm(g)])
    if not tokens:
        return ""

    # Map I↔ME for phrase matching
    normalized = ["ME" if t == "I" else t for t in tokens]

    # Longest-phrase match covering the full sequence when possible.
    sentence = ""
    for n in range(len(normalized), 0, -1):
        for i in range(0, len(normalized) - n + 1):
            key = tuple(normalized[i : i + n])
            if key in GLOSS_PHRASES:
                if n == len(normalized):
                    sentence = GLOSS_PHRASES[key]
                    sentence = _maybe_llm(normalized, sentence, use_llm)
                    return _apply_nmm(sentence, normalized, nmm)
                left = gloss_to_english(normalized[:i], use_llm=False, nmm=None)
                mid = GLOSS_PHRASES[key]
                right = gloss_to_english(normalized[i + n :], use_llm=False, nmm=None)
                stitched = " ".join(x for x in [left, mid, right] if x).strip()
                return _apply_nmm(_finalize_sentence(stitched), normalized, nmm)

    joined = _tokens_to_sentence(normalized)
    sentence = _maybe_llm(normalized, joined, use_llm)
    return _apply_nmm(sentence, normalized, nmm)


def _apply_nmm(english: str, tokens: list[str], nmm: dict | None) -> str:
    """Condition English on brows / negation / lean — soft phone-Vision cues."""
    if not english or not nmm:
        return _finalize_sentence(english)
    text = english.strip()
    brow = float(nmm.get("browRaise") or 0)
    shake = float(nmm.get("headShake") or 0)
    frown = float(nmm.get("frown") or 0)
    lean = abs(float(nmm.get("torsoLean") or 0))
    tilt = float(nmm.get("shoulderTilt") or 0)
    conf = float(nmm.get("confidence") or 0.5)

    # Negation first (head shake / frown)
    if conf >= 0.2 and (shake >= 0.38 or frown >= 0.48):
        low = text.lower()
        if not any(x in low for x in ("not ", "no ", "don't", "n’t", "dont")):
            primary = tokens[0] if tokens else ""
            if primary not in {"NO", "DONT-KNOW"}:
                core = text.rstrip(".!?")
                if low.startswith("i'm "):
                    text = "I'm not " + core[4:].strip()
                elif low.startswith("i "):
                    text = "I don't " + core[2:].strip().lower()
                else:
                    text = "Not " + core[0].lower() + core[1:] if len(core) > 1 else "Not " + core

    # Questions from raised brows
    wh = {"WHAT", "WHERE", "WHEN", "WHO", "WHY", "HOW", "WHICH", "QUESTION"}
    if conf >= 0.25 and brow >= 0.38:
        core = text.rstrip(".!")
        low = core.lower()
        if not core.endswith("?"):
            if tokens and tokens[0] in wh or (tokens and tokens[-1] in wh):
                text = core + "?"
            elif low.startswith(("are you", "do you", "what", "where", "how", "who", "why", "when", "which")):
                text = core + "?"
            elif low.startswith("i'm "):
                text = "Are you " + core[4:].strip() + "?"
            elif low.startswith("i "):
                text = "Do you " + core[2:].strip().lower() + "?"
            elif low in {"you", "good", "fine", "ok", "okay"} or low.startswith("you "):
                text = ("Are you okay?" if low == "you" else f"Are you {low.removeprefix('you ').strip()}?")
                if not text.endswith("?"):
                    text = text.rstrip(".") + "?"
            elif len(core) <= 28:
                text = f"Are you {low}?"
            else:
                text = core + "?"

    # Emphasis from lean / shoulder tilt
    if conf >= 0.25 and (lean >= 0.28 or tilt >= 0.32):
        if not text.endswith("?") and not text.endswith("!"):
            text = text.rstrip(".") + "!"

    return _finalize_sentence(text)


def _tokens_to_sentence(tokens: list[str]) -> str:
    # Mild fluency glue for leftover tokens.
    pretty = []
    for t in tokens:
        if t == "ME":
            pretty.append("I")
        elif t == "DONT-KNOW":
            pretty.append("don't know")
        elif t == "GOOD-MORNING":
            pretty.append("good morning")
        elif t == "GOOD-NIGHT":
            pretty.append("good night")
        else:
            pretty.append(t.lower())
    if not pretty:
        return ""
    text = " ".join(pretty)
    text = text[0].upper() + text[1:]
    if tokens[-1] in {"WHAT", "WHERE", "HOW", "WHO", "WHY", "WHICH", "WHEN"}:
        return text + "?"
    return text + "."


def _finalize_sentence(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    # Avoid double punctuation like "Hello. ?"
    text = re.sub(r"\.\s*\.", ".", text)
    return text


def _maybe_llm(tokens: list[str], fallback: str, use_llm: bool) -> str:
    if use_llm or os.environ.get("ASL_GLOSS_LLM_CMD"):
        llm = _try_local_llm(tokens)
        if llm:
            return llm
    return _finalize_sentence(fallback)


def _try_local_llm(tokens: list[str]) -> str | None:
    """Optional hook: set ASL_GLOSS_LLM_CMD to a local prompt command.

    Example: export ASL_GLOSS_LLM_CMD='ollama run llama3.2'
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
