"""Conversational ASL gloss vocabulary for the shipping PoseLSTM.

Limited-domain set aimed at friend-to-friend chat — not open-domain SLT.
Labels are uppercase gloss tokens used by gloss→English.
"""

from __future__ import annotations

# Order is the class index order used by sign_classifier.pt
CONVERSATION_GLOSSES: list[str] = [
    # Greetings / closings
    "HELLO", "HI", "BYE", "SEE", "LATER", "GOOD-MORNING", "GOOD-NIGHT",
    # Politeness
    "THANKS", "PLEASE", "SORRY", "EXCUSE",
    # Pronouns / people
    "ME", "YOU", "WE", "THEY", "MY", "YOUR", "NAME", "FRIEND", "FAMILY",
    # Questions
    "WHAT", "WHERE", "WHEN", "WHO", "WHY", "HOW", "WHICH",
    # Answers / stance
    "YES", "NO", "OK", "MAYBE", "TRUE", "FALSE",
    # Evaluative
    "GOOD", "BAD", "FINE", "GREAT", "MORE", "LESS", "SAME", "DIFFERENT",
    # Need / action
    "WANT", "NEED", "HELP", "UNDERSTAND", "KNOW", "DONT-KNOW", "LIKE", "LOVE",
    "GO", "COME", "STOP", "WAIT", "AGAIN", "SLOW", "FAST",
    # Everyday
    "EAT", "DRINK", "HOME", "WORK", "SCHOOL", "TIME", "TODAY", "TOMORROW",
    "HUNGRY", "TIRED", "HAPPY", "SAD", "HOT", "COLD",
    # Conversation repair
    "AGAIN", "SPELL", "WRITE", "LOOK",
]

# Deduplicate while preserving order (AGAIN listed twice above by design)
def unique_glosses() -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for g in CONVERSATION_GLOSSES:
        if g not in seen:
            seen.add(g)
            out.append(g)
    return out


GLOSS_VOCAB = unique_glosses()
