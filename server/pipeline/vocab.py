"""Conversational ASL gloss vocabulary for the shipping PoseLSTM.

Limited-domain set aimed at friend-to-friend chat — not open-domain SLT.
Labels are uppercase gloss tokens used by gloss→English.
"""

from __future__ import annotations

# Order is the class index order used by sign_classifier.pt
CONVERSATION_GLOSSES: list[str] = [
    # Greetings / closings
    "HELLO", "HI", "BYE", "SEE", "LATER", "GOOD-MORNING", "GOOD-NIGHT",
    "WELCOME", "NICE", "MEET",
    # Politeness
    "THANKS", "PLEASE", "SORRY", "EXCUSE",
    # Pronouns / people
    "ME", "YOU", "WE", "THEY", "MY", "YOUR", "NAME", "FRIEND", "FAMILY",
    "MOTHER", "FATHER", "SISTER", "BROTHER", "BABY", "PERSON",
    "DEAF", "HEARING",
    # Questions
    "WHAT", "WHERE", "WHEN", "WHO", "WHY", "HOW", "WHICH",
    # Answers / stance
    "YES", "NO", "OK", "MAYBE", "TRUE", "FALSE", "RIGHT", "WRONG",
    # Evaluative
    "GOOD", "BAD", "FINE", "GREAT", "MORE", "LESS", "SAME", "DIFFERENT",
    "BIG", "SMALL", "NEW", "OLD",
    # Need / action
    "WANT", "NEED", "HELP", "UNDERSTAND", "KNOW", "DONT-KNOW", "LIKE", "LOVE",
    "GO", "COME", "STOP", "WAIT", "AGAIN", "SLOW", "FAST",
    "GIVE", "TAKE", "HAVE", "MAKE", "THINK", "FEEL", "REMEMBER", "FORGET",
    "TELL", "ASK", "CALL", "LEAVE", "STAY", "PLAY", "READ",
    "WORK", "LOOK", "SPELL", "WRITE", "SIGN",
    # Everyday / food / body
    "EAT", "DRINK", "FOOD", "WATER", "COFFEE", "HUNGRY", "THIRSTY",
    "HOME", "SCHOOL", "STORE", "HOSPITAL", "BATHROOM", "OUTSIDE", "INSIDE",
    "HERE", "THERE", "CITY",
    "TIME", "TODAY", "TOMORROW", "YESTERDAY", "NOW", "MORNING", "NIGHT",
    "WEEK", "MONTH", "YEAR",
    "HAPPY", "SAD", "TIRED", "HOT", "COLD", "ANGRY", "SCARED", "SICK", "HURT",
    # Numbers
    "ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN", "EIGHT", "NINE", "TEN",
    # Social / tech / misc conversational
    "PHONE", "TEXT", "EMAIL", "MONEY", "BUY", "BUSY", "READY", "IMPORTANT",
    "PROBLEM", "QUESTION", "ANSWER", "IDEA",
    "CAR", "BUS", "WALK", "SLEEP", "BOOK",
    "RED", "BLUE", "GREEN", "BLACK", "WHITE",
    "RAIN", "SUN", "WEATHER",
    "MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY",
    # Conversation repair
    "SPELL", "WRITE", "LOOK", "ENGLISH", "ASL",
]


def unique_glosses() -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for g in CONVERSATION_GLOSSES:
        if g not in seen:
            seen.add(g)
            out.append(g)
    return out


GLOSS_VOCAB = unique_glosses()
