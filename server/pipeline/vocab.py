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
    # High-frequency ASL Citizen / WLASL300 conversational (first-class, not synonym merges)
    "ABOUT", "AND", "BOY", "GIRL", "MOVIE", "PARTY", "CHRISTMAS", "HALLOWEEN", "YELLOW",
]


# ~130 highest-priority daily chat glosses (subset of CONVERSATION_GLOSSES).
# Used for the optional dual "daily vocab" Core ML head.
DAILY_GLOSSES: list[str] = [
    "HELLO", "HI", "BYE", "SEE", "LATER", "GOOD-MORNING", "GOOD-NIGHT",
    "THANKS", "PLEASE", "SORRY", "EXCUSE",
    "ME", "YOU", "WE", "THEY", "MY", "YOUR", "NAME", "FRIEND", "FAMILY",
    "MOTHER", "FATHER", "SISTER", "BROTHER", "BABY", "DEAF", "HEARING",
    "WHAT", "WHERE", "WHEN", "WHO", "WHY", "HOW", "WHICH",
    "YES", "NO", "OK", "MAYBE", "TRUE", "FALSE", "RIGHT", "WRONG",
    "GOOD", "BAD", "FINE", "GREAT", "MORE", "SAME", "DIFFERENT",
    "BIG", "SMALL", "NEW", "OLD",
    "WANT", "NEED", "HELP", "UNDERSTAND", "KNOW", "DONT-KNOW", "LIKE", "LOVE",
    "GO", "COME", "STOP", "WAIT", "AGAIN", "SLOW",
    "GIVE", "TAKE", "HAVE", "MAKE", "THINK", "FEEL", "REMEMBER", "FORGET",
    "TELL", "ASK", "CALL", "LEAVE", "STAY", "PLAY", "READ", "WORK", "LOOK",
    "SPELL", "WRITE", "SIGN",
    "EAT", "DRINK", "FOOD", "WATER", "COFFEE", "HUNGRY", "THIRSTY",
    "HOME", "SCHOOL", "STORE", "HOSPITAL", "BATHROOM", "HERE", "THERE", "CITY",
    "TIME", "TODAY", "TOMORROW", "YESTERDAY", "NOW", "MORNING", "NIGHT",
    "WEEK", "MONTH", "YEAR",
    "HAPPY", "SAD", "TIRED", "HOT", "COLD", "ANGRY", "SICK", "HURT",
    "ONE", "TWO", "THREE", "FOUR", "FIVE",
    "PHONE", "MONEY", "BUY", "BUSY", "READY", "IMPORTANT", "PROBLEM", "QUESTION",
    "WALK", "SLEEP", "BOOK",
    "RED", "BLUE", "BLACK", "WHITE",
    "RAIN", "SUN", "WEATHER",
    "MONDAY", "FRIDAY", "SATURDAY", "SUNDAY",
    "ENGLISH", "ASL",
    # High-frequency WLASL conversational that friends still use
    "FINISH", "CAN", "BEFORE", "CHANGE", "DECIDE", "MEET", "NICE", "WELCOME",
    "CLOTHES", "COMPUTER", "DOCTOR", "FAMILY", "APPLE", "CANDY", "DOG",
    "ABOUT", "AND", "BOY", "GIRL", "MOVIE", "PARTY", "CHRISTMAS", "HALLOWEEN", "YELLOW",
]



# Dense daily head: conversational glosses with enough multi-source pose (≥~20 train).
# Honest closed-set eval only — do not compare to full 234-class WLASL100 holdout.
DAILY_DENSE_GLOSSES: list[str] = [
    "BOOK", "FOOD", "WHY", "WHO", "DEAF", "DOG", "CANDY", "DRINK", "FINE", "NEED",
    "TIME", "DECIDE", "EAT", "APPLE", "LATER", "COMPUTER", "GO", "BEFORE", "CLOTHES",
    "BATHROOM", "BLACK", "BLUE", "CAN", "FAMILY", "HEARING", "HELP", "HOT", "MOTHER",
    "NO", "NOW", "WALK", "WHITE", "YEAR", "YES", "WRONG", "STOP", "YESTERDAY", "WRITE",
    "CHANGE", "FINISH", "FORGET", "LIKE", "MEET", "PLAY", "SCHOOL", "WANT", "WHAT",
    "WORK", "BROTHER", "CITY", "DOCTOR", "GIVE", "TELL", "MONEY", "TAKE", "CALL",
    "LEAVE", "HOW", "RIGHT", "SAME",
    # Added Citizen conversational with real pose
    "ABOUT", "AND", "BOY", "GIRL", "MOVIE", "PARTY", "CHRISTMAS", "HALLOWEEN", "YELLOW",
]


# Ultra-dense 30-class daily head (highest sample support). Honest closed-set only.
DAILY_CORE30_GLOSSES: list[str] = [
    "BOOK", "FOOD", "WHY", "WHO", "DEAF", "DOG", "CANDY", "DRINK", "FINE", "NEED",
    "TIME", "DECIDE", "EAT", "APPLE", "LATER", "COMPUTER", "GO", "BEFORE", "CLOTHES",
    "BATHROOM", "BLACK", "BLUE", "CAN", "FAMILY", "HEARING", "HELP", "HOT", "MOTHER",
    "NO", "NOW",
]

def unique_glosses(seq: list[str] | None = None) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for g in (seq if seq is not None else CONVERSATION_GLOSSES):
        if g not in seen:
            seen.add(g)
            out.append(g)
    return out


GLOSS_VOCAB = unique_glosses()
DAILY_VOCAB = unique_glosses(DAILY_GLOSSES)
DAILY_DENSE = unique_glosses(DAILY_DENSE_GLOSSES)
DAILY_CORE30 = unique_glosses(DAILY_CORE30_GLOSSES)
