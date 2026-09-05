"""Canonicalize ASL gloss strings across WLASL / ASL Citizen / synth.

ASL Citizen (and some other packs) attach sense IDs (ABOUT1, CANDY2) and
compound lemmas (HURDLE/TRIP1). Without mapping, Citizen barely overlaps the
shipping head. This module:

1. Strips sense suffixes / slash-variants
2. Applies a conservative synonym / lemma map into shipping labels
3. Leaves unknown glosses as stripped lemmas (caller may drop under --focus)
"""

from __future__ import annotations

import re

# Conservative only: wrong merges hurt holdout more than unused Citizen rows.
# Prefer sense-stripping + near-identity lemmas over creative paraphrases.
SYNONYM_MAP: dict[str, str] = {
    # Places / hygiene
    "BATH": "BATHROOM",
    "TOILET": "BATHROOM",
    "RESTROOM": "BATHROOM",
    "REST-ROOM": "BATHROOM",
    "WASHROOM": "BATHROOM",
    # Communication
    "CALLTTY": "CALL",
    "TELEPHONE": "PHONE",
    "CELLPHONE": "PHONE",
    "CELL": "PHONE",
    "MOBILE": "PHONE",
    "TYPE": "WRITE",
    "TYPING": "WRITE",
    "ALPHABET": "SPELL",
    "FINGERSPELL": "SPELL",
    "FINGERSPELLING": "SPELL",
    "SMS": "TEXT",
    "TEXTING": "TEXT",
    "EMAILING": "EMAIL",
    # WH / people probes
    "WHATFOR": "WHY",
    "WHAT-FOR": "WHY",
    "HOWCOME": "WHY",
    "ANYONE": "WHO",
    "SOMEONE": "WHO",
    "WHAT-UP": "WHAT",
    "WHATS-UP": "WHAT",
    "WHATSUP": "WHAT",
    # Time
    "AFTER": "LATER",
    "AFTERWARDS": "LATER",
    "NOON": "TIME",
    "MIDDAY": "TIME",
    "CALENDAR": "MONTH",
    "RECENT": "YESTERDAY",
    # Numbers / money
    "THIRD": "THREE",
    "5DOLLARS": "MONEY",
    "DOLLAR": "MONEY",
    "DOLLARS": "MONEY",
    # Meals → FOOD (limited-domain chat bucket)
    "BREAKFAST": "FOOD",
    "LUNCH": "FOOD",
    "DINNER": "FOOD",
    "MEAL": "FOOD",
    # Need / cognition / emphasis
    "DEMAND": "NEED",
    "REQUIRE": "NEED",
    "BELIEVE": "THINK",
    "GUESS": "MAYBE",
    "SPECIAL": "IMPORTANT",
    # Motion
    "TAKEOFF": "LEAVE",
    "DEPART": "LEAVE",
    "BORROW": "TAKE",
    "LEND": "GIVE",
    # Weather
    "CLOUD": "WEATHER",
    "CLOUDY": "WEATHER",
    # Soft discourse
    "BECAUSE": "WHY",
    "CANCEL": "STOP",
    "CANCELLATION": "STOP",
    "CONFUSED": "DONT-KNOW",
    # Family (near-identity lemmas)
    "MOM": "MOTHER",
    "MOMMY": "MOTHER",
    "MAMA": "MOTHER",
    "DAD": "FATHER",
    "DADDY": "FATHER",
    "PAPA": "FATHER",
    # Greetings / politeness
    "GOODBYE": "BYE",
    "GOOD-BYE": "BYE",
    "THANKYOU": "THANKS",
    "THANK-YOU": "THANKS",
    "THANK": "THANKS",
    "OKAY": "OK",
    "ALRIGHT": "OK",
    "ALL-RIGHT": "OK",
    "NOPE": "NO",
    "YEP": "YES",
    "YUP": "YES",
    "YEAH": "YES",
    # Film / social (targets must be in shipping vocab)
    "FILM": "MOVIE",
    "CINEMA": "MOVIE",
    "MOVIES": "MOVIE",
    # Soft kid bucket for chat
    "KID": "BABY",
    "CHILD": "BABY",
    "CHILDREN": "BABY",
}

_SENSE_RE = re.compile(r"^(.*?)(\d+)$")
_NON_ALNUM = re.compile(r"[^A-Z0-9\-/]+")


def strip_sense_suffix(gloss: str) -> str:
    """ABOUT1 → ABOUT, CANDY2 → CANDY, HURDLE/TRIP1 → HURDLE."""
    g = (gloss or "").upper().strip().replace(" ", "-").replace("_", "-")
    g = _NON_ALNUM.sub("", g)
    if "/" in g:
        g = g.split("/", 1)[0]
    g = g.strip("-")
    while True:
        m = _SENSE_RE.match(g)
        if not m or not m.group(1):
            break
        g = m.group(1).rstrip("-")
    return g or (gloss or "").upper().strip()


def canonicalize_gloss(gloss: str, *, allowed: set[str] | None = None) -> str:
    """Strip sense id, apply synonym map, optionally snap into an allowed set."""
    stripped = strip_sense_suffix(gloss)
    mapped = SYNONYM_MAP.get(stripped, stripped)
    if allowed is None:
        return mapped
    if mapped in allowed:
        return mapped
    if stripped in allowed:
        return stripped
    compact = mapped.replace("-", "")
    for a in allowed:
        if a.replace("-", "") == compact:
            return a
    return mapped


def mapping_report(raw_labels: list[str], allowed: set[str]) -> dict:
    kept = 0
    synonym_hits = 0
    sense_only = 0
    orphan: list[str] = []
    for raw in raw_labels:
        stripped = strip_sense_suffix(raw)
        canon = canonicalize_gloss(raw, allowed=allowed)
        if canon in allowed:
            kept += 1
            if stripped in SYNONYM_MAP and SYNONYM_MAP[stripped] == canon:
                synonym_hits += 1
            elif canon == stripped and raw.upper().rstrip("0123456789").replace(" ", "-") != stripped:
                sense_only += 1
        else:
            orphan.append(f"{raw}->{canon}")
    return {
        "n_raw_unique": len(set(raw_labels)),
        "n_kept_in_allowed": kept,
        "n_synonym_hits": synonym_hits,
        "n_sense_strip_hits": sense_only,
        "n_orphan": len(orphan),
        "orphan_sample": sorted(set(orphan))[:40],
        "synonym_map_size": len(SYNONYM_MAP),
    }

