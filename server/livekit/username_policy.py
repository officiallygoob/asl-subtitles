"""Username rules for call tokens.

Keep this aligned with ``ASLSubtitles/Call/UsernameModerator.swift``.
The token server re-checks every name so a modified client cannot mint a token
for a username the app would refuse to save. Identity is a username, never a phone number.
"""

from __future__ import annotations

MIN_LENGTH = 3
MAX_LENGTH = 24

RESERVED = {
    "admin", "administrator", "admins", "mod", "moderator", "mods",
    "support", "helpsupport", "customersupport", "staff", "official",
    "aslsubtitles", "aslsubtitle", "asl", "system", "root", "owner",
    "null", "undefined", "everyone", "here", "channel", "server",
    "bot", "aslbot", "api", "security", "safety",
}

BLOCKED_TOKENS = {
    "fuck", "fucker", "fucking", "motherfucker", "mf", "mofo",
    "shit", "bullshit", "asshole", "asshat", "dumbass", "jackass",
    "bastard", "bitch", "bitches", "sonofabitch",
    "damn", "dammit", "crap", "piss", "pissed",
    "dick", "dickhead", "cock", "cocksucker", "prick",
    "pussy", "twat", "cunt",
    "whore", "slut", "skank", "hoe", "thot",
    "retard", "retarded", "idiot", "moron", "stupid",
    "fag", "faggot", "dyke", "tranny", "shemale",
    "pedo", "pedophile", "rapist", "rape", "raper",
    "nazi", "hitler", "kkk",
    "killmyself", "kys", "suicide",
    "porn", "porno", "xxx", "onlyfans",
    "cum", "jizz", "semen", "orgasm", "anal", "oral",
    "blowjob", "handjob", "fellatio", "cunnilingus",
    "dildo", "vibrator", "hentai", "nsfw",
    "nigger", "nigga", "negro", "chink", "gook", "spic", "wetback",
    "kike", "heeb", "paki", "raghead", "towelhead",
    "cracker", "honky", "gringo",
    "beaner", "coon", "darkie", "jigaboo", "porchmonkey",
    "redskin", "squaw", "injun",
    "gypsy", "gyppo",
    "ladyboy",
    "abeed", "kaffir", "kafir",
    "sex", "sexy", "nude", "nudes", "naked", "boobs", "tits", "titty",
    "penis", "vagina", "clit", "clitoris", "ballsack", "testicle",
    "masturbate", "wank", "wanker", "jerkoff",
    "incest", "zoophile", "bestiality",
    "terrorist", "isis", "alqaeda",
    "murder", "killer", "schoolshooter",
}

SEVERE_SUBSTRINGS = {
    "nigger", "nigga", "chink", "gook", "spic", "wetback",
    "kike", "paki", "raghead", "towelhead",
    "faggot", "tranny", "shemale",
    "retard",
    "cunt", "motherfucker", "cocksucker",
    "pedophile", "pedo",
    "rape", "rapist",
    "hitler", "nazi",
    "killmyself",
    "porchmonkey", "jigaboo", "beaner",
    "kaffir", "abeed",
}


def sanitize_format(raw: str) -> str:
    filtered = "".join(ch for ch in raw if ch.isalnum() or ch == "_")
    return filtered[:MAX_LENGTH]


def validate_username(raw: str) -> tuple[str, str]:
    """Return ``("ok", cleaned)`` or ``("rejected", reason)``."""
    cleaned = sanitize_format(raw.strip())

    if not cleaned:
        return "rejected", "Choose a username (3–24 letters, numbers, or _)."

    if len(cleaned) < MIN_LENGTH:
        return "rejected", f"Username must be at least {MIN_LENGTH} characters."

    if len(cleaned) > MAX_LENGTH:
        return "rejected", f"Username must be at most {MAX_LENGTH} characters."

    if not cleaned[0].isalpha():
        return "rejected", "Username must start with a letter."

    digits = sum(1 for ch in cleaned if ch.isdigit())
    letters = sum(1 for ch in cleaned if ch.isalpha())
    if digits >= 7 or (digits > 0 and digits > letters):
        return "rejected", "Usernames can’t look like phone numbers."

    for form in _normalized_forms(cleaned):
        if form in RESERVED:
            return "rejected", "That name is reserved."
        if form in BLOCKED_TOKENS:
            return "rejected", "That username isn’t allowed."
        for severe in SEVERE_SUBSTRINGS:
            if severe in form:
                return "rejected", "That username isn’t allowed."

    return "ok", cleaned


def _normalized_forms(cleaned: str) -> list[str]:
    lower = cleaned.lower().replace("_", "")
    mapped_i = _map_leet(lower, "i")
    mapped_l = _map_leet(lower, "l")
    if mapped_i == mapped_l:
        return [mapped_i]
    return [mapped_i, mapped_l]


def _map_leet(text: str, one_as: str) -> str:
    table = {
        "0": "o",
        "1": one_as,
        "3": "e",
        "4": "a",
        "5": "s",
        "7": "t",
        "@": "a",
        "$": "s",
    }
    return "".join(table.get(ch, ch) for ch in text)
