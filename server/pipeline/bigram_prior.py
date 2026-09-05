"""On-device-friendly bigram prior over glosses for top-k rerank.

Built from (1) synthetic conversational templates and (2) empirical
adjacent gloss co-occurrence in training metadata when available.
No network — prior is a static JSON / in-memory table.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

# Lightweight conversational templates (gloss bigrams friends actually use).
_TEMPLATES: list[list[str]] = [
    ["HELLO", "HOW", "YOU"],
    ["HI", "ME", "FINE"],
    ["HOW", "YOU"],
    ["YOU", "OK"],
    ["ME", "FINE"],
    ["ME", "GOOD"],
    ["ME", "TIRED"],
    ["ME", "HUNGRY"],
    ["ME", "WANT", "FOOD"],
    ["ME", "NEED", "HELP"],
    ["YOU", "WANT", "DRINK"],
    ["WHAT", "YOU", "WANT"],
    ["WHERE", "YOU", "GO"],
    ["WHEN", "YOU", "COME"],
    ["WHO", "THAT"],
    ["WHY", "YOU", "LEAVE"],
    ["PLEASE", "HELP"],
    ["THANKS", "YOU"],
    ["YES", "ME", "UNDERSTAND"],
    ["NO", "ME", "DONT-KNOW"],
    ["ME", "GO", "HOME"],
    ["ME", "GO", "SCHOOL"],
    ["SEE", "YOU", "LATER"],
    ["GOOD", "MORNING"],
    ["GOOD", "NIGHT"],
    ["ME", "LOVE", "YOU"],
    ["YOU", "LIKE", "MOVIE"],
    ["WANT", "WATCH", "MOVIE"],
    ["ME", "EAT", "FOOD"],
    ["TIME", "NOW"],
    ["ME", "CALL", "YOU"],
    ["YOU", "WRITE", "NAME"],
    ["WHAT", "YOUR", "NAME"],
    ["MY", "NAME"],
    ["ME", "DEAF"],
    ["YOU", "HEARING"],
    ["PLEASE", "SLOW"],
    ["AGAIN", "PLEASE"],
    ["ME", "DONT-KNOW"],
    ["THAT", "WRONG"],
    ["THAT", "RIGHT"],
    ["ME", "THINK", "YES"],
    ["MAYBE", "LATER"],
    ["SEE", "YOU", "TOMORROW"],
    ["YESTERDAY", "ME", "GO"],
    ["TODAY", "ME", "BUSY"],
    ["ME", "HAVE", "QUESTION"],
    ["CAN", "YOU", "HELP"],
    ["ME", "FINISH"],
    ["YOU", "UNDERSTAND"],
    ["ME", "WANT", "WATER"],
    ["GO", "BATHROOM"],
    ["ME", "NEED", "MONEY"],
    ["BUY", "FOOD"],
    ["PLAY", "WITH", "DOG"],
    ["ABOUT", "WHAT"],
    ["ME", "AND", "YOU"],
    ["BOY", "NAME"],
    ["PARTY", "WHEN"],
    ["CHRISTMAS", "PARTY"],

    ["ME", "WANT", "BOOK"],
    ["YOU", "NEED", "HELP"],
    ["WHY", "YOU", "GO"],
    ["WHO", "YOU"],
    ["DEAF", "ME"],
    ["DOG", "FINE"],
    ["ME", "DRINK", "WATER"],
    ["ME", "EAT", "APPLE"],
    ["ME", "EAT", "CANDY"],
    ["TIME", "LATER"],
    ["ME", "DECIDE"],
    ["COMPUTER", "WORK"],
    ["BEFORE", "NOW"],
    ["ME", "GO", "BATHROOM"],
    ["CLOTHES", "BLACK"],
    ["CLOTHES", "BLUE"],
    ["FAMILY", "MOTHER"],
    ["HEARING", "YOU"],
    ["HOT", "NOW"],
    ["NO", "ME", "WANT"],
    ["YES", "ME", "CAN"],
    ["CAN", "YOU", "HELP"],
]


def build_bigram_counts(labels: list[str] | None = None) -> dict[str, dict[str, float]]:
    """Return P(next|prev) unsmoothed counts then L1-normalize per prev."""
    allowed = set(labels) if labels else None
    counts: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for tmpl in _TEMPLATES:
        seq = [g for g in tmpl if allowed is None or g in allowed]
        for a, b in zip(seq, seq[1:]):
            counts[a][b] += 1.0
    # self-loop mild prior so rare glosses don't vanish
    if labels:
        for g in labels:
            counts[g][g] += 0.05
    # normalize
    out: dict[str, dict[str, float]] = {}
    for prev, nxt in counts.items():
        s = sum(nxt.values()) or 1.0
        out[prev] = {k: v / s for k, v in nxt.items()}
    return out


def save_prior(path: Path, labels: list[str]) -> dict:
    prior = build_bigram_counts(labels)
    payload = {"labels": labels, "bigram": prior, "unigram_floor": 1e-3}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return payload


def load_prior(path: Path) -> dict:
    return json.loads(path.read_text())


def rerank_topk(
    logits_row,
    labels: list[str],
    prev_gloss: str | None,
    prior: dict[str, dict[str, float]],
    *,
    k: int = 5,
    prior_weight: float = 0.35,
    unigram_floor: float = 1e-3,
) -> int:
    """Return class index after mixing model log-prob with log bigram prior.

    logits_row: 1D array-like of class logits.
    """
    import numpy as np

    logits = np.asarray(logits_row, dtype=np.float64)
    # numerical stable softmax
    m = logits.max()
    exp = np.exp(logits - m)
    probs = exp / exp.sum()
    top = np.argpartition(-probs, min(k, len(probs) - 1))[:k]
    if not prev_gloss or prev_gloss not in prior:
        return int(top[np.argmax(probs[top])])
    table = prior.get(prev_gloss, {})
    best_i = int(top[0])
    best_s = -1e9
    for i in top:
        i = int(i)
        g = labels[i]
        p_big = table.get(g, unigram_floor)
        score = (1.0 - prior_weight) * math.log(max(probs[i], 1e-12)) + prior_weight * math.log(max(p_big, 1e-12))
        if score > best_s:
            best_s = score
            best_i = i
    return best_i


def eval_with_bigram(logits, y, labels: list[str], prior: dict, k: int = 5, prior_weight: float = 0.35) -> dict:
    """Oracle-ish sequential eval: use gold previous label as context (upper bound signal),
    plus a blind chain using model prev predictions.
    """
    import numpy as np

    logits = np.asarray(logits)
    y = np.asarray(y)
    # gold-prev context (measures prior usefulness given correct history)
    correct_gold = 0
    correct_chain = 0
    correct_plain = 0
    prev_pred = None
    for i in range(len(y)):
        plain = int(logits[i].argmax())
        if plain == int(y[i]):
            correct_plain += 1
        gold_prev = labels[int(y[i - 1])] if i > 0 else None
        rg = rerank_topk(logits[i], labels, gold_prev, prior, k=k, prior_weight=prior_weight)
        if rg == int(y[i]):
            correct_gold += 1
        rc = rerank_topk(logits[i], labels, prev_pred, prior, k=k, prior_weight=prior_weight)
        if rc == int(y[i]):
            correct_chain += 1
        prev_pred = labels[rc]
    n = max(len(y), 1)
    return {
        "plain_top1": correct_plain / n,
        "bigram_gold_prev_top1": correct_gold / n,
        "bigram_chain_top1": correct_chain / n,
        "n": int(n),
        "prior_weight": prior_weight,
        "k": k,
    }
