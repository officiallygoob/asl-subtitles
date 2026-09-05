"""On-device-friendly bigram prior over glosses for top-k rerank.

Built from (1) synthetic conversational templates and (2) real-train
unigram weights (template edges scaled by sqrt(freq[a]*freq[b])).
Isolated-sign packs have no true multi-gloss transitions — we do NOT
invent gold-prev cheats; chain eval uses model predictions only for
the shipping metric narrative.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
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
    ["GIRL", "NAME"],
    ["PARTY", "WHEN"],
    ["CHRISTMAS", "PARTY"],
    ["HALLOWEEN", "PARTY"],
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
    ["CLOTHES", "YELLOW"],
    ["FAMILY", "MOTHER"],
    ["FAMILY", "FATHER"],
    ["HEARING", "YOU"],
    ["HOT", "NOW"],
    ["NO", "ME", "WANT"],
    ["YES", "ME", "CAN"],
    ["CAN", "YOU", "HELP"],
    ["HELLO", "MY", "NAME"],
    ["WHERE", "BATHROOM"],
    ["ME", "SORRY"],
    ["YOU", "PLEASE", "SLOW"],
    ["YOU", "HUNGRY"],
    ["WHAT", "TIME"],
    ["ME", "GO", "WORK"],
    ["YOU", "UNDERSTAND", "ME"],
    ["PLEASE", "AGAIN"],
    ["ME", "DEAF", "YOU", "HEARING"],
    ["FAMILY", "FINE"],
    ["MOTHER", "FATHER"],
    ["BOOK", "READ"],
    ["MOVIE", "WANT"],
]


def build_bigram_counts(
    labels: list[str] | None = None,
    *,
    train_labels: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    """Return P(next|prev). Template edges scaled by real-train unigram mass."""
    allowed = set(labels) if labels else None
    freq: Counter[str] = Counter()
    if train_labels:
        for g in train_labels:
            if allowed is None or g in allowed:
                freq[g] += 1
    elif labels:
        for g in labels:
            freq[g] += 1

    def w_edge(a: str, b: str) -> float:
        # Real-train mass: prefer edges whose endpoints appear in real pose.
        fa = max(freq.get(a, 0), 1)
        fb = max(freq.get(b, 0), 1)
        return math.sqrt(fa * fb)

    counts: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for tmpl in _TEMPLATES:
        seq = [g for g in tmpl if allowed is None or g in allowed]
        for a, b in zip(seq, seq[1:]):
            counts[a][b] += w_edge(a, b)
    # Mild self-loop + unigram continuation from real train (no invented pairs)
    if labels:
        total = sum(freq.values()) or 1
        for g in labels:
            counts[g][g] += 0.05
            # weak unigram backoff as continuation prior
            if freq:
                for h, c in freq.items():
                    if h == g:
                        continue
                    counts[g][h] += 0.02 * (c / total)
    out: dict[str, dict[str, float]] = {}
    for prev, nxt in counts.items():
        s = sum(nxt.values()) or 1.0
        out[prev] = {k: v / s for k, v in nxt.items()}
    return out


def save_prior(path: Path, labels: list[str], train_labels: list[str] | None = None) -> dict:
    prior = build_bigram_counts(labels, train_labels=train_labels)
    payload = {
        "labels": labels,
        "bigram": prior,
        "unigram_floor": 1e-3,
        "train_unigram": dict(Counter(train_labels)) if train_labels else {},
        "source": "templates×sqrt(train_unigram) + weak unigram backoff",
    }
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
    """Return class index after mixing model log-prob with log bigram prior."""
    import numpy as np

    logits = np.asarray(logits_row, dtype=np.float64)
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


def eval_with_bigram(
    logits,
    y,
    labels: list[str],
    prior: dict,
    k: int = 5,
    prior_weight: float = 0.35,
) -> dict:
    """Report plain / gold-prev (oracle upper) / chain (model-prev, honest)."""
    import numpy as np

    logits = np.asarray(logits)
    y = np.asarray(y)
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


def tune_prior_weight(
    logits,
    y,
    labels: list[str],
    prior: dict,
    *,
    k: int = 5,
    grid: tuple[float, ...] = (0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5),
) -> tuple[float, dict]:
    """Pick prior_weight maximizing honest chain top-1 (not gold-prev)."""
    best_w, best = 0.25, None
    for w in grid:
        m = eval_with_bigram(logits, y, labels, prior, k=k, prior_weight=w)
        if best is None or m["bigram_chain_top1"] > best["bigram_chain_top1"]:
            best_w, best = w, m
    assert best is not None
    return best_w, best
