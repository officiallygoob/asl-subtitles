#!/usr/bin/env python3
"""Convert WholeBodyPose COCO-135 HDF5 dumps → NPZ in FEATURE_DIM=170 layout.

Supports merging multiple license-clean pose packs from
https://huggingface.co/datasets/CristianLazoQuispe/pose-action-recognition
(MIT packaging of landmarks; underlying video rights stay with each dataset).

Gloss strings are canonicalized via pipeline.gloss_map (sense-suffix strip +
synonym/lemma map) so ASL Citizen rows merge into the training head.

Usage:
  python scripts/convert_pose_hdf5.py --sources wlasl100,aslcitizen100,wlasl300 --mix-synth --focus-wlasl100
  python scripts/convert_pose_hdf5.py --sources wlasl100,aslcitizen100,wlasl300 --mix-synth --daily-vocab
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.coco135 import coco135_sequence_to_features, pad_or_trim  # noqa: E402
from pipeline.gloss_map import canonicalize_gloss, mapping_report, strip_sense_suffix  # noqa: E402
from pipeline.vocab import DAILY_VOCAB, GLOSS_VOCAB  # noqa: E402

SOURCE_SPECS = {
    "wlasl100": ("wlasl100", "WLASL100_135"),
    "wlasl300": ("wlasl300", "WLASL300_135"),
    "aslcitizen100": ("aslcitizen100", "ASLCitizen100_135"),
}


def load_split(path: Path, frames: int, allowed: set[str] | None) -> tuple[list[np.ndarray], list[str], list[str]]:
    xs: list[np.ndarray] = []
    labels: list[str] = []
    raw_labels: list[str] = []
    with h5py.File(path, "r") as f:
        for key in sorted(f.keys(), key=lambda k: int(k) if k.isdigit() else k):
            g = f[key]
            raw = g["data"][:]  # (T, 2, 135)
            lab = g["label"][()]
            if isinstance(lab, bytes):
                lab = lab.decode("utf-8")
            raw_g = str(lab).strip()
            gloss = canonicalize_gloss(raw_g, allowed=allowed)
            if raw.shape[0] < 6:
                continue
            feat = coco135_sequence_to_features(raw.astype(np.float32))
            feat = pad_or_trim(feat, frames)
            xs.append(feat)
            labels.append(gloss)
            raw_labels.append(raw_g)
    return xs, labels, raw_labels


def resolve_split_paths(data_root: Path, folder: str, prefix: str) -> dict[str, Path]:
    d = data_root / folder
    return {
        "train": d / f"{prefix}-Train.hdf5",
        "val": d / f"{prefix}-Val.hdf5",
        "test": d / f"{prefix}-Test.hdf5",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=ROOT / "data")
    ap.add_argument(
        "--sources",
        default="wlasl100",
        help="Comma-separated: wlasl100,wlasl300,aslcitizen100",
    )
    ap.add_argument("--out", type=Path, default=ROOT / "models" / "pose_features.npz")
    ap.add_argument("--frames", type=int, default=32)
    ap.add_argument("--mix-synth", action="store_true")
    ap.add_argument("--synth-per-class", type=int, default=24)
    ap.add_argument(
        "--focus-wlasl100",
        action="store_true",
        help="Keep only GLOSS_VOCAB ∪ WLASL100 glosses (extra samples from other sources still help)",
    )
    ap.add_argument(
        "--daily-vocab",
        action="store_true",
        help="Closed-set daily conversational head (~DAILY_VOCAB, ~120–150)",
    )
    args = ap.parse_args()

    source_names = [s.strip().lower() for s in args.sources.split(",") if s.strip()]

    # Seed allowed set for synonym snapping. Expand with WLASL100 labels once loaded.
    w100_path = args.data_root / "wlasl100" / "wlasl_100_maplabels.json"
    w100_glosses: set[str] = set()
    if w100_path.exists():
        raw_map = json.loads(w100_path.read_text())
        w100_glosses = {strip_sense_suffix(k) for k in raw_map.get("id_to_label", {}).keys()}

    if args.daily_vocab:
        base_vocab = list(DAILY_VOCAB)
        keep_seed = set(DAILY_VOCAB)
    else:
        base_vocab = list(GLOSS_VOCAB)
        keep_seed = set(GLOSS_VOCAB) | w100_glosses

    # Pass 1 allowed = full conversational∪w100 so synonyms snap during load
    allowed_snap = set(GLOSS_VOCAB) | w100_glosses | set(DAILY_VOCAB)

    all_X: list[np.ndarray] = []
    all_y_gloss: list[str] = []
    all_raw: list[str] = []
    split_ids: list[str] = []
    source_ids: list[str] = []
    blockers: list[str] = []
    loaded: list[str] = []

    for name in source_names:
        if name not in SOURCE_SPECS:
            blockers.append(f"unknown source '{name}'")
            continue
        folder, prefix = SOURCE_SPECS[name]
        splits = resolve_split_paths(args.data_root, folder, prefix)
        missing = [str(p) for p in splits.values() if not p.exists()]
        if missing:
            blockers.append(f"{name}: missing files {missing}")
            print(f"SKIP {name}: missing {missing}", file=sys.stderr)
            continue
        for split, path in splits.items():
            xs, labs, raws = load_split(path, args.frames, allowed_snap)
            print(f"{name}/{split}: {len(xs)} from {path.name}")
            all_X.extend(xs)
            all_y_gloss.extend(labs)
            all_raw.extend(raws)
            split_ids.extend([split] * len(xs))
            source_ids.extend([name] * len(xs))
        loaded.append(name)

    if not all_X:
        print("No pose data loaded.", file=sys.stderr)
        for b in blockers:
            print(f"  blocker: {b}", file=sys.stderr)
        return 1

    report = mapping_report(all_raw, allowed_snap)
    print(
        f"gloss_map: kept_in_allowed={report['n_kept_in_allowed']} "
        f"synonym_hits≈{report['n_synonym_hits']} orphans={report['n_orphan']}"
    )

    real_glosses = sorted(set(all_y_gloss))
    if args.daily_vocab:
        keep_set = set(DAILY_VOCAB)
        # Also keep daily labels that appear after mapping even if not in seed list
        keep_set |= {g for g in real_glosses if g in DAILY_VOCAB}
        kept = [(x, g, sp, src) for x, g, sp, src in zip(all_X, all_y_gloss, split_ids, source_ids) if g in keep_set]
        all_X = [k[0] for k in kept]
        all_y_gloss = [k[1] for k in kept]
        split_ids = [k[2] for k in kept]
        source_ids = [k[3] for k in kept]
        real_glosses = sorted(set(all_y_gloss))
        print(f"daily-vocab: kept {len(all_X)} samples, glosses={len(real_glosses)}")
        union = list(dict.fromkeys([g for g in DAILY_VOCAB if g in set(real_glosses) or True]))
        # Prefer stable order: DAILY_VOCAB first, then any extras
        extras = [g for g in real_glosses if g not in set(DAILY_VOCAB)]
        union = list(dict.fromkeys(list(DAILY_VOCAB) + extras))
        # Drop daily glosses with zero samples after filter? Keep them for synth fill.
    elif args.focus_wlasl100:
        w100 = {g for g, s in zip(all_y_gloss, source_ids) if s == "wlasl100"}
        keep_set = set(GLOSS_VOCAB) | w100
        kept = [(x, g, sp, src) for x, g, sp, src in zip(all_X, all_y_gloss, split_ids, source_ids) if g in keep_set]
        all_X = [k[0] for k in kept]
        all_y_gloss = [k[1] for k in kept]
        split_ids = [k[2] for k in kept]
        source_ids = [k[3] for k in kept]
        real_glosses = sorted(set(all_y_gloss))
        print(f"focus-wlasl100: kept {len(all_X)} samples, glosses={len(real_glosses)}")
        union = list(dict.fromkeys(list(GLOSS_VOCAB) + real_glosses))
    else:
        union = list(dict.fromkeys(list(base_vocab) + real_glosses))

    if args.mix_synth:
        from scripts.synthesize_pose_dataset import synthesize_sequence

        rng = np.random.default_rng(11)
        target_vocab = list(DAILY_VOCAB) if args.daily_vocab else list(GLOSS_VOCAB)
        missing = [g for g in target_vocab if g not in set(real_glosses)]
        print(f"synth-fill {len(missing)} conversational glosses × {args.synth_per_class}")
        for g in missing:
            for _ in range(args.synth_per_class):
                T = int(rng.integers(24, min(40, args.frames + 8)))
                seq = synthesize_sequence(g, T, rng)
                all_X.append(pad_or_trim(seq, args.frames))
                all_y_gloss.append(g)
                split_ids.append("synth")
                source_ids.append("synth")
        # Ensure union includes synth targets
        union = list(dict.fromkeys(union + target_vocab))

    gloss_to_id = {g: i for i, g in enumerate(union)}
    # Drop any rows whose gloss somehow isn't in union (shouldn't happen)
    keep_idx = [i for i, g in enumerate(all_y_gloss) if g in gloss_to_id]
    if len(keep_idx) != len(all_y_gloss):
        all_X = [all_X[i] for i in keep_idx]
        all_y_gloss = [all_y_gloss[i] for i in keep_idx]
        split_ids = [split_ids[i] for i in keep_idx]
        source_ids = [source_ids[i] for i in keep_idx]

    y = np.array([gloss_to_id[g] for g in all_y_gloss], dtype=np.int64)
    X = np.stack(all_X, axis=0).astype(np.float32)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        X=X,
        y=y,
        labels=np.array(union),
        gloss=np.array(all_y_gloss),
        split=np.array(split_ids),
        source=np.array(source_ids),
    )
    legacy = args.out.parent / "wlasl100_features.npz"
    if args.out.resolve() != legacy.resolve() and not args.daily_vocab:
        np.savez_compressed(
            legacy,
            X=X,
            y=y,
            labels=np.array(union),
            gloss=np.array(all_y_gloss),
            split=np.array(split_ids),
            source=np.array(source_ids),
        )

    # Citizen contribution after mapping
    cit_mask = [s == "aslcitizen100" for s in source_ids]
    cit_glosses = sorted({g for g, m in zip(all_y_gloss, cit_mask) if m})

    meta = {
        "n_samples": int(len(y)),
        "n_classes": len(union),
        "frames": args.frames,
        "feature_dim": int(X.shape[-1]),
        "sources_requested": source_names,
        "sources_loaded": loaded,
        "blockers": blockers,
        "split_counts": {s: int(sum(1 for x in split_ids if x == s)) for s in sorted(set(split_ids))},
        "source_counts": {s: int(sum(1 for x in source_ids if x == s)) for s in sorted(set(source_ids))},
        "gloss_map": report,
        "citizen_mapped_glosses": cit_glosses,
        "n_citizen_rows": int(sum(cit_mask)),
        "mode": "daily" if args.daily_vocab else ("focus-wlasl100" if args.focus_wlasl100 else "full"),
        "packaging": "CristianLazoQuispe/pose-action-recognition (MIT landmark packaging)",
        "underlying_video": "WLASL research/C-UDA; ASL Citizen Microsoft research license — landmarks only stored",
        "how2sign": "skipped — continuous SLT shards large / different task; document as next continuous path",
    }
    meta_path = args.out.with_suffix(".meta.json")
    if args.out.name.endswith(".npz"):
        meta_path = args.out.parent / (args.out.stem + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))
    if not args.daily_vocab:
        (args.out.parent / "pose_features.meta.json").write_text(json.dumps(meta, indent=2))
        (args.out.parent / "wlasl100_features.meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {args.out} X={X.shape} classes={len(union)} loaded={loaded} citizen_rows={sum(cit_mask)}")
    if blockers:
        print("blockers:", blockers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
