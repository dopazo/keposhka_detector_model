"""
Builds a split from explicit image lists (no shuffling), so experiments are comparable.

Usage:
    uv run python scripts/prepare_holdout_split.py --out data_split_A --test 48emimjuxsnf1 exercise_book_first_page \
        --drop keposhka_abecedario letra_L --oversample 3 keposhka_abecedario2
    uv run python scripts/prepare_holdout_split.py --out data_split_final --test none   # train on everything

- The output folder is deleted first: files from a previous split can never leak into this one.
- --test: stems held out as val (use "none" for no hold-out: val then points at a copy of the train images,
  only useful to keep the training script running).
- --drop: stems left out of train and val.
- --oversample N STEM...: the stems go N times into train (copies named STEM__dupK). Ultralytics has no class-balanced
  sampling, and copy_paste is a no-op without segmentation masks, so this is the cheap way to weight rare glyphs.
"""

import argparse
import shutil
from pathlib import Path

import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--test", nargs="+", required=True)
    ap.add_argument("--drop", nargs="*", default=[])
    ap.add_argument("--oversample", nargs="*", default=[], help="N STEM [STEM ...]")
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    n_dup = int(args.oversample[0]) if args.oversample else 1
    dup_stems = set(args.oversample[1:])
    test = set() if args.test == ["none"] else set(args.test)

    images = {p.stem: p for p in Path("data/images").iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
    unknown = (test | set(args.drop) | dup_stems) - set(images)
    if unknown:
        raise SystemExit(f"unknown stems: {sorted(unknown)}")

    def put(split: str, stem: str, new_stem: str):
        (out / split / "images").mkdir(parents=True, exist_ok=True)
        (out / split / "labels").mkdir(parents=True, exist_ok=True)
        src = images[stem]
        shutil.copy2(src, out / split / "images" / f"{new_stem}{src.suffix}")
        shutil.copy2(Path("data/labels") / f"{stem}.txt", out / split / "labels" / f"{new_stem}.txt")

    for stem in sorted(images):
        if stem in args.drop or not (Path("data/labels") / f"{stem}.txt").exists():
            continue
        if stem in test:
            put("val", stem, stem)
            continue
        put("train", stem, stem)
        if stem in dup_stems:
            for k in range(1, n_dup):
                put("train", stem, f"{stem}__dup{k}")
    if not test:  # no hold-out: val is a copy of a few train images
        for stem in sorted(images)[:3]:
            put("val", stem, stem)

    names = [l.strip() for l in Path("data/classes.txt").read_text().splitlines() if l.strip()]
    (out / "data.yaml").write_text(yaml.safe_dump({
        "path": str(out.resolve()), "train": "train/images", "val": "val/images",
        "nc": len(names), "names": names}, allow_unicode=True), encoding="utf-8")
    for s in ("train", "val"):
        n = len(list((out / s / "images").glob("*"))) if (out / s / "images").exists() else 0
        print(f"{s}: {n} images")


if __name__ == "__main__":
    main()
