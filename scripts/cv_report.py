"""
Out-of-fold report of the 2 grouped CV models (cv0, cv1, see make_cv_folds.py): every image is predicted by the model
that did not train on it or on other photos of the same sign. Prints per-image results, totals by image type and by
glyph size as seen by the network (box height after the resize to imgsz).

Usage:
    uv run python scripts/cv_report.py [--conf 0.25] [--imgsz 1280]
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).parent))
from error_breakdown import iou, read_labels  # noqa: E402
from make_cv_folds import STAGE  # noqa: E402

STAGE_SET = {s for g in STAGE for s in g}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=1280)
    args = ap.parse_args()

    rows = []
    for k in range(2):
        model = YOLO(f"runs/detect/runs/cv{k}/weights/last.pt")
        val = Path(f"data_split_cv{k}/val")
        for img in sorted((val / "images").iterdir()):
            r = model.predict(str(img), conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
            h, w = r.orig_shape
            truth = read_labels(val / "labels" / f"{img.stem}.txt", w, h)
            preds = [(int(c), b) for c, b in zip(r.boxes.cls.tolist(), r.boxes.xyxy.tolist())]
            used, cnt = set(), Counter()
            for tc, tb in truth:
                cand = [(iou(tb, pb), i) for i, (pc, pb) in enumerate(preds) if i not in used]
                best, bi = max(cand, default=(0, None))
                if bi is None or best < 0.5:
                    cnt["missed"] += 1
                else:
                    used.add(bi)
                    cnt["correct" if preds[bi][0] == tc else "wrong"] += 1
            cnt["fp"] = len(preds) - len(used)
            scale = min(1.0, args.imgsz / max(h, w)) if max(h, w) > args.imgsz else args.imgsz / max(h, w)
            glyph_h = float(np.median([(b[3] - b[1]) * scale for _, b in truth]))
            rows.append(dict(name=img.stem[:34], fold=k, kind="stage" if img.stem in STAGE_SET else "other",
                             n=len(truth), glyph=glyph_h, **{x: cnt[x] for x in ("correct", "wrong", "missed", "fp")}))

    print(f"{'image':34s} fold kind   boxes  glyph_px  correct wrong missed  fp")
    for r in sorted(rows, key=lambda r: (r["kind"], r["name"])):
        print(f"{r['name']:34s} {r['fold']:4d} {r['kind']:6s} {r['n']:5d}  {r['glyph']:7.0f}  {r['correct']:7d} {r['wrong']:5d} {r['missed']:6d} {r['fp']:3d}")

    def agg(sel, label):
        if not sel:
            return
        n = sum(r["n"] for r in sel); c = sum(r["correct"] for r in sel); fp = sum(r["fp"] for r in sel)
        wr = sum(r["wrong"] for r in sel); ms = sum(r["missed"] for r in sel)
        macro = np.mean([r["correct"] / r["n"] for r in sel]) if sel else float("nan")
        print(f"{label:28s} images {len(sel):2d} boxes {n:4d} | correct {c / n:5.1%} wrong-class {wr / n:5.1%} missed {ms / n:5.1%} "
              f"| precision {c / max(c + wr + fp, 1):5.1%} | mean of per-image correct {macro:5.1%}")

    print()
    agg(rows, "ALL")
    agg([r for r in rows if r["kind"] == "stage"], "stage (rule screens)")
    agg([r for r in rows if r["kind"] == "other"], "other")
    print("\nby glyph height seen by the network (px at imgsz):")
    for lo, hi in ((0, 15), (15, 25), (25, 40), (40, 1e9)):
        agg([r for r in rows if lo <= r["glyph"] < hi], f"glyph {lo}-{int(hi) if hi < 1e9 else '+'} px")


if __name__ == "__main__":
    main()
