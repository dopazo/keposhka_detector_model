"""
Evaluates one or more weights on the val set of a data.yaml and prints a compact comparison.

Usage:
    uv run python scripts/eval_holdout.py --data data_split_A/data.yaml w1.pt w2.pt [--imgsz 1280]
"""

import argparse

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("weights", nargs="+")
    args = ap.parse_args()

    print(f"{'weights':58s} mAP50  mAP50-95  P      R      classes_seen  AP50>0.5")
    for w in args.weights:
        m = YOLO(w).val(data=args.data, imgsz=args.imgsz, verbose=False, plots=False, workers=0)
        seen = list(m.ap_class_index)
        good = sum(1 for i, c in enumerate(seen) if m.box.ap50[i] > 0.5)
        print(f"{w:58s} {m.box.map50:.3f}  {m.box.map:.3f}    {m.box.mp:.3f}  {m.box.mr:.3f}  {len(seen):3d}           {good}")


if __name__ == "__main__":
    main()
