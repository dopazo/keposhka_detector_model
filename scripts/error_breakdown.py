"""
Splits detection errors into: correct / right place but wrong class / missed / false positive.

Usage:
    uv run python scripts/error_breakdown.py --weights w.pt --split data_split_A/val [--conf 0.25] [--train data_split_B/train]
Matching ignores class: each true box takes the unused prediction with the highest IoU >= 0.5.
With --train, also prints recall per class next to its number of training boxes.
"""

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from ultralytics import YOLO


def iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def predict_tiled(model, img, conf, imgsz, tile=1280, stride=960, min_side=1800):
    """Full image (downscaled by the model) + native-resolution tiles of big images, merged with class-agnostic NMS."""
    import torch
    from torchvision.ops import nms
    from PIL import Image
    r = model.predict(str(img), conf=conf, imgsz=imgsz, verbose=False)[0]
    h, w = r.orig_shape
    boxes, scores, classes = [r.boxes.xyxy.cpu()], [r.boxes.conf.cpu()], [r.boxes.cls.cpu()]
    if max(h, w) > min_side:
        im = Image.open(img).convert("RGB")
        ys = list(range(0, max(h - tile, 0) + 1, stride)); ys += [h - tile] if ys[-1] + tile < h else []
        xs = list(range(0, max(w - tile, 0) + 1, stride)); xs += [w - tile] if xs[-1] + tile < w else []
        for y0 in ys:
            for x0 in xs:
                t = model.predict(im.crop((x0, y0, min(x0 + tile, w), min(y0 + tile, h))), conf=conf, imgsz=imgsz, verbose=False)[0]
                b = t.boxes.xyxy.cpu().clone(); b[:, [0, 2]] += x0; b[:, [1, 3]] += y0
                boxes.append(b); scores.append(t.boxes.conf.cpu()); classes.append(t.boxes.cls.cpu())
    boxes, scores, classes = torch.cat(boxes), torch.cat(scores), torch.cat(classes)
    keep = nms(boxes, scores, 0.5)
    return (h, w), [(int(classes[i]), boxes[i].tolist()) for i in keep]


def read_labels(path, w, h):
    out = []
    for l in Path(path).read_text().splitlines():
        if l.strip():
            c, cx, cy, bw, bh = l.split()
            cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
            out.append((int(c), (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--split", required=True, type=Path, help="folder with images/ and labels/")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--tile", action="store_true", help="also run native-resolution tiles on images with a side > 1800")
    ap.add_argument("--train", type=Path, default=None)
    args = ap.parse_args()

    model = YOLO(args.weights)
    names = model.names
    total, pairs = Counter(), Counter()
    per_class = defaultdict(lambda: [0, 0])  # class -> [true boxes, correct]
    for img in sorted((args.split / "images").iterdir()):
        if args.tile:
            (h, w), preds = predict_tiled(model, img, args.conf, args.imgsz)
        else:
            r = model.predict(str(img), conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
            h, w = r.orig_shape
            preds = [(int(c), b) for c, b in zip(r.boxes.cls.tolist(), r.boxes.xyxy.tolist())]
        truth = read_labels(args.split / "labels" / f"{img.stem}.txt", w, h)
        used, cnt = set(), Counter()
        for tc, tb in truth:
            best, bi = 0.0, None
            for i, (pc, pb) in enumerate(preds):
                if i in used:
                    continue
                v = iou(tb, pb)
                if v > best:
                    best, bi = v, i
            per_class[tc][0] += 1
            if bi is None or best < 0.5:
                cnt["missed"] += 1
            else:
                used.add(bi)
                if preds[bi][0] == tc:
                    cnt["correct"] += 1
                    per_class[tc][1] += 1
                else:
                    cnt["wrong class"] += 1
                    pairs[(names[tc], names[preds[bi][0]])] += 1
        cnt["false positive"] = len(preds) - len(used)
        print(f"{img.name[:40]:40s} true {len(truth):3d} | " + " ".join(f"{k}={cnt[k]}" for k in ("correct", "wrong class", "missed", "false positive")))
        total.update(cnt)
    print("TOTAL", dict(total))
    print("wrong-class pairs (true->pred):", pairs.most_common(15))

    train_n = Counter()
    if args.train:
        for p in (args.train / "labels").glob("*.txt"):
            for l in p.read_text().splitlines():
                if l.strip():
                    train_n[int(l.split()[0])] += 1
    print("\nclass  train_boxes  test_boxes  recall")
    for c, (n, ok) in sorted(per_class.items(), key=lambda kv: train_n[kv[0]]):
        print(f"{names[c]:>5s}  {train_n[c]:10d}  {n:10d}  {ok / n:6.2f}")
    if args.train:
        xs = np.array([train_n[c] for c in per_class]); ys = np.array([per_class[c][1] / per_class[c][0] for c in per_class])
        w = np.array([per_class[c][0] for c in per_class])
        for lo, hi in ((0, 20), (20, 60), (60, 10**9)):
            m = (xs >= lo) & (xs < hi)
            if m.any():
                print(f"classes with {lo}-{hi if hi < 10**9 else '+'} train boxes: {m.sum()} classes, weighted recall {np.average(ys[m], weights=w[m]):.2f}")


if __name__ == "__main__":
    main()
