"""
Class-balanced synthetic training images made of real glyph crops.

Why: the real data has 3-9 examples for ~25 of the 70 classes, and Ultralytics' copy_paste does nothing without
segmentation masks. Here every class is sampled equally often: glyph crops (the labeled boxes of the TRAIN split
only, so no hold-out image leaks) are pasted in text-like lines on backgrounds taken from train photos with the
labeled text painted out. Labels are the pasted rectangles, i.e. the same box style as the real labels.

Usage:
    uv run python scripts/make_synthetic.py --split data_split_C --n 150 [--seed 0]
Adds syn_XXXX.jpg/.txt to <split>/train/images|labels.
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

SIZE = 1280
MAX_COLOR_DIST = 45  # BGR distance between a crop's own background and the colour of the spot it is pasted on


def ring_color(crop):
    """Mean colour of the outer 20% border of a crop: the glyph's own background."""
    h, w = crop.shape[:2]
    m = np.ones((h, w), bool)
    m[int(h * .2):h - int(h * .2), int(w * .2):w - int(w * .2)] = False
    return crop[m].mean(0)


def pick_crop(options, patch_color, max_dist):
    """Crop of this class whose own background is closest to the colour where it will be pasted."""
    best = min(options, key=lambda o: np.linalg.norm(o[1] - patch_color))
    return best[0] if np.linalg.norm(best[1] - patch_color) <= max_dist else None


def load_train(split: Path):
    crops, bgs = {}, []
    for img_path in sorted((split / "train" / "images").iterdir()):
        if "__dup" in img_path.stem or img_path.stem.startswith("syn_"):
            continue
        im = cv2.imread(str(img_path))
        lbl = split / "train" / "labels" / f"{img_path.stem}.txt"
        H, W = im.shape[:2]
        boxes = []
        for line in lbl.read_text().splitlines():
            c, cx, cy, w, h = line.split()
            c, cx, cy, w, h = int(c), float(cx) * W, float(cy) * H, float(w) * W, float(h) * H
            x0, y0, x1, y1 = (int(round(v)) for v in (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
            boxes.append((c, x0, y0, x1, y1))
            if min(x1 - x0, y1 - y0) >= 10 and x0 >= 0 and y0 >= 0 and x1 <= W and y1 <= H:
                crop = im[y0:y1, x0:x1].copy()
                crops.setdefault(c, []).append((crop, ring_color(crop)))
        bgs.append((im, boxes))
    return crops, bgs


def make_background(bgs, rng):
    """Random window of a train photo with all labeled glyph boxes painted out."""
    im, boxes = bgs[rng.randrange(len(bgs))]
    H, W = im.shape[:2]
    scale = SIZE / min(H, W) * rng.uniform(1.0, 2.0)  # smallest side >= SIZE; varied zoom on the source
    im = cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    H, W = im.shape[:2]
    mask = np.zeros((H, W), np.uint8)
    for _, x0, y0, x1, y1 in boxes:
        px, py = 0.15 * (x1 - x0), 0.15 * (y1 - y0)
        cv2.rectangle(mask, (int(x0 * scale - px), int(y0 * scale - py)), (int(x1 * scale + px), int(y1 * scale + py)), 255, -1)
    H, W = im.shape[:2]
    x, y = rng.randint(0, W - SIZE), rng.randint(0, H - SIZE)
    win, wmask = im[y:y + SIZE, x:x + SIZE].copy(), mask[y:y + SIZE, x:x + SIZE]
    if (wmask > 0).mean() > 0.35:  # mostly erased text: the inpainting would look like a smear
        return make_background(bgs, rng)
    if wmask.any():
        win = cv2.inpaint(win, wmask, 5, cv2.INPAINT_TELEA)
    return win


def paste(canvas, crop, x, y, w, h, rng):
    crop = cv2.resize(crop, (w, h), interpolation=cv2.INTER_AREA if w < crop.shape[1] else cv2.INTER_LINEAR)
    alpha = np.ones((h, w), np.float32)
    k = max(1, min(w, h) // 8)  # feathered border: no hard rectangle edge to learn
    alpha[:k, :] *= np.linspace(0.3, 1, k)[:, None]; alpha[-k:, :] *= np.linspace(1, 0.3, k)[:, None]
    alpha[:, :k] *= np.linspace(0.3, 1, k)[None]; alpha[:, -k:] *= np.linspace(1, 0.3, k)[None]
    gain, bias = rng.uniform(0.8, 1.2), rng.uniform(-20, 20)
    crop = np.clip(crop.astype(np.float32) * gain + bias, 0, 255)
    region = canvas[y:y + h, x:x + w].astype(np.float32)
    canvas[y:y + h, x:x + w] = (alpha[..., None] * crop + (1 - alpha[..., None]) * region).astype(np.uint8)


def make_image(crops, bgs, classes, rng):
    canvas = make_background(bgs, rng)
    labels, occupied = [], np.zeros((SIZE, SIZE), bool)
    for _ in range(rng.randint(1, 3)):  # text blocks
        gh = rng.randint(24, 90)  # glyph height for this block
        n_lines, per_line = rng.randint(1, 5), rng.randint(4, 12)
        x0, y0 = rng.randint(20, SIZE // 2), rng.randint(20, SIZE - 40)
        for li in range(n_lines):
            x, y = x0, y0 + int(li * gh * rng.uniform(1.4, 1.9))
            for _ in range(per_line):
                c = rng.choice(classes)
                h = int(gh * rng.uniform(0.85, 1.15))
                w0 = max(8, int(h * 0.8))
                yy = y + rng.randint(-3, 3)
                if x + w0 + h >= SIZE - 5 or yy + h >= SIZE - 5 or yy < 5:
                    break
                patch = canvas[yy:yy + h, x:x + w0 + h // 2].reshape(-1, 3).mean(0)
                crop = pick_crop(crops[c], patch, MAX_COLOR_DIST)
                if crop is None:  # no example of this class fits the local colours: try another class
                    continue
                w = max(8, int(h * crop.shape[1] / crop.shape[0]))
                if x + w >= SIZE - 5 or occupied[yy:yy + h, x:x + w].any():
                    break
                paste(canvas, crop, x, yy, w, h, rng)
                occupied[yy:yy + h, x:x + w] = True
                labels.append((c, x + w / 2, yy + h / 2, w, h))
                x += w + int(h * rng.uniform(0.1, 0.45))
    if rng.random() < 0.5:
        canvas = cv2.GaussianBlur(canvas, (0, 0), rng.uniform(0.4, 1.2))
    noise = np.random.default_rng(rng.randrange(1 << 30)).normal(0, rng.uniform(0, 4), canvas.shape)
    return np.clip(canvas + noise, 0, 255).astype(np.uint8), labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, type=Path)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    crops, bgs = load_train(args.split)
    classes = sorted(crops)
    print(f"{len(classes)} classes with crops, {sum(len(v) for v in crops.values())} crops, {len(bgs)} backgrounds")
    n_boxes = 0
    for i in range(args.n):
        img, labels = make_image(crops, bgs, classes, rng)
        stem = f"syn_{i:04d}"
        cv2.imwrite(str(args.split / "train" / "images" / f"{stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        (args.split / "train" / "labels" / f"{stem}.txt").write_text("".join(
            f"{c} {cx / SIZE:.6f} {cy / SIZE:.6f} {w / SIZE:.6f} {h / SIZE:.6f}\n" for c, cx, cy, w, h in labels))
        n_boxes += len(labels)
    print(f"wrote {args.n} images, {n_boxes} boxes")


if __name__ == "__main__":
    main()
