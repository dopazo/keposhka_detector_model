"""
Adds native-resolution tiles of the large TRAIN images to a split, so glyphs are seen at their real size
(at imgsz 1280 a 2252x4000 photo is downscaled ~3x and glyphs shrink to ~25 px).

Usage:
    uv run python scripts/make_tiles.py --split data_split_D [--tile 1280] [--stride 960] [--min-side 1800]
Originals stay in train; tiles are added as <stem>__tXX_YY. A box is kept (clipped) if >= 60% of it lies in the tile;
tiles with fewer than 3 boxes are skipped. Only train images are tiled (val is never touched).
"""

import argparse
from pathlib import Path

from PIL import Image


def starts(n: int, tile: int, stride: int) -> list[int]:
    out = list(range(0, max(n - tile, 0) + 1, stride))
    if out[-1] + tile < n:
        out.append(n - tile)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, type=Path)
    ap.add_argument("--tile", type=int, default=1280)
    ap.add_argument("--stride", type=int, default=960)
    ap.add_argument("--min-side", type=int, default=1800, help="tile images whose longest side exceeds this")
    args = ap.parse_args()

    n_tiles = n_boxes = 0
    for img_path in sorted((args.split / "train" / "images").iterdir()):
        if "__dup" in img_path.stem or "__t" in img_path.stem:
            continue
        im = Image.open(img_path)
        W, H = im.size
        if max(W, H) <= args.min_side:
            continue
        boxes = []
        for l in (args.split / "train" / "labels" / f"{img_path.stem}.txt").read_text().splitlines():
            c, cx, cy, w, h = l.split()
            cx, cy, w, h = float(cx) * W, float(cy) * H, float(w) * W, float(h) * H
            boxes.append((int(c), cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
        for y0 in starts(H, args.tile, args.stride):
            for x0 in starts(W, args.tile, args.stride):
                x1, y1 = min(x0 + args.tile, W), min(y0 + args.tile, H)
                lines = []
                for c, bx0, by0, bx1, by1 in boxes:
                    ix0, iy0, ix1, iy1 = max(bx0, x0), max(by0, y0), min(bx1, x1), min(by1, y1)
                    if ix1 <= ix0 or iy1 <= iy0 or (ix1 - ix0) * (iy1 - iy0) < 0.6 * (bx1 - bx0) * (by1 - by0):
                        continue
                    tw, th = x1 - x0, y1 - y0
                    lines.append(f"{c} {((ix0 + ix1) / 2 - x0) / tw:.6f} {((iy0 + iy1) / 2 - y0) / th:.6f} "
                                 f"{(ix1 - ix0) / tw:.6f} {(iy1 - iy0) / th:.6f}")
                if len(lines) < 3:
                    continue
                stem = f"{img_path.stem}__t{y0:04d}_{x0:04d}"
                im.crop((x0, y0, x1, y1)).convert("RGB").save(args.split / "train" / "images" / f"{stem}.jpg", quality=95)
                (args.split / "train" / "labels" / f"{stem}.txt").write_text("\n".join(lines) + "\n")
                n_tiles += 1
                n_boxes += len(lines)
    print(f"added {n_tiles} tiles, {n_boxes} boxes")


if __name__ == "__main__":
    main()
