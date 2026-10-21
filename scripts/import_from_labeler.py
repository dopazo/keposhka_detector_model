"""
Copies annotations marked "done" in the labeling app (sibling project keposhka_labeling) into data/.

- Only images that have a YOLO .txt in the labeler and are not yet in data/labels/ are copied.
- Aborts if the labeler's classes.txt differs from data/classes.txt (a shifted class id would corrupt every box).
- Applies the EXIF rotation to the pixels and saves WITHOUT EXIF: the labeler normalises boxes on the rotated image,
  so the copy must hold exactly those pixels (no dependence on how each reader handles EXIF).
- CROPS removes unlabeled text from an image: glyphs without a box are trained as background.

Usage:
    uv run python scripts/import_from_labeler.py [--labeler ../keposhka_labeling/backend/data] [--dry-run]
"""

import argparse
import shutil
from pathlib import Path

from PIL import Image, ImageOps

# stem -> (x0, y0, x1, y1) in pixels of the (rotated) image. Boxes outside are dropped, the rest renormalised.
CROPS = {
    # collage: keep the title block, drop the strip of screen paragraphs at the right edge (unlabeled glyphs)
    "mcr_keposhka_guide": (0, 0, 1400, 1122),
}

# Not imported: the image is mostly unlabeled glyphs (only the header row of a screen was boxed)
EXCLUDE = {"mcr_keposhka_guide_screen"}


def crop_labels(lines: list[str], w: int, h: int, box: tuple[int, int, int, int]) -> list[str]:
    x0, y0, x1, y1 = box
    out = []
    for line in lines:
        c, cx, cy, bw, bh = line.split()
        cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
        if not (x0 <= cx - bw / 2 and cx + bw / 2 <= x1 and y0 <= cy - bh / 2 and cy + bh / 2 <= y1):
            raise SystemExit(f"a box crosses the crop edge ({c} at {cx:.0f},{cy:.0f}): adjust CROPS")
        out.append(f"{c} {(cx - x0) / (x1 - x0):.6f} {(cy - y0) / (y1 - y0):.6f} "
                   f"{bw / (x1 - x0):.6f} {bh / (y1 - y0):.6f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeler", type=Path, default=Path("../keposhka_labeling/backend/data"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    ann_dir, img_dir = args.labeler / "annotations", args.labeler / "images"
    if (ann_dir / "classes.txt").read_text() != Path("data/classes.txt").read_text():
        raise SystemExit("classes.txt differs between the labeler and data/: refusing to copy")

    existing = {p.stem for p in Path("data/labels").glob("*.txt")}
    for txt in sorted(ann_dir.glob("*.txt")):
        stem = txt.stem
        if stem == "classes" or stem in existing or stem in EXCLUDE:
            continue
        src = next((p for p in img_dir.glob(f"{stem}.*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}), None)
        if src is None:
            print(f"  {stem}: no image, skipped")
            continue
        im = ImageOps.exif_transpose(Image.open(src))
        lines = [l for l in txt.read_text().splitlines() if l.strip()]
        if stem in CROPS:
            lines = crop_labels(lines, im.width, im.height, CROPS[stem])
            im = im.crop(CROPS[stem])
        print(f"  {stem}: {src.suffix} {im.width}x{im.height}, {len(lines)} boxes")
        if args.dry_run:
            continue
        dst = Path("data/images") / src.name
        if src.suffix.lower() == ".png":
            im.save(dst)
        else:
            im.convert("RGB").save(dst, quality=95)  # no exif= argument: the orientation tag is dropped
        Path("data/labels", f"{stem}.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
