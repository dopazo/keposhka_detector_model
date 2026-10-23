"""
Checks an exported TFLite model the way the Android app must use it: manual letterbox + RGB/255 + raw output decoding
+ NMS, compared with the PyTorch model on the same images.

Usage (isolated env, TensorFlow has no wheel for Python 3.13):
    uv run --no-project --python 3.12 --with ultralytics==8.4.9 --with "tensorflow<=2.19.0" --with ai_edge_litert --with tf_keras \
        python scripts/check_tflite.py --tflite X.tflite --pt X.pt img1.png img2.jpg
"""

import argparse

import numpy as np
from PIL import Image, ImageOps
from ai_edge_litert.interpreter import Interpreter


def letterbox(im: Image.Image, size: int):
    w, h = im.size
    r = min(size / w, size / h)
    nw, nh = round(w * r), round(h * r)
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    px, py = (size - nw) // 2, (size - nh) // 2
    canvas.paste(im.resize((nw, nh), Image.BILINEAR), (px, py))
    return canvas, r, px, py


def nms(boxes, scores, thr):
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]; keep.append(i)
        x0 = np.maximum(boxes[i, 0], boxes[order[1:], 0]); y0 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        x1 = np.minimum(boxes[i, 2], boxes[order[1:], 2]); y1 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
        a = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        b = (boxes[order[1:], 2] - boxes[order[1:], 0]) * (boxes[order[1:], 3] - boxes[order[1:], 1])
        order = order[1:][inter / (a + b - inter + 1e-9) <= thr]
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tflite", required=True)
    ap.add_argument("--pt", required=True)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("images", nargs="+")
    args = ap.parse_args()

    it = Interpreter(model_path=args.tflite)
    it.allocate_tensors()
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    print("input ", inp["shape"], inp["dtype"].__name__, "| output", out["shape"], out["dtype"].__name__)
    size = int(inp["shape"][1])

    from ultralytics import YOLO
    pt = YOLO(args.pt)
    for path in args.images:
        im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        lb, r, px, py = letterbox(im, size)
        x = (np.asarray(lb, np.float32) / 255.0)[None]  # NHWC, RGB, 0..1
        it.set_tensor(inp["index"], x)
        it.invoke()
        o = it.get_tensor(out["index"])[0]  # [4+nc, N] or [N, 4+nc]
        if o.shape[0] < o.shape[1]:
            o = o.T
        xywh, cls = o[:, :4], o[:, 4:]
        scores, ids = cls.max(1), cls.argmax(1)
        m = scores >= args.conf
        xywh, scores, ids = xywh[m], scores[m], ids[m]
        scale = size if xywh.max() <= 1.5 else 1.0  # tflite export normalizes xywh to 0..1
        xywh = xywh * scale
        boxes = np.stack([xywh[:, 0] - xywh[:, 2] / 2, xywh[:, 1] - xywh[:, 3] / 2,
                          xywh[:, 0] + xywh[:, 2] / 2, xywh[:, 1] + xywh[:, 3] / 2], 1)
        keep = nms(boxes, scores, args.iou)
        boxes = (boxes[keep] - [px, py, px, py]) / r  # back to original image pixels
        ids, scores = ids[keep], scores[keep]

        res = pt.predict(path, conf=args.conf, iou=args.iou, imgsz=size, verbose=False)[0]
        pb, pc = res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy().astype(int)
        matched = 0
        for b, c in zip(boxes, ids):
            if len(pb):
                xx0 = np.maximum(b[0], pb[:, 0]); yy0 = np.maximum(b[1], pb[:, 1])
                xx1 = np.minimum(b[2], pb[:, 2]); yy1 = np.minimum(b[3], pb[:, 3])
                inter = np.clip(xx1 - xx0, 0, None) * np.clip(yy1 - yy0, 0, None)
                u = (b[2] - b[0]) * (b[3] - b[1]) + (pb[:, 2] - pb[:, 0]) * (pb[:, 3] - pb[:, 1]) - inter
                j = (inter / (u + 1e-9)).argmax()
                matched += int((inter / (u + 1e-9))[j] > 0.5 and pc[j] == c)
        print(f"{path}: tflite {len(boxes)} detections | pytorch {len(pb)} | tflite boxes matching pytorch (IoU>0.5, same class): {matched}")


if __name__ == "__main__":
    main()
