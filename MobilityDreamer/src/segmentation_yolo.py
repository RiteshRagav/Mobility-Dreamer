"""segmentation_yolo.py

Run YOLOv8 (CPU-friendly) on frames in `data/frames/` and save binary
segmentation masks to `data/masks/`. Optionally save color overlays to
`results/before_after/overlays/` for quick visualization.

Requirements:
- ultralytics (`pip install ultralytics`)
- torch CPU (optional but recommended for performance)

Usage (from project root):
    python src/segmentation_yolo.py --frames data/frames --out data/masks

Notes:
- Uses the lightweight segmentation model by default: `yolov8n-seg.pt`.
- If segmentation model is unavailable, will fallback to detection (`yolov8n.pt`)
  and produce rectangle masks from bounding boxes.
"""

import os
import sys
import argparse
from typing import Optional, Tuple

import numpy as np
import cv2

try:
    from ultralytics import YOLO
except Exception as e:  # pragma: no cover
    YOLO = None


def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def load_model(model_name: str) -> Optional[YOLO]:
    if YOLO is None:
        print("ERROR: ultralytics not available. Install with `pip install ultralytics`.", file=sys.stderr)
        return None
    try:
        return YOLO(model_name)
    except Exception as e:
        print(f"WARN: Failed to load {model_name}: {e}")
        return None


def render_segmentation_mask(pred) -> Optional[np.ndarray]:
    """Render a binary mask from YOLOv8 segmentation predictions.
    Returns a single-channel uint8 mask (0/255) or None.
    """
    masks = getattr(pred, "masks", None)
    if masks is None or masks.data is None:
        return None
    # masks.data: (N, H, W) or (N, downsampled) depending on model; use .xy to get polygons when needed
    try:
        # Convert instance masks to a single union mask
        # masks.data is a torch tensor; convert to numpy
        data = masks.data.cpu().numpy()  # (N, H, W) floats {0,1}
        union = (data.sum(axis=0) > 0).astype(np.uint8) * 255
        return union
    except Exception:
        return None


def render_detection_mask(pred, shape: Tuple[int, int]) -> Optional[np.ndarray]:
    """Fallback: render rectangle masks from detection boxes.
    `shape` is (height, width) of the original image.
    """
    boxes = getattr(pred, "boxes", None)
    if boxes is None or boxes.xyxy is None:
        return None
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    try:
        xyxy = boxes.xyxy.cpu().numpy()  # (N,4)
        for x1, y1, x2, y2 in xyxy:
            x1i = max(0, min(w - 1, int(x1)))
            y1i = max(0, min(h - 1, int(y1)))
            x2i = max(0, min(w - 1, int(x2)))
            y2i = max(0, min(h - 1, int(y2)))
            mask[y1i:y2i, x1i:x2i] = 255
        return mask
    except Exception:
        return None


def process_frame(model: YOLO, img_path: str, out_mask_path: str, conf: float = 0.25) -> bool:
    img = cv2.imread(img_path)
    if img is None:
        print(f"WARN: failed to read image: {img_path}")
        return False

    # Run inference with CPU (device='cpu')
    try:
        results = model.predict(source=img, conf=conf, device="cpu", verbose=False)
    except Exception as e:
        print(f"ERROR: inference failed on {img_path}: {e}")
        return False

    if not results:
        print(f"WARN: no predictions for {img_path}")
        return False

    pred = results[0]
    mask = render_segmentation_mask(pred)
    if mask is None:
        mask = render_detection_mask(pred, (img.shape[0], img.shape[1]))

    if mask is None:
        print(f"WARN: no mask produced for {img_path}")
        return False

    ensure_dir(os.path.dirname(out_mask_path))
    cv2.imwrite(out_mask_path, mask)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLOv8 segmentation over frames → masks")
    parser.add_argument("--frames", default=os.path.join("data", "frames"), help="Directory of input frames (JPG/PNG)")
    parser.add_argument("--out", default=os.path.join("data", "masks"), help="Directory to save output masks")
    parser.add_argument("--model", default="yolov8n-seg.pt", help="YOLOv8 model name or path (default: yolov8n-seg.pt)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--save-overlay", action="store_true", help="Save colored overlays to results/before_after/overlays/")
    args = parser.parse_args()

    ensure_dir(args.out)

    # Try segmentation model, fallback to detection if needed
    model = load_model(args.model)
    if model is None:
        print("Attempting fallback to detection model 'yolov8n.pt'...")
        model = load_model("yolov8n.pt")
    if model is None:
        print("ERROR: Failed to load any YOLO model. Aborting.", file=sys.stderr)
        sys.exit(1)

    # Process all images in frames dir
    exts = (".jpg", ".jpeg", ".png")
    names = sorted([n for n in os.listdir(args.frames) if n.lower().endswith(exts)])
    if not names:
        print(f"ERROR: no images found in {args.frames}", file=sys.stderr)
        sys.exit(1)

    ok_count = 0
    overlay_dir = os.path.join("results", "before_after", "overlays") if args.save_overlay else None
    if overlay_dir:
        ensure_dir(overlay_dir)
    for n in names:
        in_path = os.path.join(args.frames, n)
        base = os.path.splitext(n)[0]
        out_path = os.path.join(args.out, f"{base}_mask.png")
        # Run inference
        img = cv2.imread(in_path)
        if img is None:
            continue
        try:
            results = model.predict(source=img, conf=args.conf, device="cpu", verbose=False)
        except Exception as e:
            print(f"ERROR: inference failed on {in_path}: {e}")
            continue

        if not results:
            continue
        pred = results[0]

        # Save mask (segmentation preferred, else detection rectangles)
        mask = render_segmentation_mask(pred)
        if mask is None:
            mask = render_detection_mask(pred, (img.shape[0], img.shape[1]))
        if mask is not None:
            ensure_dir(os.path.dirname(out_path))
            cv2.imwrite(out_path, mask)
            ok_count += 1

        # Optional overlay
        if overlay_dir:
            overlay_img = img.copy()
            # If we have instance masks with classes, color by class
            masks = getattr(pred, "masks", None)
            boxes = getattr(pred, "boxes", None)
            names = getattr(pred, "names", None)
            class_colors = {}
            if masks is not None and masks.data is not None and boxes is not None and boxes.cls is not None:
                try:
                    mdata = masks.data.cpu().numpy()  # (N,H,W)
                    cls_idx = boxes.cls.cpu().numpy().astype(int)  # (N,)
                    H, W = img.shape[:2]
                    # Resize masks to image size if needed
                    if mdata.shape[1] != H or mdata.shape[2] != W:
                        mdata = np.stack([cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) for m in mdata], axis=0)
                    for i, m in enumerate(mdata):
                        c = cls_idx[i] if i < len(cls_idx) else -1
                        if c not in class_colors:
                            # deterministic color per class id
                            rng = np.random.default_rng(c)
                            class_colors[c] = tuple(int(x) for x in rng.integers(0, 255, size=3))
                        color = class_colors[c]
                        mask_bin = m > 0.5
                        overlay = np.zeros_like(overlay_img)
                        overlay[mask_bin] = color
                        overlay_img = cv2.addWeighted(overlay, 0.5, overlay_img, 0.5, 0)
                except Exception:
                    # Fallback: use binary mask if available
                    if mask is not None:
                        overlay = np.zeros_like(overlay_img)
                        overlay[mask > 0] = (0, 255, 0)
                        overlay_img = cv2.addWeighted(overlay, 0.5, overlay_img, 0.5, 0)
            else:
                # Only detection rectangles available
                if boxes is not None and boxes.xyxy is not None:
                    try:
                        xyxy = boxes.xyxy.cpu().numpy()
                        cls_idx = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else np.zeros(len(xyxy), dtype=int)
                        for i, (x1, y1, x2, y2) in enumerate(xyxy):
                            c = cls_idx[i] if i < len(cls_idx) else -1
                            if c not in class_colors:
                                rng = np.random.default_rng(c)
                                class_colors[c] = tuple(int(x) for x in rng.integers(0, 255, size=3))
                            color = class_colors[c]
                            cv2.rectangle(overlay_img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    except Exception:
                        pass

            ov_out = os.path.join(overlay_dir, f"{base}_overlay.jpg")
            cv2.imwrite(ov_out, overlay_img)

    msg = f"Done. Wrote {ok_count} masks to {args.out}"
    if overlay_dir:
        msg += f" and overlays to {overlay_dir}"
    print(msg)


if __name__ == "__main__":
    main()
