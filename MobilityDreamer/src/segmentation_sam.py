"""segmentation_sam.py

Refine masks with Segment Anything (SAM) where available; otherwise fallback
to GrabCut using existing YOLO masks. Saves refined masks to an output dir.

Usage (from project root):
    python src/segmentation_sam.py --frames data/frames --masks data/masks \
        --out data/masks_refined --sam-checkpoint models/sam/sam_vit_b_01ec64.pth

Dependencies:
- segment-anything (optional, for SAM refinement)
- opencv-python
"""

import os
import sys
import argparse
from typing import Optional

import cv2
import numpy as np

SAM_AVAILABLE = False
sam_predictor = None

def try_init_sam(checkpoint_path: Optional[str]) -> None:
    """Initialize SAM predictor if checkpoint is provided and package is available."""
    global SAM_AVAILABLE, sam_predictor
    if not checkpoint_path:
        return
    try:
        from segment_anything import sam_model_registry, SamPredictor  # type: ignore
    except Exception:
        return
    try:
        sam = sam_model_registry.get("vit_b")(checkpoint=checkpoint_path)
        # SAM runs on CPU if no GPU; for CPU ensure torch available
        try:
            import torch  # noqa: F401
            device = "cuda" if hasattr(sam, "to") and hasattr(torch, "cuda") and torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
        sam.to(device)
        sam_predictor = SamPredictor(sam)
        SAM_AVAILABLE = True
    except Exception as e:
        print(f"WARN: SAM init failed: {e}")
        SAM_AVAILABLE = False


def refine_with_grabcut(image: np.ndarray, init_mask: np.ndarray, iters: int = 5) -> np.ndarray:
    """Refine a binary mask using OpenCV GrabCut.
    Ensures init_mask matches image size before seeding GrabCut.
    """
    h, w = image.shape[:2]
    # Resize init_mask to image size if needed
    if init_mask.shape[:2] != (h, w):
        init_mask = cv2.resize(init_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    mask_gc = np.zeros((h, w), np.uint8)
    # Use existing mask as foreground
    mask_gc[init_mask > 0] = cv2.GC_PR_FGD
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (1, 1, w - 2, h - 2)
    cv2.grabCut(image, mask_gc, rect, bgdModel, fgdModel, iters, cv2.GC_INIT_WITH_MASK)
    refined = np.where((mask_gc == cv2.GC_FGD) | (mask_gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    return refined


def refine_with_sam(image: np.ndarray, init_mask: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """Refine mask using SAM; if init_mask provided, sample points from mask as prompts."""
    global SAM_AVAILABLE, sam_predictor
    if not SAM_AVAILABLE or sam_predictor is None:
        return None
    try:
        sam_predictor.set_image(image)
        # Prompt selection: sample a few foreground points from init_mask or center
        if init_mask is not None and init_mask.any():
            ys, xs = np.where(init_mask > 0)
            if len(xs) == 0:
                pts = np.array([[image.shape[1] // 2, image.shape[0] // 2]])
            else:
                # sample up to 10 points
                idx = np.linspace(0, len(xs) - 1, num=min(10, len(xs)), dtype=int)
                pts = np.stack([xs[idx], ys[idx]], axis=1)
        else:
            pts = np.array([[image.shape[1] // 2, image.shape[0] // 2]])
        labels = np.ones((pts.shape[0],), dtype=np.int32)
        masks, _, _ = sam_predictor.predict(point_coords=pts, point_labels=labels, multimask_output=False)
        if masks is None or len(masks) == 0:
            return None
        m = masks[0]
        # SAM mask is boolean; convert to uint8 0/255
        return (m.astype(np.uint8) * 255)
    except Exception as e:
        print(f"WARN: SAM predict failed: {e}")
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Refine masks with SAM (if available) or GrabCut")
    parser.add_argument("--frames", default=os.path.join("data", "frames"), help="Input frames dir")
    parser.add_argument("--masks", default=os.path.join("data", "masks"), help="Input masks dir")
    parser.add_argument("--out", default=os.path.join("data", "masks_refined"), help="Output refined masks dir")
    parser.add_argument("--sam-checkpoint", default=None, help="Path to SAM checkpoint (e.g., models/sam/sam_vit_b_01ec64.pth)")
    args = parser.parse_args()

    if args.sam_checkpoint:
        try_init_sam(args.sam_checkpoint)

    if not os.path.exists(args.out):
        os.makedirs(args.out, exist_ok=True)

    names = sorted([n for n in os.listdir(args.frames) if n.lower().endswith((".jpg", ".jpeg", ".png"))])
    if not names:
        print(f"ERROR: no images found in {args.frames}", file=sys.stderr)
        sys.exit(1)

    refined_count = 0
    for n in names:
        fpath = os.path.join(args.frames, n)
        base = os.path.splitext(n)[0]
        mpath = os.path.join(args.masks, f"{base}_mask.png")
        outpath = os.path.join(args.out, f"{base}_mask.png")

        img = cv2.imread(fpath)
        if img is None:
            continue
        init_mask = cv2.imread(mpath, cv2.IMREAD_GRAYSCALE)
        # Ensure mask is single-channel uint8
        if init_mask is not None and init_mask.ndim == 3:
            init_mask = cv2.cvtColor(init_mask, cv2.COLOR_BGR2GRAY)

        refined = None
        if SAM_AVAILABLE:
            refined = refine_with_sam(img, init_mask)
        if refined is None and init_mask is not None:
            refined = refine_with_grabcut(img, init_mask)
        if refined is None:
            # If we can't refine, copy the original mask if present
            if init_mask is not None:
                refined = init_mask
            else:
                continue

        cv2.imwrite(outpath, refined)
        refined_count += 1

    print(f"Done. Wrote {refined_count} refined masks to {args.out}")


if __name__ == "__main__":
    main()
