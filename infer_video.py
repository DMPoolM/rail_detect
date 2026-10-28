from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from datasets.railsem19 import IMAGENET_MEAN, IMAGENET_STD
from models.ufld_rail import build_model
from utils.anchor import build_anchor_rows, expectation_to_x
from utils.config import load_cfg
from utils.temporal import RailSmoother
from utils.visualization import draw_rails

Crop = tuple[float, float, float, float]


def preprocess(frame_bgr: np.ndarray, w: int, h: int) -> torch.Tensor:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rs = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
    f = rs.astype(np.float32) / 255.0
    f = (f - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(np.ascontiguousarray(f.transpose(2, 0, 1))).unsqueeze(0)


def parse_crop(spec: str | None) -> Crop | None:
    """Parse ``--crop top,bottom,left,right`` (each in [0, 1]) with real errors.

    Previously a three-value spec reached ``apply_crop`` and died there with an
    unhelpful "not enough values to unpack" from inside the frame loop.
    """
    if spec is None:
        return None
    parts = [p.strip() for p in spec.split(",")]
    if len(parts) != 4:
        raise ValueError(
            f"--crop needs 4 comma-separated values (top,bottom,left,right), got {len(parts)}: {spec!r}"
        )
    try:
        top, bottom, left, right = (float(p) for p in parts)
    except ValueError as e:
        raise ValueError(f"--crop values must be numbers: {spec!r}") from e
    for name, v in (("top", top), ("bottom", bottom), ("left", left), ("right", right)):
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"--crop {name}={v} must be within [0, 1]")
    if not top < bottom:
        raise ValueError(f"--crop needs top < bottom, got {top} >= {bottom}")
    if not left < right:
        raise ValueError(f"--crop needs left < right, got {left} >= {right}")
    return (top, bottom, left, right)


def apply_crop(frame_bgr: np.ndarray, crop: Crop) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Crop frame to ROI (relative coords: top, bottom, left, right in [0,1]).

    Returns ``(cropped_frame, (x0, y0, crop_w, crop_h))`` in original pixels.
    """
    h, w = frame_bgr.shape[:2]
    y0 = int(h * crop[0])
    y1 = int(h * crop[1])
    x0 = int(w * crop[2])
    x1 = int(w * crop[3])
    if y1 - y0 < 2 or x1 - x0 < 2:
        raise ValueError(
            f"crop {crop} degenerates to {x1 - x0}x{y1 - y0} pixels on a {w}x{h} frame"
        )
    return frame_bgr[y0:y1, x0:x1].copy(), (x0, y0, x1 - x0, y1 - y0)


def resolve_device(spec: str = "auto") -> torch.device:
    if spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if spec == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    if spec == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("--device mps requested but MPS is unavailable")
    return torch.device(spec)


@torch.no_grad()
def infer_frame(
    model,
    frame_bgr: np.ndarray,
    cfg: dict,
    device,
    conf: float,
    crop: Crop | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(left_pts, right_pts, left_valid, right_valid)`` in frame coordinates."""
    iw, ih = cfg["input"]["width"], cfg["input"]["height"]
    ng = cfg["anchors"]["num_grid"]
    num_rows = cfg["anchors"]["num_rows"]

    if crop is not None:
        roi, (cx0, cy0, cw, ch) = apply_crop(frame_bgr, crop)
    else:
        roi = frame_bgr
        cx0, cy0 = 0, 0
        ch, cw = frame_bgr.shape[:2]

    x = preprocess(roi, iw, ih).to(device)
    logits, _ = model(x)
    logits = logits[0]

    full_probs = F.softmax(logits.float(), dim=-1).cpu().numpy()
    grid_probs = full_probs[..., :ng]
    no_rail_prob = full_probs[..., ng]

    exp_x_input = expectation_to_x(grid_probs, iw, ng)
    anchor_ys_input = build_anchor_rows(num_rows, cfg["anchors"]["row_start"], cfg["anchors"]["row_end"], ih)

    sx = cw / iw
    sy = ch / ih
    pts = np.zeros((2, num_rows, 2), dtype=np.float32)
    pts[..., 0] = exp_x_input * sx + cx0
    pts[..., 1] = anchor_ys_input[None, :] * sy + cy0

    valid = no_rail_prob < (1.0 - conf)
    return pts[0], pts[1], valid[0], valid[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw ego-rail polylines on a video.")
    parser.add_argument("--cfg", default="configs/default.yaml")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--ema", type=float, default=None)
    parser.add_argument(
        "--crop",
        type=str,
        default=None,
        help="ROI crop as top,bottom,left,right in [0,1]. "
        "E.g. '0.05,0.65,0.2,0.75' to crop out a cockpit dashboard/frame.",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--max-frames", type=int, default=None, help="stop early (quick preview / testing)")
    args = parser.parse_args()

    cfg = load_cfg(args.cfg)
    conf = args.conf if args.conf is not None else cfg["infer"]["conf"]
    ema = args.ema if args.ema is not None else cfg["infer"]["ema"]
    if not 0.0 < conf <= 1.0:
        parser.error(f"--conf must be in (0, 1], got {conf}")
    try:
        crop = parse_crop(args.crop)
    except ValueError as e:
        parser.error(str(e))

    device = resolve_device(args.device)
    model = build_model(cfg).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    if total and args.max_frames:
        total = min(total, args.max_frames)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"cannot open video writer for {out_path} (codec mp4v unavailable?)")

    left_smoother = RailSmoother(ema)
    right_smoother = RailSmoother(ema)

    n = 0
    pbar = tqdm(total=total, desc="infer", unit="frame")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        lp, rp, lv, rv = infer_frame(model, frame, cfg, device, conf, crop=crop)
        if ema > 0:
            lp = left_smoother.update(lp, lv)
            rp = right_smoother.update(rp, rv)
        draw_rails(frame, lp, rp, lv, rv)
        writer.write(frame)
        n += 1
        pbar.update(1)
        if args.max_frames and n >= args.max_frames:
            break

    pbar.close()
    cap.release()
    writer.release()
    print(f"wrote {out_path} ({n} frames @ {fps:.1f} fps)")


if __name__ == "__main__":
    main()
