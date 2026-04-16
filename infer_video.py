from __future__ import annotations

import argparse

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from datasets.railsem19 import IMAGENET_MEAN, IMAGENET_STD
from models.ufld_rail import build_model
from train import load_cfg
from utils.anchor import build_anchor_rows, expectation_to_x
from utils.visualization import draw_rails


def preprocess(frame_bgr: np.ndarray, w: int, h: int) -> torch.Tensor:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rs = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
    f = rs.astype(np.float32) / 255.0
    f = (f - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(np.ascontiguousarray(f.transpose(2, 0, 1))).unsqueeze(0)


def apply_crop(frame_bgr: np.ndarray, crop: tuple[float, float, float, float]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Crop frame to ROI (relative coords: top, bottom, left, right in [0,1]).
    Returns (cropped_frame, (x0, y0, crop_w, crop_h) in original pixels).
    """
    h, w = frame_bgr.shape[:2]
    y0 = int(h * crop[0])
    y1 = int(h * crop[1])
    x0 = int(w * crop[2])
    x1 = int(w * crop[3])
    return frame_bgr[y0:y1, x0:x1].copy(), (x0, y0, x1 - x0, y1 - y0)


@torch.no_grad()
def infer_frame(
    model,
    frame_bgr: np.ndarray,
    cfg: dict,
    device,
    conf: float,
    crop: tuple[float, float, float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    iw, ih = cfg["input"]["width"], cfg["input"]["height"]
    ng = cfg["anchors"]["num_grid"]
    H = cfg["anchors"]["num_rows"]

    if crop is not None:
        roi, (cx0, cy0, cw, ch) = apply_crop(frame_bgr, crop)
    else:
        roi = frame_bgr
        cx0, cy0 = 0, 0
        ch, cw = frame_bgr.shape[:2]

    x = preprocess(roi, iw, ih).to(device)
    logits, _ = model(x)
    logits = logits[0]

    full_probs = F.softmax(logits, dim=-1).cpu().numpy()
    grid_probs = full_probs[..., :ng]
    no_rail_prob = full_probs[..., ng]

    exp_x_input = expectation_to_x(grid_probs, iw, ng)
    anchor_ys_input = build_anchor_rows(H, cfg["anchors"]["row_start"], cfg["anchors"]["row_end"], ih)

    sx = cw / iw
    sy = ch / ih
    pts = np.zeros((2, H, 2), dtype=np.float32)
    pts[..., 0] = exp_x_input * sx + cx0
    pts[..., 1] = anchor_ys_input[None, :] * sy + cy0

    valid = no_rail_prob < (1.0 - conf)
    return pts[0], pts[1], valid[0], valid[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="configs/default.yaml")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--ema", type=float, default=None)
    parser.add_argument("--crop", type=str, default=None,
                        help="ROI crop as top,bottom,left,right in [0,1]. "
                             "E.g. '0.05,0.65,0.2,0.75' to crop cockpit dashboard/frames.")
    args = parser.parse_args()

    cfg = load_cfg(args.cfg)
    conf = args.conf if args.conf is not None else cfg["infer"]["conf"]
    ema = args.ema if args.ema is not None else cfg["infer"]["ema"]
    crop = tuple(float(x) for x in args.crop.split(",")) if args.crop else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    prev_l = prev_r = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        lp, rp, lv, rv = infer_frame(model, frame, cfg, device, conf, crop=crop)
        if ema > 0 and prev_l is not None:
            lp = ema * prev_l + (1 - ema) * lp
            rp = ema * prev_r + (1 - ema) * rp
        prev_l, prev_r = lp, rp
        draw_rails(frame, lp, rp, lv, rv)
        writer.write(frame)

    cap.release()
    writer.release()
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
