import numpy as np


def build_anchor_rows(num_rows: int, row_start: float, row_end: float, img_h: int) -> np.ndarray:
    ys = np.linspace(row_start, row_end, num_rows, endpoint=True) * img_h
    return ys.astype(np.float32)


def x_to_grid(x: np.ndarray, img_w: int, num_grid: int) -> np.ndarray:
    bin_w = img_w / num_grid
    g = np.floor(x / bin_w).astype(np.int64)
    return np.clip(g, 0, num_grid - 1)


def grid_to_x(g: np.ndarray, img_w: int, num_grid: int) -> np.ndarray:
    bin_w = img_w / num_grid
    return (g.astype(np.float32) + 0.5) * bin_w


def expectation_to_x(probs: np.ndarray, img_w: int, num_grid: int) -> np.ndarray:
    """probs: (..., num_grid) softmax over the grid classes (without the "no-rail" bin).
    Returns expected x in pixels."""
    centers = (np.arange(num_grid, dtype=np.float32) + 0.5) * (img_w / num_grid)
    return (probs * centers).sum(axis=-1)


def interp_polyline_at_ys(points: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Linearly interpolate x at given y values along a polyline (must be monotonic in y after sort).

    points: (N, 2) array of (x, y)
    ys: (M,) query y's
    Returns: (M,) x values, NaN where ys is outside the polyline's y range.
    """
    if points.shape[0] < 2:
        return np.full_like(ys, np.nan, dtype=np.float32)
    order = np.argsort(points[:, 1])
    pts = points[order]
    py = pts[:, 1]
    px = pts[:, 0]
    py_unique, idx = np.unique(py, return_index=True)
    px_unique = px[idx]
    if py_unique.shape[0] < 2:
        return np.full_like(ys, np.nan, dtype=np.float32)
    out = np.full_like(ys, np.nan, dtype=np.float32)
    in_range = (ys >= py_unique[0]) & (ys <= py_unique[-1])
    out[in_range] = np.interp(ys[in_range], py_unique, px_unique).astype(np.float32)
    return out
