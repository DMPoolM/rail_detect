"""Temporal smoothing for per-frame rail polylines.

The naive ``prev = ema * prev + (1 - ema) * cur`` has a failure mode: when a rail
disappears for a few frames and comes back somewhere else, the stale coordinates are
still in ``prev``, so the first frames after re-acquisition are dragged towards where
the rail *used* to be. This smoother blends only rows that were valid in both the
previous and the current frame; re-acquired rows snap to the new measurement.
"""

from __future__ import annotations

import numpy as np


class RailSmoother:
    """Exponential moving average over a (H, 2) point series with a validity mask."""

    def __init__(self, ema: float) -> None:
        if not 0.0 <= ema < 1.0:
            raise ValueError(f"ema must be in [0, 1), got {ema}")
        self.ema = float(ema)
        self._prev: np.ndarray | None = None
        self._prev_valid: np.ndarray | None = None

    def reset(self) -> None:
        self._prev = None
        self._prev_valid = None

    def update(self, pts: np.ndarray, valid: np.ndarray) -> np.ndarray:
        """Return the smoothed points; ``pts``/``valid`` are not modified."""
        valid = np.asarray(valid, dtype=bool)
        pts = np.asarray(pts, dtype=np.float32)
        if self.ema == 0.0 or self._prev is None or self._prev_valid is None:
            self._prev = pts.copy()
            self._prev_valid = valid.copy()
            return pts.copy()

        out = pts.copy()
        both = valid & self._prev_valid
        if both.any():
            out[both] = self.ema * self._prev[both] + (1.0 - self.ema) * pts[both]
        # Rows that are invalid now keep their raw (unused) values; rows that just
        # became valid are left unblended on purpose.
        self._prev = out.copy()
        self._prev_valid = valid.copy()
        return out
