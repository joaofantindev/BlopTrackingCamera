from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

from .tracks import BBox


def _green_colormap() -> np.ndarray:
    lut = np.zeros((256, 1, 3), dtype=np.uint8)
    for i in range(256):
        t = i / 255.0
        if t < 0.5:
            g = int(255 * (t / 0.5))
            lut[i, 0] = (0, g, 0)
        else:
            s = int(255 * ((t - 0.5) / 0.5))
            lut[i, 0] = (s, 255, s)
    return lut


_GREEN_LUT = _green_colormap()


class ActivityMap:
    """Accumulates where activity happens: a spatial heatmap + an occupancy grid."""

    def __init__(
        self,
        width: int,
        height: int,
        grid_cols: int = 16,
        grid_rows: int = 9,
        decay: float = 0.995,
    ) -> None:
        self.width = width
        self.height = height
        self.grid_cols = grid_cols
        self.grid_rows = grid_rows
        self.decay = decay
        self.heat = np.zeros((height, width), dtype=np.float32)
        self.grid = np.zeros((grid_rows, grid_cols), dtype=np.float32)
        self.hits = 0

    def update(self, detections: Sequence[BBox]) -> None:
        if self.decay < 1.0:
            self.heat *= self.decay
            self.grid *= self.decay
        for x, y, w, h in detections:
            cx, cy = x + w // 2, y + h // 2
            cv2.circle(self.heat, (cx, cy), max(w, h) // 2 + 2, 1.0, -1)
            col = min(int(cx / self.width * self.grid_cols), self.grid_cols - 1)
            row = min(int(cy / self.height * self.grid_rows), self.grid_rows - 1)
            self.grid[row, col] += 1.0
            self.hits += 1

    def heat_color(self) -> np.ndarray:
        peak = float(self.heat.max())
        if peak <= 0:
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)
        norm = np.clip(self.heat / peak, 0.0, 1.0)
        gray = (norm * 255).astype(np.uint8)
        return cv2.LUT(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), _GREEN_LUT)

    def overlay(self, frame: np.ndarray, alpha: float = 0.45) -> np.ndarray:
        color = self.heat_color()
        mask = self.heat > (self.heat.max() * 0.05 if self.heat.max() > 0 else 1)
        out = frame.copy()
        blended = cv2.addWeighted(frame, 1.0 - alpha, color, alpha, 0)
        out[mask] = blended[mask]
        return out

    def draw_grid(self, frame: np.ndarray, threshold: float = 0.25) -> np.ndarray:
        cw = self.width / self.grid_cols
        ch = self.height / self.grid_rows
        peak = float(self.grid.max()) or 1.0
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                x0, y0 = int(c * cw), int(r * ch)
                x1, y1 = int((c + 1) * cw), int((r + 1) * ch)
                ratio = self.grid[r, c] / peak
                if ratio > threshold:
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 200, 0), -1)
                    cv2.addWeighted(overlay, 0.18 * ratio, frame, 1 - 0.18 * ratio, 0, frame)
                cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 90, 0), 1)
        return frame

    def hotspots(self, top: int = 5) -> list[tuple[int, int]]:
        flat = self.grid.flatten()
        if flat.max() <= 0:
            return []
        idx = np.argsort(flat)[::-1][:top]
        return [(int(i // self.grid_cols), int(i % self.grid_cols)) for i in idx if flat[i] > 0]
