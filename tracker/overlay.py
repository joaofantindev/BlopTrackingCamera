from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np

from .tracks import Track

GREEN = (0, 255, 0)
DARK_GREEN = (0, 120, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


def color_for(track_id: int) -> tuple[int, int, int]:
    shades = [
        (0, 255, 0),
        (0, 200, 60),
        (0, 160, 90),
        (80, 255, 80),
        (0, 255, 160),
        (120, 220, 120),
    ]
    return shades[track_id % len(shades)]


def draw_tracks(frame: np.ndarray, tracks: Iterable[Track], show_trail: bool = True) -> None:
    for track in tracks:
        if track.missing > 0:
            continue
        color = color_for(track.id)
        x, y, w, h = track.bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        if show_trail and len(track.trail) > 1:
            points = np.array(track.trail, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(frame, [points], False, color, 2, cv2.LINE_AA)

        cx, cy = int(track.centroid[0]), int(track.centroid[1])
        cv2.circle(frame, (cx, cy), 4, WHITE, -1)

        label = f"ID {track.id}  ({cx},{cy})  v={track.speed:.0f}px"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x, y - th - 8), (x + tw + 6, y), DARK_GREEN, -1)
        cv2.rectangle(frame, (x, y - th - 8), (x + tw + 6, y), GREEN, 1)
        cv2.putText(frame, label, (x + 3, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)


def draw_hud(
    frame: np.ndarray,
    fps: float,
    tracked: int,
    detections: int,
    warmup: bool = False,
) -> None:
    lines = [
        f"FPS: {fps:5.1f}",
        f"Blobs: {detections}",
        f"Tracking: {tracked}",
    ]
    if warmup:
        lines.append("Calibrando fundo...")
    for i, text in enumerate(lines):
        y = 26 + i * 24
        cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, BLACK, 4, cv2.LINE_AA)
        cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, GREEN, 1, cv2.LINE_AA)
