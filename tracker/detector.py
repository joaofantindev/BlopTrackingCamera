from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

from .tracks import BBox


class BlobDetector:
    """Foreground blob detection via MOG2 background subtraction + morphology."""

    def __init__(
        self,
        min_area: int = 400,
        max_area: int = 200_000,
        learning_rate: float = -1.0,
        history: int = 500,
        var_threshold: float = 32.0,
        detect_shadows: bool = False,
        warmup: int = 30,
    ) -> None:
        self.min_area = min_area
        self.max_area = max_area
        self.learning_rate = learning_rate
        self.warmup = warmup
        self._frames = 0
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self._bg = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=detect_shadows,
        )

    @property
    def warming_up(self) -> bool:
        return self._frames < self.warmup

    def detect(self, frame: np.ndarray) -> tuple[list[BBox], np.ndarray]:
        blurred = cv2.GaussianBlur(frame, (7, 7), 0)
        mask = self._bg.apply(blurred, learningRate=self.learning_rate)
        self._frames += 1

        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel, iterations=3)
        mask = cv2.dilate(mask, self._kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections: list[BBox] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_area <= area <= self.max_area:
                detections.append(cv2.boundingRect(contour))
        return detections, mask
