from __future__ import annotations

from collections import deque
from math import hypot
from typing import Sequence

BBox = tuple[int, int, int, int]
Point = tuple[float, float]


def bbox_centroid(bbox: BBox) -> Point:
    x, y, w, h = bbox
    return (x + w / 2.0, y + h / 2.0)


class Track:
    """A single blob being followed across frames."""

    _next_id = 1

    def __init__(self, bbox: BBox, history: int = 40) -> None:
        self.id = Track._next_id
        Track._next_id += 1
        self.bbox = bbox
        self.centroid = bbox_centroid(bbox)
        self.prev_centroid: Point | None = None
        self.missing = 0
        self.age = 1
        self.trail: deque[Point] = deque(maxlen=history)
        self.trail.append(self.centroid)

    def update(self, bbox: BBox) -> None:
        self.prev_centroid = self.centroid
        self.bbox = bbox
        self.centroid = bbox_centroid(bbox)
        self.trail.append(self.centroid)
        self.missing = 0
        self.age += 1

    @property
    def velocity(self) -> Point:
        if self.prev_centroid is None:
            return (0.0, 0.0)
        return (self.centroid[0] - self.prev_centroid[0], self.centroid[1] - self.prev_centroid[1])

    @property
    def speed(self) -> float:
        vx, vy = self.velocity
        return hypot(vx, vy)


class TrackManager:
    """Greedy nearest-centroid matcher that keeps stable IDs over time."""

    def __init__(
        self,
        max_distance: float = 80.0,
        max_missing: int = 30,
        history: int = 40,
        min_age: int = 1,
    ) -> None:
        self.max_distance = max_distance
        self.max_missing = max_missing
        self.history = history
        self.min_age = min_age
        self.tracks: list[Track] = []

    def update(self, detections: Sequence[BBox]) -> list[Track]:
        centroids = [bbox_centroid(b) for b in detections]

        pairs: list[tuple[float, int, int]] = []
        for ti, track in enumerate(self.tracks):
            for di, centroid in enumerate(centroids):
                dist = hypot(track.centroid[0] - centroid[0], track.centroid[1] - centroid[1])
                if dist <= self.max_distance:
                    pairs.append((dist, ti, di))
        pairs.sort(key=lambda p: p[0])

        used_tracks: set[int] = set()
        used_dets: set[int] = set()
        for _, ti, di in pairs:
            if ti in used_tracks or di in used_dets:
                continue
            self.tracks[ti].update(detections[di])
            used_tracks.add(ti)
            used_dets.add(di)

        for di, bbox in enumerate(detections):
            if di not in used_dets:
                self.tracks.append(Track(bbox, history=self.history))

        for ti, track in enumerate(self.tracks):
            if ti not in used_tracks:
                track.missing += 1

        self.tracks = [t for t in self.tracks if t.missing <= self.max_missing]
        return self.tracks

    def confirmed(self) -> list[Track]:
        return [t for t in self.tracks if t.age >= self.min_age and t.missing == 0]
