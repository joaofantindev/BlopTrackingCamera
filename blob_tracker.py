#!/usr/bin/env python3
"""Camera blob-tracking system: detect moving blobs, follow them with stable IDs,
and map activity over the whole scene (heatmap + occupancy grid)."""
from __future__ import annotations

import argparse
import time

import cv2

from tracker.detector import BlobDetector
from tracker.mapping import ActivityMap
from tracker.overlay import draw_hud, draw_tracks
from tracker.tracks import TrackManager


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Camera blob tracking and scene mapping")
    p.add_argument("--source", default="0", help="camera index (0) or video file path")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--min-area", type=int, default=400, help="minimum blob area in pixels")
    p.add_argument("--max-area", type=int, default=200000, help="maximum blob area in pixels")
    p.add_argument("--max-distance", type=float, default=80, help="max centroid jump to keep an ID")
    p.add_argument("--max-missing", type=int, default=30, help="frames before a lost track is dropped")
    p.add_argument("--history", type=int, default=40, help="trail length in frames")
    p.add_argument("--grid", default="16x9", help="occupancy grid size, e.g. 16x9")
    p.add_argument("--no-heatmap", action="store_true", help="disable activity heatmap")
    p.add_argument("--show-grid", action="store_true", help="draw occupancy grid cells")
    p.add_argument("--color", action="store_true", help="keep the color image (default: black & white)")
    p.add_argument("--record", default=None, help="save annotated output to a video file")
    p.add_argument("--display", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def open_source(source: str, width: int, height: int, fps: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        raise SystemExit(f"nao foi possivel abrir a fonte de video: {source}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    return cap


def main() -> None:
    args = parse_args()
    cap = open_source(args.source, args.width, args.height, args.fps)

    ok, frame = cap.read()
    if not ok:
        raise SystemExit("nao foi possivel ler o primeiro frame")
    h, w = frame.shape[:2]

    detector = BlobDetector(min_area=args.min_area, max_area=args.max_area)
    manager = TrackManager(
        max_distance=args.max_distance,
        max_missing=args.max_missing,
        history=args.history,
    )
    cols, rows = (int(v) for v in args.grid.lower().split("x"))
    activity = None if args.no_heatmap else ActivityMap(w, h, grid_cols=cols, grid_rows=rows)

    writer = None
    if args.record:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.record, fourcc, args.fps, (w, h))

    prev = time.time()
    fps = 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            detections, _mask = detector.detect(frame)

            if not args.color:
                frame = cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)

            tracks = manager.update(detections)

            if activity is not None:
                activity.update(detections)
                frame = activity.overlay(frame)
                if args.show_grid:
                    frame = activity.draw_grid(frame)

            draw_tracks(frame, tracks)
            now = time.time()
            dt = now - prev
            prev = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else 1.0 / dt
            draw_hud(frame, fps, len(manager.confirmed()), len(detections), detector.warming_up)

            if writer is not None:
                writer.write(frame)
            if args.display:
                cv2.imshow("blopTracking", frame)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if args.display:
            cv2.destroyAllWindows()

    if activity is not None:
        print(f"Total de deteccoes mapeadas: {activity.hits}")
        peaks = activity.hotspots(5)
        if peaks:
            print("Zonas mais ativas (linha,coluna):", peaks)


if __name__ == "__main__":
    main()
