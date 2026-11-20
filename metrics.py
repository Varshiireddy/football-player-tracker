"""Pixel -> field mapping and per-player speed/distance stats."""

import json

import cv2
import numpy as np
import pandas as pd

YARDS_PER_SEC_TO_MPH = 3600 / 1760  # 1 yd/s = 2.045 mph
MAX_HUMAN_SPEED_YPS = 12.5          # ~25.5 mph; faster jumps are tracking noise


class FieldMapper:
    """Maps pixel coordinates to field coordinates.

    Modes:
      - homography: 4+ pixel points with known field positions (yards). Most accurate,
        but only valid while the camera doesn't move.
      - scale: a single yards-per-pixel number. Rough; ignores perspective.
      - pixels: no calibration; speeds are in pixels/second.
    """

    def __init__(self, H=None, scale=None):
        self.H, self.scale = H, scale
        self.units = "yards" if (H is not None or scale) else "pixels"

    @classmethod
    def from_args(cls, field_points_path, yards_per_pixel):
        if field_points_path:
            with open(field_points_path) as f:
                pts = json.load(f)
            src = np.array(pts["pixel"], dtype=np.float32)
            dst = np.array(pts["field_yards"], dtype=np.float32)
            if len(src) < 4 or len(src) != len(dst):
                raise ValueError("Need 4+ matching pixel / field_yards points")
            H, _ = cv2.findHomography(src, dst)
            return cls(H=H)
        if yards_per_pixel:
            return cls(scale=yards_per_pixel)
        return cls()

    def to_field(self, pixel_xy):
        pixel_xy = np.asarray(pixel_xy, dtype=np.float32).reshape(-1, 2)
        if self.H is not None:
            return cv2.perspectiveTransform(pixel_xy.reshape(-1, 1, 2), self.H).reshape(-1, 2)
        if self.scale:
            return pixel_xy * self.scale
        return pixel_xy

    def speed_label(self, speed):
        if self.units == "yards":
            return f"{speed * YARDS_PER_SEC_TO_MPH:.1f} mph"
        return f"{speed:.0f} px/s"


def summarize_tracks(tracks: pd.DataFrame, fps: float, mapper: FieldMapper, min_frames=15):
    """Per-player distance, max speed and average speed.

    Steps: drop short (flicker) tracks -> smooth positions -> frame-to-frame steps ->
    remove impossible jumps (ID switches / box jitter) -> aggregate.
    """
    cols = ["track_id", "frames", "seconds", f"distance_{mapper.units}", "max_speed", "avg_speed", "speed_units"]
    if tracks.empty:
        return pd.DataFrame(columns=cols)

    out = []
    window = max(3, int(fps // 5))  # ~0.2 s rolling mean to remove box jitter
    for tid, g in tracks.sort_values("frame").groupby("track_id"):
        if len(g) < min_frames:
            continue
        x = g["x_field"].rolling(window, center=True, min_periods=1).mean().to_numpy()
        y = g["y_field"].rolling(window, center=True, min_periods=1).mean().to_numpy()
        t = g["time_s"].to_numpy()

        step = np.hypot(np.diff(x), np.diff(y))
        dt = np.diff(t)
        inst_speed = np.divide(step, dt, out=np.zeros_like(step), where=dt > 0)

        if mapper.units == "yards":
            valid = inst_speed <= MAX_HUMAN_SPEED_YPS
            step, inst_speed = step[valid], inst_speed[valid]

        # Max speed over ~0.5 s windows is more honest than the single noisiest frame
        span = max(1, int(fps // 2))
        win_speed = pd.Series(inst_speed).rolling(span, min_periods=span).mean()
        max_speed = float(win_speed.max()) if win_speed.notna().any() else float(np.max(inst_speed, initial=0))
        seconds = float(t[-1] - t[0])
        distance = float(step.sum())
        avg_speed = distance / seconds if seconds > 0 else 0.0

        if mapper.units == "yards":
            max_speed *= YARDS_PER_SEC_TO_MPH
            avg_speed *= YARDS_PER_SEC_TO_MPH
            unit = "mph"
        else:
            unit = "px/s"

        out.append([int(tid), len(g), round(seconds, 2), round(distance, 2),
                    round(max_speed, 2), round(avg_speed, 2), unit])

    return pd.DataFrame(out, columns=cols).sort_values(f"distance_{mapper.units}", ascending=False)
