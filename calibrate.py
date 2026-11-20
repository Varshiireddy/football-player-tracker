"""
Make field_points.json by clicking 4+ points on a frame whose real field position you know
(e.g. where yard lines meet the sideline or hash marks).

Usage:
    python calibrate.py --video clips/play.mp4 --frame 0

Click points in order, press any key when done, then type each point's field position in
yards as "x,y" (x = along the field, 0-120 incl. end zones; y = across, 0-53.3).
Only valid while the camera does not move.
"""

import argparse
import json

import cv2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--out", default="field_points.json")
    args = p.parse_args()

    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("Could not read that frame")

    clicks = []

    def on_click(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicks.append([x, y])
            cv2.circle(frame, (x, y), 6, (0, 0, 255), -1)
            cv2.putText(frame, str(len(clicks)), (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("calibrate", frame)

    cv2.imshow("calibrate", frame)
    cv2.setMouseCallback("calibrate", on_click)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(clicks) < 4:
        raise SystemExit("Need at least 4 points")

    field = []
    for i, (x, y) in enumerate(clicks, 1):
        fx, fy = input(f"Point {i} at pixel ({x},{y}) -> field yards x,y: ").split(",")
        field.append([float(fx), float(fy)])

    with open(args.out, "w") as f:
        json.dump({"pixel": clicks, "field_yards": field}, f, indent=2)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
