from __future__ import annotations
import argparse
import json
from pathlib import Path

from ultralytics import YOLO

from .detect import detect_players_ball
from .possession import infer_possession


def draw_possession_box(image_path: str, bbox_xyxy, output_path: str) -> str:
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "OpenCV is required for --out_image. Install with: pip install opencv-python"
        ) from exc

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
    cv2.putText(
        img,
        "Ball Handler",
        (x1, max(20, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(out), img)
    if not ok:
        raise ValueError(f"Could not write overlay image: {output_path}")
    return str(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Path to a broadcast frame image")
    ap.add_argument("--model", default="models/yolo_players_ball.pt", help="YOLO weights (.pt)")
    ap.add_argument("--out_image", default=None, help="Optional output path for image overlay")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--iou", type=float, default=0.45)
    args = ap.parse_args()

    model = YOLO(args.model)

    players, ball = detect_players_ball(model, args.image, imgsz=args.imgsz, conf=args.conf, iou=args.iou)
    poss = infer_possession(players, ball)
    poss_bbox = list(players[poss.player_index].xyxy) if poss.player_index is not None else None

    overlay_image = None
    if args.out_image is not None and poss_bbox is not None:
        overlay_image = draw_possession_box(args.image, poss_bbox, args.out_image)

    out = {
        "image": args.image,
        "detections": {
            "num_players": len(players),
            "ball_detected": ball is not None,
            "ball_conf": float(ball.conf) if ball else None,
        },
        "possession": {
            "player_index": poss.player_index,
            "confidence": poss.confidence,
            "reason": poss.reason,
            "player_bbox_xyxy": poss_bbox,
        },
        "overlay_image": overlay_image,
    }

    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
