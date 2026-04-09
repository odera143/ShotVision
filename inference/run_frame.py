from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference.detect import detect_players_ball
from inference.paint_homography import draw_quad, fit_paint_homography
from inference.possession import infer_possession

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def mask_from_segmentation_result(result, image_shape: tuple[int, int]) -> np.ndarray:
    image_height, image_width = image_shape
    if result.masks is None or result.masks.data is None or len(result.masks.data) == 0:
        raise ValueError("No paint mask predicted.")

    masks = result.masks.data.cpu().numpy()
    areas = masks.sum(axis=(1, 2))
    best_index = int(np.argmax(areas))
    mask = (masks[best_index] > 0.5).astype(np.uint8) * 255
    if mask.shape != (image_height, image_width):
        mask = cv2.resize(mask, (image_width, image_height), interpolation=cv2.INTER_NEAREST)
    return mask


def _extract_xyxy(det_or_dict) -> tuple[int, int, int, int]:
    xyxy = det_or_dict.xyxy if hasattr(det_or_dict, "xyxy") else det_or_dict["xyxy"]
    x1, y1, x2, y2 = xyxy
    return int(x1), int(y1), int(x2), int(y2)


def render_overlay(
    image: np.ndarray,
    pos_bbox_xyxy,
    players,
    handler_court_xy=None,
    paint_quad=None,
    possession_label: str = "Ball Handler",
) -> np.ndarray:
    img = image.copy()

    # Draw ball handler bbox when possession is available.
    if pos_bbox_xyxy is not None:
        x1, y1, x2, y2 = [int(v) for v in pos_bbox_xyxy]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(
            img,
            possession_label,
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    # Draw all player bboxes
    for p in players:
        x1, y1, x2, y2 = _extract_xyxy(p)
        if pos_bbox_xyxy is not None and (x1, y1, x2, y2) == (
            int(pos_bbox_xyxy[0]),
            int(pos_bbox_xyxy[1]),
            int(pos_bbox_xyxy[2]),
            int(pos_bbox_xyxy[3]),
        ):
            continue  # Skip ball handler (already drawn)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
    
    # Mark all player foot positions
    for p in players:
        xyxy = p.xyxy if hasattr(p, "xyxy") else p["xyxy"]
        foot_x = int((xyxy[0] + xyxy[2]) / 2)
        foot_y = int(xyxy[3])
        cv2.circle(img, (foot_x, foot_y), 5, (255, 0, 255), -1)

    if paint_quad is not None:
        img = draw_quad(img, np.asarray(paint_quad, dtype=np.float32))

    if pos_bbox_xyxy is not None and handler_court_xy is not None:
        foot_x = int((pos_bbox_xyxy[0] + pos_bbox_xyxy[2]) / 2)
        foot_y = int(pos_bbox_xyxy[3])
        label = f"court=({handler_court_xy[0]:.2f}, {handler_court_xy[1]:.2f})"
        cv2.putText(
            img,
            label,
            (foot_x + 10, foot_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return img


def save_overlay_image(image: np.ndarray, output_path: str) -> str:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(out), image)
    if not ok:
        raise ValueError(f"Could not write overlay image: {output_path}")
    return str(out)


def draw_bounding_boxes(
    image_path: str,
    pos_bbox_xyxy,
    players,
    output_path: str,
    handler_court_xy=None,
    paint_quad=None,
) -> str:
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    overlay = render_overlay(
        img,
        pos_bbox_xyxy,
        players,
        handler_court_xy=handler_court_xy,
        paint_quad=paint_quad,
    )
    return save_overlay_image(overlay, output_path)


def infer_paint_homography_for_frame(
    model: YOLO,
    image_source,
    image: np.ndarray,
    basket_side: str,
    imgsz: int,
    conf: float,
    device: str,
):
    results = model.predict(
        source=image_source,
        imgsz=imgsz,
        conf=conf,
        device=device or None,
        verbose=False,
    )
    if not results:
        raise ValueError("No paint segmentation result returned.")

    mask = mask_from_segmentation_result(results[0], image.shape[:2])
    fit = fit_paint_homography(mask, basket_side=basket_side)
    return fit, mask


def load_models(player_model_path: str, paint_model_path: str = "") -> tuple[YOLO, YOLO | None]:
    model = YOLO(player_model_path)
    paint_model = YOLO(paint_model_path) if paint_model_path else None
    return model, paint_model


def process_frame(
    image_path: str,
    model: YOLO,
    imgsz: int = 1280,
    conf: float = 0.15,
    iou: float = 0.45,
    out_image: str | None = None,
    paint_model: YOLO | None = None,
    paint_basket_side: str = "",
    paint_imgsz: int = 1024,
    paint_conf: float = 0.25,
    device: str = "",
    image: np.ndarray | None = None,
) -> dict[str, Any]:
    frame = image if image is not None else cv2.imread(image_path, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError(f"Could not read image: {image_path}")

    image_source = frame if image is not None else image_path
    players, ball = detect_players_ball(model, image_source, imgsz=imgsz, conf=conf, iou=iou, device=device)
    poss = infer_possession(players, ball)
    poss_bbox = list(players[poss.player_index].xyxy) if poss.player_index is not None else None
    foot_x = (poss_bbox[0] + poss_bbox[2]) / 2 if poss_bbox is not None else None
    foot_y = poss_bbox[3] if poss_bbox is not None else None
    handler_court_xy = None
    paint_info = {
        "available": False,
        "reason": None,
        "basket_side": paint_basket_side or None,
        "image_points": None,
        "player_foot_court_xy": None,
    }
    paint_quad = None

    if paint_model is not None:
        if not paint_basket_side:
            raise ValueError("--paint-basket-side is required when --paint-model is provided.")
        try:
            fit, _mask = infer_paint_homography_for_frame(
                paint_model,
                image_source,
                frame,
                basket_side=paint_basket_side,
                imgsz=paint_imgsz,
                conf=paint_conf,
                device=device,
            )
            paint_quad = fit.image_points
            paint_info["available"] = True
            paint_info["image_points"] = fit.image_points.tolist()
            if foot_x is not None and foot_y is not None:
                handler_court_xy = fit.img_to_court(foot_x, foot_y)
                paint_info["player_foot_court_xy"] = list(handler_court_xy)
        except ValueError as exc:
            paint_info["reason"] = str(exc)

    overlay_image = None
    if out_image is not None:
        overlay = render_overlay(
            frame,
            poss_bbox,
            players,
            handler_court_xy=handler_court_xy,
            paint_quad=paint_quad,
            possession_label="Ball Handler",
        )
        overlay_image = save_overlay_image(overlay, out_image)

    return {
        "image": image_path,
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
            "player_foot_xy": (foot_x, foot_y) if foot_x is not None and foot_y is not None else None,
            "player_foot_court_xy": list(handler_court_xy) if handler_court_xy is not None else None,
        },
        "paint_homography": paint_info,
        "player_detections": [{
            "cls_id": p.cls_id,
            "conf": p.conf,
            "xyxy": p.xyxy,
            "foot_xy": ((p.xyxy[0] + p.xyxy[2]) / 2, p.xyxy[3])
        } for p in players],
        "overlay_image": overlay_image,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Path to a broadcast frame image")
    ap.add_argument("--model", default="models/yolo_players_ball.pt", help="YOLO weights (.pt)")
    ap.add_argument("--paint-model", default="", help="Optional YOLO segmentation weights for paint detection")
    ap.add_argument("--paint-basket-side", choices=["left", "right"], default="", help="Basket side for the visible half court")
    ap.add_argument("--paint-imgsz", type=int, default=1024)
    ap.add_argument("--paint-conf", type=float, default=0.25)
    ap.add_argument("--device", default="", help="Inference device, e.g. 0 or cpu")
    ap.add_argument("--out_image", default=None, help="Optional output path for image overlay")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--iou", type=float, default=0.45)
    args = ap.parse_args()

    model, paint_model = load_models(args.model, args.paint_model)
    out = process_frame(
        image_path=args.image,
        model=model,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        out_image=args.out_image,
        paint_model=paint_model,
        paint_basket_side=args.paint_basket_side,
        paint_imgsz=args.paint_imgsz,
        paint_conf=args.paint_conf,
        device=args.device,
    )

    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
