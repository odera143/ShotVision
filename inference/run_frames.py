from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference.run_frame import IMAGE_EXTS, load_models, process_frame, render_overlay, save_overlay_image
from inference.temporal_possession import TemporalPossessionState, smooth_possession


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full possession + paint pipeline on a folder of frames.")
    parser.add_argument("--source", required=True, help="Image file or folder of frame images.")
    parser.add_argument("--model", default="models/yolo_players_ball.pt", help="YOLO weights (.pt)")
    parser.add_argument("--paint-model", default="", help="Optional YOLO segmentation weights for paint detection")
    parser.add_argument("--paint-basket-side", choices=["left", "right"], default="", help="Basket side for the visible half court")
    parser.add_argument("--paint-imgsz", type=int, default=1024)
    parser.add_argument("--paint-conf", type=float, default=0.25)
    parser.add_argument("--device", default="", help="Inference device, e.g. 0 or cpu")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--output", default="runs/run-frames", help="Output directory for JSON results and optional overlays")
    parser.add_argument("--save-overlays", action="store_true", help="Write overlay images for each frame")
    parser.add_argument("--limit", type=int, default=0, help="Optional max number of frames to process")
    parser.add_argument("--hold-frames", type=int, default=8, help="How many uncertain frames to carry handler possession")
    parser.add_argument("--max-match-distance", type=float, default=140.0, help="Max pixel distance for nearest-player smoothing")
    return parser.parse_args()


def collect_images(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    return sorted([p for p in source.iterdir() if p.suffix.lower() in IMAGE_EXTS])


def main() -> None:
    args = parse_args()
    source = Path(args.source)
    image_paths = collect_images(source)
    if not image_paths:
        raise RuntimeError(f"No images found in {source}")
    if args.limit > 0:
        image_paths = image_paths[: args.limit]

    output_root = Path(args.output)
    json_dir = output_root / "json"
    overlay_dir = output_root / "overlays"
    json_dir.mkdir(parents=True, exist_ok=True)
    if args.save_overlays:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    model, paint_model = load_models(args.model, args.paint_model)
    results: list[dict] = []
    counts = {
        "frames_processed": 0,
        "ball_detected": 0,
        "possession_found": 0,
        "court_xy_found": 0,
        "paint_homography_available": 0,
    }
    state = TemporalPossessionState()

    for image_path in image_paths:
        result = process_frame(
            image_path=str(image_path),
            model=model,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            out_image=None,
            paint_model=paint_model,
            paint_basket_side=args.paint_basket_side,
            paint_imgsz=args.paint_imgsz,
            paint_conf=args.paint_conf,
            device=args.device,
        )
        smoothed_possession, state = smooth_possession(
            result,
            state,
            hold_frames=args.hold_frames,
            max_match_distance_px=args.max_match_distance,
        )
        result["smoothed_possession"] = smoothed_possession

        if args.save_overlays:
            frame = cv2.imread(str(image_path))
            if frame is None:
                raise RuntimeError(f"Could not read image for overlay: {image_path}")
            overlay = render_overlay(
                image=frame,
                pos_bbox_xyxy=smoothed_possession["player_bbox_xyxy"],
                players=result["player_detections"],
                handler_court_xy=smoothed_possession["player_foot_court_xy"],
                paint_quad=result["paint_homography"]["image_points"],
                possession_label="Ball Handler"
                if smoothed_possession["source"] != "smoothed"
                else "Ball Handler (smoothed)",
            )
            result["overlay_image"] = save_overlay_image(overlay, str(overlay_dir / image_path.name))

        results.append(result)
        (json_dir / f"{image_path.stem}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

        counts["frames_processed"] += 1
        counts["ball_detected"] += int(bool(result["detections"]["ball_detected"]))
        counts["possession_found"] += int(smoothed_possession["player_bbox_xyxy"] is not None)
        counts["court_xy_found"] += int(smoothed_possession["player_foot_court_xy"] is not None)
        counts["paint_homography_available"] += int(bool(result["paint_homography"]["available"]))

    summary = {
        "source": str(source),
        "output": str(output_root.resolve()),
        "counts": counts,
        "frames": results,
    }
    (output_root / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary["counts"], indent=2))
    print(f"Wrote per-frame JSON to: {json_dir.resolve()}")
    print(f"Wrote summary to: {(output_root / 'results.json').resolve()}")
    if args.save_overlays:
        print(f"Wrote overlays to: {overlay_dir.resolve()}")


if __name__ == "__main__":
    main()
