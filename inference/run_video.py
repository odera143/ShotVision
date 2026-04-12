from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference.run_frame import load_models, process_frame, render_overlay
from inference.temporal_possession import TemporalPossessionState, smooth_possession


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full possession + paint pipeline on a video.")
    parser.add_argument("--video", required=True, help="Path to the input video file.")
    parser.add_argument("--model", default="models/yolo_players_ball.pt", help="YOLO weights (.pt)")
    parser.add_argument("--paint-model", default="", help="Optional YOLO segmentation weights for paint detection")
    parser.add_argument("--paint-basket-side", choices=["left", "right"], default="", help="Basket side for the visible half court")
    parser.add_argument("--paint-imgsz", type=int, default=1024)
    parser.add_argument("--paint-conf", type=float, default=0.25)
    parser.add_argument("--device", default="", help="Inference device, e.g. 0 or cpu")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--output", default="runs/run-video", help="Output directory for JSON and overlay video")
    parser.add_argument("--frame-step", type=int, default=1, help="Process every Nth frame")
    parser.add_argument("--max-frames", type=int, default=0, help="Optional max number of processed frames")
    parser.add_argument("--no-overlay-video", action="store_true", help="Skip writing the overlayed output video")
    parser.add_argument("--hold-frames", type=int, default=8, help="How many uncertain frames to carry handler possession")
    parser.add_argument("--max-match-distance", type=float, default=140.0, help="Max pixel distance for nearest-player smoothing")
    return parser.parse_args()


def create_video_writer(output_path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc_factory = getattr(cv2, "VideoWriter_fourcc", None)
    if fourcc_factory is None:
        fourcc_factory = getattr(cv2.VideoWriter, "fourcc", None)
    if fourcc_factory is None:
        raise RuntimeError("OpenCV build does not expose a fourcc helper.")

    writer = cv2.VideoWriter(
        str(output_path),
        int(fourcc_factory(*"mp4v")),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to create video writer: {output_path}")
    return writer


def main() -> None:
    args = parse_args()
    video_path = Path(args.video)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0:
        fps = 30.0

    output_root = Path(args.output)
    json_dir = output_root / "json"
    json_dir.mkdir(parents=True, exist_ok=True)
    video_writer = None
    overlay_video_path = output_root / f"{video_path.stem}_overlay.mp4"
    if not args.no_overlay_video:
        video_writer = create_video_writer(overlay_video_path, fps, width, height)

    model, paint_model = load_models(args.model, args.paint_model)
    counts = {
        "frames_seen": 0,
        "frames_processed": 0,
        "ball_detected": 0,
        "possession_found": 0,
        "court_xy_found": 0,
        "paint_homography_available": 0,
    }
    results: list[dict] = []
    state = TemporalPossessionState()

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            frame_index = counts["frames_seen"]
            counts["frames_seen"] += 1
            overlay_frame = frame.copy()

            if args.frame_step > 1 and frame_index % args.frame_step != 0:
                if video_writer is not None:
                    frame_label = f"frame {frame_index:06d}"
                    cv2.putText(
                        overlay_frame,
                        frame_label,
                        (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    video_writer.write(overlay_frame)
                continue
            if args.max_frames > 0 and counts["frames_processed"] >= args.max_frames:
                break

            image_ref = f"{video_path.name}:frame_{frame_index:06d}"
            result = process_frame(
                image_path=image_ref,
                image=frame,
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
            result["frame_index"] = frame_index
            results.append(result)
            (json_dir / f"frame_{frame_index:06d}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

            counts["frames_processed"] += 1
            counts["ball_detected"] += int(bool(result["detections"]["ball_detected"]))
            counts["possession_found"] += int(smoothed_possession["player_bbox_xyxy"] is not None)
            counts["court_xy_found"] += int(smoothed_possession["player_foot_court_xy"] is not None)
            counts["paint_homography_available"] += int(bool(result["paint_homography"]["available"]))

            if video_writer is not None:
                overlay_frame = render_overlay(
                    frame,
                    smoothed_possession["player_bbox_xyxy"],
                    result["detections"]["ball_xyxy"],
                    result["player_detections"],
                    handler_court_xy=smoothed_possession["player_foot_court_xy"],
                    paint_quad=result["paint_homography"]["image_points"],
                    possession_label="Ball Handler"
                    if smoothed_possession["source"] != "smoothed"
                    else "Ball Handler (smoothed)",
                )
                frame_label = f"frame {frame_index:06d}"
                cv2.putText(
                    overlay_frame,
                    frame_label,
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                video_writer.write(overlay_frame)
    finally:
        capture.release()
        if video_writer is not None:
            video_writer.release()

    summary = {
        "video": str(video_path),
        "output": str(output_root.resolve()),
        "counts": counts,
        "frames": results,
    }
    (output_root / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary["counts"], indent=2))
    print(f"Wrote per-frame JSON to: {json_dir.resolve()}")
    print(f"Wrote summary to: {(output_root / 'results.json').resolve()}")
    if video_writer is not None:
        print(f"Wrote overlay video to: {overlay_video_path.resolve()}")


if __name__ == "__main__":
    main()
