from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference.paint_homography import draw_quad, fit_paint_homography


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
PARTIAL_PAINT_REASON_SNIPPET = "too close to the image border"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLO paint segmentation and fit a homography from the predicted mask."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("runs/segment/train/weights/best.pt"),
        help="Path to the trained YOLO segmentation model.",
    )
    parser.add_argument("--source", type=Path, required=True, help="Image file or folder of images.")
    parser.add_argument("--basket-side", choices=["left", "right"], required=True)
    parser.add_argument("--output", type=Path, default=Path("runs/paint-seg/predict"))
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", type=str, default="")
    return parser.parse_args()


def collect_images(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    return sorted([p for p in source.iterdir() if p.suffix.lower() in IMAGE_EXTS])


def mask_from_result(result, image_shape: tuple[int, int]) -> np.ndarray:
    image_height, image_width = image_shape
    if result.masks is None or result.masks.data is None or len(result.masks.data) == 0:
        raise ValueError("No paint mask predicted.")

    masks = result.masks.data.cpu().numpy()
    if masks.ndim != 3:
        raise ValueError(f"Unexpected mask tensor shape: {masks.shape}")

    areas = masks.sum(axis=(1, 2))
    best_index = int(np.argmax(areas))
    mask = (masks[best_index] > 0.5).astype(np.uint8) * 255
    if mask.shape != (image_height, image_width):
        mask = cv2.resize(mask, (image_width, image_height), interpolation=cv2.INTER_NEAREST)
    return mask


def make_overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = image.copy()
    tint = np.array([0, 255, 255], dtype=np.uint8)
    active = mask > 0
    overlay[active] = (0.6 * overlay[active] + 0.4 * tint).astype(np.uint8)
    return overlay


def write_homography_json(output_path: Path, basket_side: str, fit) -> None:
    payload = {
        "basket_side": basket_side,
        "image_points": fit.image_points.tolist(),
        "court_points": fit.court_points.tolist(),
        "homography": fit.homography.tolist(),
        "inverse_homography": fit.inverse_homography.tolist(),
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_partial_paint_review(
    review_root: Path,
    image_path: Path,
    image: np.ndarray,
    mask: np.ndarray,
) -> None:
    images_dir = review_root / "images"
    masks_dir = review_root / "masks"
    overlays_dir = review_root / "overlays"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(images_dir / image_path.name), image)
    cv2.imwrite(str(masks_dir / f"{image_path.stem}.png"), mask)
    cv2.imwrite(str(overlays_dir / f"{image_path.stem}.jpg"), make_overlay(image, mask))


def main() -> None:
    args = parse_args()
    image_paths = collect_images(args.source)
    if not image_paths:
        raise RuntimeError(f"No images found in {args.source}")

    mask_dir = args.output / "masks"
    overlay_dir = args.output / "overlays"
    homography_dir = args.output / "homography"
    quad_dir = args.output / "quad_overlays"
    partial_review_dir = args.output / "partial_paint"
    failures_path = args.output / "failures.json"
    mask_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    homography_dir.mkdir(parents=True, exist_ok=True)
    quad_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    failures: list[dict[str, str]] = []
    homography_count = 0
    partial_paint_count = 0

    for image_path in image_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            failures.append({"image": str(image_path), "reason": "failed_to_read_image"})
            continue

        results = model.predict(
            source=str(image_path),
            imgsz=args.imgsz,
            conf=args.conf,
            device=args.device or None,
            verbose=False,
        )
        if not results:
            failures.append({"image": str(image_path), "reason": "no_result_returned"})
            continue

        mask: np.ndarray | None = None
        try:
            mask = mask_from_result(results[0], image.shape[:2])
            fit = fit_paint_homography(mask, basket_side=args.basket_side)
        except ValueError as exc:
            reason = str(exc)
            failure_record = {"image": str(image_path), "reason": reason}
            if mask is not None and PARTIAL_PAINT_REASON_SNIPPET in reason:
                save_partial_paint_review(partial_review_dir, image_path, image, mask)
                failure_record["category"] = "partial_paint"
                partial_paint_count += 1
            failures.append(failure_record)
            continue

        cv2.imwrite(str(mask_dir / f"{image_path.stem}.png"), mask)
        cv2.imwrite(str(overlay_dir / f"{image_path.stem}.jpg"), make_overlay(image, mask))
        cv2.imwrite(str(quad_dir / f"{image_path.stem}.jpg"), draw_quad(image, fit.image_points))
        write_homography_json(homography_dir / f"{image_path.stem}.json", args.basket_side, fit)
        homography_count += 1

    failures_path.write_text(json.dumps(failures, indent=2), encoding="utf-8")
    print(f"Wrote masks to: {mask_dir.resolve()}")
    print(f"Wrote overlays to: {overlay_dir.resolve()}")
    print(f"Wrote homographies to: {homography_dir.resolve()}")
    print(f"Wrote quad overlays to: {quad_dir.resolve()}")
    print(f"Homography fits: {homography_count}/{len(image_paths)}")
    print(f"Partial paint review cases: {partial_paint_count}")
    if partial_paint_count > 0:
        print(f"Wrote partial paint review set to: {partial_review_dir.resolve()}")
    print(f"Wrote failures to: {failures_path.resolve()}")


if __name__ == "__main__":
    main()
