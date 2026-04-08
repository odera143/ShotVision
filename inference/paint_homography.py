from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


HOOP_TO_BASELINE_FT = 5.25
LANE_HALF_WIDTH_FT = 8.0
LANE_DEPTH_FT = 19.0
EDGE_MARGIN_PX = 4

# Ordered to match the first four keypoints in the existing dataset:
# lane_bl_l, lane_bl_r, lane_ft_l, lane_ft_r
PAINT_COURT_POINTS = np.array(
    [
        [-LANE_HALF_WIDTH_FT, -HOOP_TO_BASELINE_FT],
        [LANE_HALF_WIDTH_FT, -HOOP_TO_BASELINE_FT],
        [-LANE_HALF_WIDTH_FT, LANE_DEPTH_FT - HOOP_TO_BASELINE_FT],
        [LANE_HALF_WIDTH_FT, LANE_DEPTH_FT - HOOP_TO_BASELINE_FT],
    ],
    dtype=np.float32,
)


@dataclass(frozen=True)
class PaintHomography:
    homography: np.ndarray
    inverse_homography: np.ndarray
    image_points: np.ndarray
    court_points: np.ndarray

    def img_to_court(self, x: float, y: float) -> tuple[float, float]:
        pt = np.array([[[x, y]]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(pt, self.homography)[0, 0]
        return float(mapped[0]), float(mapped[1])

    def court_to_img(self, x: float, y: float) -> tuple[float, float]:
        pt = np.array([[[x, y]]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(pt, self.inverse_homography)[0, 0]
        return float(mapped[0]), float(mapped[1])


def find_paint_quad(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 127).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No paint contour found in mask.")

    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 100.0:
        raise ValueError("Paint contour is too small to fit a homography.")

    image_height, image_width = mask.shape[:2]
    x, y, width, height = cv2.boundingRect(contour)
    if (
        x <= EDGE_MARGIN_PX
        or y <= EDGE_MARGIN_PX
        or (x + width) >= (image_width - EDGE_MARGIN_PX)
        or (y + height) >= (image_height - EDGE_MARGIN_PX)
    ):
        raise ValueError("Paint contour is too close to the image border, so one or more corners are likely out of frame.")

    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
    if len(approx) == 4:
        return approx[:, 0, :].astype(np.float32)

    rect = cv2.minAreaRect(contour)
    return cv2.boxPoints(rect).astype(np.float32)


def order_paint_corners(quad: np.ndarray, basket_side: str) -> np.ndarray:
    if quad.shape != (4, 2):
        raise ValueError(f"Expected a 4x2 quad, got {quad.shape}")

    sorted_by_x = quad[np.argsort(quad[:, 0])]
    if basket_side == "right":
        ft_pair = sorted_by_x[:2]
        baseline_pair = sorted_by_x[2:]
    elif basket_side == "left":
        baseline_pair = sorted_by_x[:2]
        ft_pair = sorted_by_x[2:]
    else:
        raise ValueError("basket_side must be 'left' or 'right'")

    baseline_pair = baseline_pair[np.argsort(-baseline_pair[:, 1])]
    ft_pair = ft_pair[np.argsort(-ft_pair[:, 1])]
    return np.vstack([baseline_pair, ft_pair]).astype(np.float32)


def fit_paint_homography(mask: np.ndarray, basket_side: str) -> PaintHomography:
    quad = find_paint_quad(mask)
    ordered_img_points = order_paint_corners(quad, basket_side=basket_side)
    homography, _ = cv2.findHomography(ordered_img_points, PAINT_COURT_POINTS, cv2.RANSAC, 3.0)
    if homography is None:
        raise ValueError("Homography fit failed.")
    inverse_homography = np.linalg.inv(homography)
    return PaintHomography(
        homography=homography,
        inverse_homography=inverse_homography,
        image_points=ordered_img_points,
        court_points=PAINT_COURT_POINTS.copy(),
    )


def draw_quad(image: np.ndarray, quad: np.ndarray) -> np.ndarray:
    overlay = image.copy()
    poly = quad.astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(overlay, [poly], True, (0, 255, 255), 3)
    for idx, (x, y) in enumerate(quad):
        cv2.circle(overlay, (int(round(x)), int(round(y))), 6, (0, 0, 255), -1)
        cv2.putText(
            overlay,
            str(idx),
            (int(round(x)) + 10, int(round(y)) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )
    return overlay


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit a hoop-centered homography from a predicted paint mask.")
    parser.add_argument("--mask", type=Path, required=True, help="Binary paint mask image.")
    parser.add_argument("--basket-side", choices=["left", "right"], required=True)
    parser.add_argument("--image", type=Path, default=None, help="Optional original frame for debug overlay.")
    parser.add_argument("--out-image", type=Path, default=Path("runs/paint-homography/debug.jpg"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mask = cv2.imread(str(args.mask), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Failed to read mask: {args.mask}")

    fit = fit_paint_homography(mask, basket_side=args.basket_side)
    print("image_points:", fit.image_points.tolist())
    print("court_points:", fit.court_points.tolist())
    print("homography:")
    print(fit.homography)

    if args.image is not None:
        image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read image: {args.image}")
        args.out_image.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.out_image), draw_quad(image, fit.image_points))
        print(f"Wrote debug overlay to: {args.out_image.resolve()}")


if __name__ == "__main__":
    main()
