from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import cv2
import numpy as np

from inference.paint_homography import PAINT_COURT_POINTS


UNCERTAIN_REASONS = {"no_ball", "ball_in_air"}


@dataclass
class TemporalPossessionState:
    last_handler_bbox_xyxy: list[float] | None = None
    last_handler_foot_xy: list[float] | None = None
    last_handler_court_xy: list[float] | None = None
    last_confidence: float = 0.0
    frames_since_observed: int = 0
    active: bool = False


def _player_foot_xy(player_detection: dict[str, Any]) -> tuple[float, float]:
    foot_xy = player_detection.get("foot_xy")
    if foot_xy is not None:
        return float(foot_xy[0]), float(foot_xy[1])
    xyxy = player_detection["xyxy"]
    return (float(xyxy[0] + xyxy[2]) / 2.0, float(xyxy[3]))


def _player_bbox_xyxy(player_detection: dict[str, Any]) -> list[float]:
    return [float(v) for v in player_detection["xyxy"]]


def _map_foot_to_court(foot_xy: tuple[float, float], paint_homography: dict[str, Any]) -> list[float] | None:
    image_points = paint_homography.get("image_points")
    if not paint_homography.get("available") or not image_points:
        return None

    src = np.asarray(image_points, dtype=np.float32)
    if src.shape != (4, 2):
        return None
    homography, _ = cv2.findHomography(src, PAINT_COURT_POINTS, cv2.RANSAC, 3.0)
    if homography is None:
        return None

    point = np.array([[[float(foot_xy[0]), float(foot_xy[1])]]], dtype=np.float32)
    mapped = cv2.perspectiveTransform(point, homography)[0, 0]
    return [float(mapped[0]), float(mapped[1])]


def _match_nearest_player(
    previous_foot_xy: list[float],
    player_detections: list[dict[str, Any]],
    max_match_distance_px: float,
) -> dict[str, Any] | None:
    if not player_detections:
        return None

    best_player = None
    best_distance = float("inf")
    prev_x, prev_y = float(previous_foot_xy[0]), float(previous_foot_xy[1])
    for player in player_detections:
        foot_x, foot_y = _player_foot_xy(player)
        distance = math.hypot(foot_x - prev_x, foot_y - prev_y)
        if distance < best_distance:
            best_distance = distance
            best_player = player

    if best_player is None or best_distance > max_match_distance_px:
        return None
    return best_player


def smooth_possession(
    result: dict[str, Any],
    state: TemporalPossessionState,
    hold_frames: int = 8,
    max_match_distance_px: float = 140.0,
    confidence_decay: float = 0.1,
) -> tuple[dict[str, Any], TemporalPossessionState]:
    possession = result["possession"]
    player_detections = result["player_detections"]
    reason = possession.get("reason")
    observed = reason == "ok" and possession.get("player_bbox_xyxy") is not None

    if observed:
        smoothed = {
            "player_bbox_xyxy": possession["player_bbox_xyxy"],
            "player_foot_xy": possession["player_foot_xy"],
            "player_foot_court_xy": possession["player_foot_court_xy"],
            "confidence": possession["confidence"],
            "source": "observed",
            "frames_since_observed": 0,
        }
        state.last_handler_bbox_xyxy = possession["player_bbox_xyxy"]
        state.last_handler_foot_xy = possession["player_foot_xy"]
        state.last_handler_court_xy = possession["player_foot_court_xy"]
        state.last_confidence = float(possession["confidence"] or 0.0)
        state.frames_since_observed = 0
        state.active = True
        return smoothed, state

    if (
        reason in UNCERTAIN_REASONS
        and state.active
        and state.last_handler_foot_xy is not None
        and state.frames_since_observed < hold_frames
    ):
        matched_player = _match_nearest_player(
            previous_foot_xy=state.last_handler_foot_xy,
            player_detections=player_detections,
            max_match_distance_px=max_match_distance_px,
        )
        if matched_player is not None:
            state.frames_since_observed += 1
            state.last_handler_bbox_xyxy = _player_bbox_xyxy(matched_player)
            foot_xy = _player_foot_xy(matched_player)
            state.last_handler_foot_xy = [float(foot_xy[0]), float(foot_xy[1])]
            state.last_handler_court_xy = _map_foot_to_court(foot_xy, result["paint_homography"])
            state.last_confidence = max(state.last_confidence - confidence_decay, 0.0)
            smoothed = {
                "player_bbox_xyxy": state.last_handler_bbox_xyxy,
                "player_foot_xy": state.last_handler_foot_xy,
                "player_foot_court_xy": state.last_handler_court_xy,
                "confidence": state.last_confidence,
                "source": "smoothed",
                "frames_since_observed": state.frames_since_observed,
            }
            return smoothed, state

    state.active = False
    state.last_handler_bbox_xyxy = None
    state.last_handler_foot_xy = None
    state.last_handler_court_xy = None
    state.last_confidence = 0.0
    state.frames_since_observed = 0
    smoothed = {
        "player_bbox_xyxy": None,
        "player_foot_xy": None,
        "player_foot_court_xy": None,
        "confidence": 0.0,
        "source": None,
        "frames_since_observed": 0,
    }
    return smoothed, state
