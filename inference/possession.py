from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
import math
from .detect import Det

@dataclass
class PossessionResult:
    player_index: Optional[int]        # index into players list, or None
    confidence: float                  # 0..1
    reason: str                        # "ok" | "no_ball" | "no_players" | "ball_in_air"

def _center(xyxy: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = xyxy
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0

def _top_half_center(xyxy: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = xyxy
    cx = (x1 + x2) / 2.0
    cy = (y1 + (y1 + y2) / 2.0) / 2.0  # center of top half
    return cx, cy

def _height(xyxy: Tuple[float, float, float, float]) -> float:
    return max(1.0, xyxy[3] - xyxy[1])

def infer_possession(players: List[Det], ball: Optional[Det]) -> PossessionResult:
    if ball is None:
        return PossessionResult(player_index=None, confidence=0.0, reason="no_ball")
    if not players:
        return PossessionResult(player_index=None, confidence=0.0, reason="no_players")

    bx, by = _center(ball.xyxy)

    best_i = None
    best_d = float("inf")
    best_thresh = None

    for i, p in enumerate(players):
        px, py = _top_half_center(p.xyxy)
        d = math.hypot(bx - px, by - py)

        # Threshold scales with player height (broadcast zoom changes)
        thresh = 0.35 * _height(p.xyxy)

        if d < best_d:
            best_d = d
            best_i = i
            best_thresh = thresh

    assert best_thresh is not None
    if best_d > best_thresh:
        # Ball likely in air / pass / shot
        return PossessionResult(player_index=None, confidence=0.0, reason="ball_in_air")

    # Convert distance to a 0..1 confidence (simple heuristic)
    conf = max(0.0, min(1.0, 1.0 - (best_d / best_thresh)))
    return PossessionResult(player_index=best_i, confidence=conf, reason="ok")