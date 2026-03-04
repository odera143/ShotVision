from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
from ultralytics import YOLO

@dataclass
class Det:
    cls_id: int            # 0=player, 1=ball
    conf: float
    xyxy: Tuple[float, float, float, float]  # x1,y1,x2,y2

def _to_det(result) -> List[Det]:
    dets: List[Det] = []
    if result.boxes is None or len(result.boxes) == 0:
        return dets
    for b in result.boxes:
        cls_id = int(b.cls[0])
        conf = float(b.conf[0])
        x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())
        dets.append(Det(cls_id=cls_id, conf=conf, xyxy=(x1, y1, x2, y2)))
    return dets

def detect_players_ball(
    model: YOLO,
    image_path: str,
    imgsz: int = 1280,
    conf: float = 0.15,
    iou: float = 0.45,
) -> Tuple[List[Det], Optional[Det]]:
    """
    Returns (players, ball). Ball is the highest-confidence ball detection (if any).
    """
    results = model.predict(source=image_path, imgsz=imgsz, conf=conf, iou=iou, verbose=False)
    dets = _to_det(results[0])

    players = [d for d in dets if d.cls_id == 0]
    balls = [d for d in dets if d.cls_id == 1]
    ball = max(balls, key=lambda d: d.conf, default=None)

    return players, ball