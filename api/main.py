from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from inference.run_video import main as run_video_main

ROOT = Path(__file__).resolve().parents[1]
JOBS_DIR = ROOT / "jobs"
JOB_TTL_SECONDS = int(os.getenv("JOB_TTL_SECONDS", "1800"))
RESULTS_MODES = {"FULL", "POSSESSION_ONLY"}

PLAYER_MODEL_PATH = Path(
    os.getenv("PLAYER_MODEL_PATH", ROOT / "runs" / "detect" / "players_ball_2" / "weights" / "best.pt")
)
PAINT_MODEL_PATH = Path(
    os.getenv("PAINT_MODEL_PATH", ROOT / "runs" / "segment" / "train" / "weights" / "best.pt")
)

app = FastAPI(title="Shot Vision API")
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in ALLOWED_ORIGINS if origin.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _normalize_device(device: str) -> str:
    return "0" if device.upper() == "GPU" else "cpu"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_json(path: Path) -> dict:
    if not path.exists():
        raise HTTPException(status_code=404, detail="Job not found.")
    return json.loads(path.read_text(encoding="utf-8"))


def _job_paths(job_id: str) -> dict[str, Path]:
    root = JOBS_DIR / job_id
    uploads = root / "uploads"
    outputs = root / "outputs"
    return {
        "root": root,
        "uploads": uploads,
        "outputs": outputs,
        "status": root / "status.json",
        "results": outputs / "results.json",
    }


def _normalize_results_mode(mode: str) -> str:
    normalized = mode.upper()
    if normalized not in RESULTS_MODES:
        raise HTTPException(
            status_code=400,
            detail="results_mode must be FULL or POSSESSION_ONLY.",
        )
    return normalized


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _isoformat_utc(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _with_expiry(payload: dict, ttl_seconds: int = JOB_TTL_SECONDS) -> dict:
    completed_at = _now_utc()
    expires_at = completed_at + timedelta(seconds=ttl_seconds)
    return {
        **payload,
        "completed_at": _isoformat_utc(completed_at),
        "expires_at": _isoformat_utc(expires_at),
    }


def _cleanup_expired_jobs() -> None:
    if not JOBS_DIR.exists():
        return

    now = _now_utc()
    for job_dir in JOBS_DIR.iterdir():
        if not job_dir.is_dir():
            continue
        status_path = job_dir / "status.json"
        if not status_path.exists():
            shutil.rmtree(job_dir, ignore_errors=True)
            continue

        try:
            payload = json.loads(status_path.read_text(encoding="utf-8"))
        except Exception:
            shutil.rmtree(job_dir, ignore_errors=True)
            continue

        status = payload.get("status")
        expires_at = payload.get("expires_at")
        if status in {"completed", "failed"} and expires_at:
            try:
                if _parse_utc(expires_at) <= now:
                    shutil.rmtree(job_dir, ignore_errors=True)
            except ValueError:
                shutil.rmtree(job_dir, ignore_errors=True)


def _read_job_status(job_id: str) -> dict:
    _cleanup_expired_jobs()
    return _read_json(_job_paths(job_id)["status"])


def _public_job_response(job_id: str, status_payload: dict) -> dict:
    response = {
        "job_id": job_id,
        "status": status_payload["status"],
        "input_video_name": status_payload.get("input_video_name"),
        "save_overlays": status_payload.get("save_overlays"),
        "basket_side": status_payload.get("basket_side"),
        "device": status_payload.get("device"),
        "frame_step": status_payload.get("frame_step"),
        "hold_frames": status_payload.get("hold_frames"),
        "results_mode": status_payload.get("results_mode"),
        "status_url": f"/jobs/{job_id}",
        "results_url": f"/jobs/{job_id}/results",
        "expires_at": status_payload.get("expires_at"),
    }
    if status_payload.get("save_overlays"):
        response["overlay_video_url"] = f"/jobs/{job_id}/overlay-video"
    if status_payload.get("status") == "failed":
        response["error"] = status_payload.get("error")
    return response


def _project_results(summary: dict, results_mode: str) -> dict:
    if results_mode == "FULL":
        return summary

    frames = []
    for frame in summary.get("frames", []):
        smoothed = frame.get("smoothed_possession") or {}
        raw_possession = frame.get("possession") or {}
        frames.append(
            {
                "frame_index": frame.get("frame_index"),
                "image": frame.get("image"),
                "possession": {
                    "reason": raw_possession.get("reason"),
                    "player_bbox_xyxy": smoothed.get("player_bbox_xyxy"),
                    "player_foot_xy": smoothed.get("player_foot_xy"),
                    "player_foot_court_xy": smoothed.get("player_foot_court_xy"),
                    "confidence": smoothed.get("confidence"),
                    "source": smoothed.get("source"),
                    "frames_since_observed": smoothed.get("frames_since_observed"),
                },
            }
        )

    return {
        "summary_type": "POSSESSION_ONLY",
        "video": summary.get("video"),
        "counts": summary.get("counts"),
        "frames": frames,
    }


def _build_run_args(
    *,
    video_path: Path,
    output_dir: Path,
    save_overlays: bool,
    basket_side: str,
    device: str,
    frame_step: int,
    hold_frames: int,
) -> SimpleNamespace:
    return SimpleNamespace(
        video=str(video_path),
        model=str(PLAYER_MODEL_PATH),
        paint_model=str(PAINT_MODEL_PATH),
        paint_basket_side=basket_side.lower(),
        paint_imgsz=1024,
        paint_conf=0.25,
        device=_normalize_device(device),
        imgsz=1280,
        conf=0.15,
        iou=0.45,
        output=str(output_dir),
        frame_step=frame_step,
        max_frames=0,
        no_overlay_video=not save_overlays,
        hold_frames=hold_frames,
        max_match_distance=140.0,
    )


def _run_job(job_id: str, run_args: SimpleNamespace, request_summary: dict) -> None:
    paths = _job_paths(job_id)
    _write_json(
        paths["status"],
        {
            **request_summary,
            "job_id": job_id,
            "status": "running",
        },
    )

    try:
        if not PLAYER_MODEL_PATH.exists():
            raise FileNotFoundError(f"Player model not found: {PLAYER_MODEL_PATH}")
        if not PAINT_MODEL_PATH.exists():
            raise FileNotFoundError(f"Paint model not found: {PAINT_MODEL_PATH}")

        run_video_main(run_args)

        status_payload = {
            **request_summary,
            "job_id": job_id,
            "status": "completed",
            "results_path": str(paths["results"]),
        }
        if not run_args.no_overlay_video:
            overlay_path = paths["outputs"] / f"{Path(run_args.video).stem}_overlay.mp4"
            status_payload["overlay_video_path"] = str(overlay_path)
        _write_json(paths["status"], _with_expiry(status_payload))
    except Exception as exc:
        _write_json(
            paths["status"],
            _with_expiry(
                {
                    **request_summary,
                    "job_id": job_id,
                    "status": "failed",
                    "error": str(exc),
                }
            ),
        )


@app.get("/")
def read_root() -> dict[str, str]:
    _cleanup_expired_jobs()
    return {"status": "ok", "docs": "/docs"}


@app.post("/jobs")
async def create_job(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(..., description="MP4 video to process"),
    save_overlays: bool = Form(True),
    basket_side: str = Form(..., description="LEFT or RIGHT"),
    device: str = Form("GPU", description="CPU or GPU"),
    frame_step: int = Form(1),
    hold_frames: int = Form(8),
    results_mode: str = Form("POSSESSION_ONLY", description="FULL or POSSESSION_ONLY"),
) -> dict:
    _cleanup_expired_jobs()
    if video.content_type not in {"video/mp4", "application/octet-stream"}:
        raise HTTPException(status_code=400, detail="Only MP4 uploads are supported right now.")

    basket_side = basket_side.upper()
    device = device.upper()
    results_mode = _normalize_results_mode(results_mode)
    if basket_side not in {"LEFT", "RIGHT"}:
        raise HTTPException(status_code=400, detail="basket_side must be LEFT or RIGHT.")
    if device not in {"CPU", "GPU"}:
        raise HTTPException(status_code=400, detail="device must be CPU or GPU.")
    if frame_step < 1:
        raise HTTPException(status_code=400, detail="frame_step must be at least 1.")
    if hold_frames < 0:
        raise HTTPException(status_code=400, detail="hold_frames must be at least 0.")

    job_id = uuid4().hex
    paths = _job_paths(job_id)
    paths["uploads"].mkdir(parents=True, exist_ok=True)
    paths["outputs"].mkdir(parents=True, exist_ok=True)

    suffix = Path(video.filename or "input.mp4").suffix or ".mp4"
    upload_path = paths["uploads"] / f"input{suffix}"
    with upload_path.open("wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    request_summary = {
        "input_video_name": video.filename or upload_path.name,
        "input_video_path": str(upload_path),
        "save_overlays": save_overlays,
        "basket_side": basket_side,
        "device": device,
        "frame_step": frame_step,
        "hold_frames": hold_frames,
        "results_mode": results_mode,
        "output_dir": str(paths["outputs"]),
    }

    _write_json(
        paths["status"],
        {
            **request_summary,
            "job_id": job_id,
            "status": "queued",
        },
    )

    run_args = _build_run_args(
        video_path=upload_path,
        output_dir=paths["outputs"],
        save_overlays=save_overlays,
        basket_side=basket_side,
        device=device,
        frame_step=frame_step,
        hold_frames=hold_frames,
    )
    background_tasks.add_task(_run_job, job_id, run_args, request_summary)

    return {
        "job_id": job_id,
        "status": "queued",
        "results_mode": results_mode,
        "status_url": f"/jobs/{job_id}",
        "results_url": f"/jobs/{job_id}/results",
        "overlay_video_url": f"/jobs/{job_id}/overlay-video" if save_overlays else None,
    }


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    status_payload = _read_job_status(job_id)
    return _public_job_response(job_id, status_payload)


@app.get("/jobs/{job_id}/results")
def get_job_results(job_id: str) -> dict:
    paths = _job_paths(job_id)
    status = _read_job_status(job_id)
    if status.get("status") != "completed":
        raise HTTPException(status_code=409, detail="Job has not completed yet.")
    results_mode = _normalize_results_mode(status.get("results_mode", "FULL"))
    summary = _read_json(paths["results"])
    return _project_results(summary, results_mode)


@app.get("/jobs/{job_id}/overlay-video")
def get_overlay_video(job_id: str) -> FileResponse:
    paths = _job_paths(job_id)
    status = _read_job_status(job_id)
    if status.get("status") != "completed":
        raise HTTPException(status_code=409, detail="Job has not completed yet.")
    if not status.get("save_overlays"):
        raise HTTPException(status_code=404, detail="Overlay video was not requested for this job.")

    overlay_path = paths["outputs"] / "input_overlay.mp4"
    if not overlay_path.exists():
        raise HTTPException(status_code=404, detail="Overlay video not found.")
    return FileResponse(path=overlay_path, media_type="video/mp4", filename=overlay_path.name)
