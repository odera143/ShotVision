# UI

React + Vite frontend for the Shot Vision API.

The UI is currently focused on:

- uploading a video to the FastAPI backend
- selecting inference options
- polling job status
- fetching inference results
- downloading the overlay video as a debug artifact when requested

## Current Flow

1. Select an `.mp4` file.
2. Choose inference options:
   - `Save overlays`
   - `Basket Side`
   - `Device`
   - `Frame Step`
   - `Results Mode`
   - `Hold Frames`
3. Submit the job to `POST /jobs`.
4. Poll `GET /jobs/{job_id}` until the job completes or fails.
5. Fetch `GET /jobs/{job_id}/results`.
6. If overlays were requested, download `GET /jobs/{job_id}/overlay-video`.

## Dev Setup

From the repo root:

```powershell
cd ui
npm install
npm run dev
```

The Vite dev server runs on `http://localhost:5174` in the current setup.

## Backend Requirement

The UI expects the FastAPI service to be running locally on `http://localhost:8080`.

Example:

```powershell
uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload
```

## Result Modes

The frontend can request one of two result shapes:

- `POSSESSION_ONLY`
  Best for the current UI. Returns mostly the ball-handler possession/court-position data needed for rendering.
- `FULL`
  Returns the full backend summary payload for debugging and deeper inspection.

## Notes

- The backend stores jobs temporarily and deletes them after a TTL (during another run).
- The current UI is simple and is meant to support iteration on the inference pipeline rather than be a finished product.
