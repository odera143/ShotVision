from fastapi import FastAPI, Query
from inference.run_video import main as run_video_main
import json

app = FastAPI()

@app.get("/inference")
async def run_inference(
    video_path: str = Query(..., description="Path to the input video file"),
    save_overlays: bool = Query(False, description="Whether to save overlay images"),
    basket_side: str = Query('LEFT', description="Side of the basket (LEFT or RIGHT)"),
    device: str = Query('CPU', description="Device to run inference on (CPU or GPU)"),
    frame_step: int = Query(1, description="Number of frames to skip between inferences"),
    hold_frames: int = Query(0, description="Number of frames to hold before processing")
):
    apiArgs = {
        "video": video_path,
        "no_overlay_video": not save_overlays,
        "paint_basket_side": basket_side,
        "device": device,
        "frame_step": frame_step,
        "hold_frames": hold_frames
    }
    return json.dumps(apiArgs, indent=2)