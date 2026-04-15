export interface InferenceJob {
  job_id: string;
  status: 'failed' | 'queued' | 'running' | 'completed';
  input_video_name: string;
  save_overlays: boolean;
  basket_side: 'LEFT' | 'RIGHT';
  device: 'GPU' | 'CPU';
  frame_step: number;
  hold_frames: number;
  status_url: string;
  results_url: string;
  overlay_video_url: string | null;
}
