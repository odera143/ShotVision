export interface RunInferenceOptions {
  saveOverlays: boolean;
  basketSide: 'LEFT' | 'RIGHT';
  device: 'CPU' | 'GPU';
  frameStep: number;
  holdFrames: number;
}
