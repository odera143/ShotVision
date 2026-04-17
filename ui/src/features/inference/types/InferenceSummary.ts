export interface FullInferenceSummary {
  counts: {
    frames_seen: number;
    frames_processed: number;
    ball_detected: number;
    possession_found: number;
    court_xy_found: number;
    paint_homography_available: number;
  };
  frames: {
    image: string;
    detections: {
      num_players: number;
      ball_detected: boolean;
      ball_conf: number;
      ball_xyxy: number[];
    };
    possession: {
      player_index: number;
      confidence: number;
      reason: string | null;
      player_bbox_xyxy: number[];
    };
    paint_homography: {
      available: boolean;
      reason: string | null;
      basket_side: 'left' | 'right';
      image_points: number[][];
      player_foot_court_xy: number[];
    };
    player_detections: {
      cls_id: number;
      conf: number;
      xyxy: number[];
      foot_xy: number[];
    }[];
    overlay_image: string | null;
    smoothed_possession: {
      player_bbox_xyxy: number[];
      player_foot_xy: number[];
      player_foot_court_xy: number[];
      confidence: number;
      source: 'observed' | 'smoothed';
      frames_since_observed: number;
    };
    frame_index: number;
  }[];
}

export interface PossessionOnlyInferenceSummary {
  summary_type: 'POSSESSION_ONLY';
  video: string;
  counts: {
    frames_seen: number;
    frames_processed: number;
    ball_detected: number;
    possession_found: number;
    court_xy_found: number;
    paint_homography_available: number;
  };
  frames: {
    frame_index: number;
    image: string;
    possession: {
      reason: string | null;
      player_bbox_xyxy: number[] | null;
      player_foot_xy: number[] | null;
      player_foot_court_xy: number[] | null;
      confidence: number;
      source: 'observed' | 'smoothed' | null;
      frames_since_observed: number;
    };
  }[];
}

export type InferenceSummary =
  | FullInferenceSummary
  | PossessionOnlyInferenceSummary;
