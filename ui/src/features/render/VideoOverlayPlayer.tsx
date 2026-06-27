import { useEffect, useMemo, useRef, useState } from 'react';
import { Alert, Form, Stack } from 'react-bootstrap';
import type {
  FullInferenceSummary,
  InferenceSummary,
  PossessionOnlyInferenceSummary,
} from '../inference/types/InferenceSummary';
import PlayerStatsTooltip from './PlayerStatsTooltip';
import { findShotGridCell, formatShotGridPercent } from './shotGridStats';
import type { PlayerShotGrid } from './types/PlayerShotGrid';

type VideoOverlayPlayerProps = {
  videoUrl: string | null;
  results: InferenceSummary | null;
  title?: string;
  defaultFps?: number;
  sharedPlaybackEnabled?: boolean;
  sharedFrameIndex?: number;
  onSharedFrameIndexChange?: (frameIndex: number) => void;
  sharedFps?: number;
  onSharedFpsChange?: (fps: number) => void;
  playerShotGrid: PlayerShotGrid | null;
};

type OverlayFrame = {
  frameIndex: number;
  reason: string | null;
  confidence: number;
  source: string | null;
  playerBbox: [number, number, number, number] | null;
  playerFoot: [number, number] | null;
  playerFootCourt: [number, number] | null;
};

const DEFAULT_FPS = 30;

const isBox = (
  value: number[] | null | undefined,
): value is [number, number, number, number] =>
  Array.isArray(value) && value.length === 4;

const isPoint = (
  value: number[] | null | undefined,
): value is [number, number] => Array.isArray(value) && value.length === 2;

const isPossessionOnlySummary = (
  results: InferenceSummary,
): results is PossessionOnlyInferenceSummary =>
  'summary_type' in results && results.summary_type === 'POSSESSION_ONLY';

const getPossessionOnlyOverlayFrame = (
  results: PossessionOnlyInferenceSummary,
  frameIndex: number,
): OverlayFrame => {
  const frame = results.frames[frameIndex];

  return {
    frameIndex: frame.frame_index,
    reason: frame.possession.reason,
    confidence: frame.possession.confidence,
    source: frame.possession.source,
    playerBbox: isBox(frame.possession.player_bbox_xyxy)
      ? frame.possession.player_bbox_xyxy
      : null,
    playerFoot: isPoint(frame.possession.player_foot_xy)
      ? frame.possession.player_foot_xy
      : null,
    playerFootCourt: isPoint(frame.possession.player_foot_court_xy)
      ? frame.possession.player_foot_court_xy
      : null,
  };
};

const getFullOverlayFrame = (
  results: FullInferenceSummary,
  frameIndex: number,
): OverlayFrame => {
  const frame = results.frames[frameIndex];

  return {
    frameIndex: frame.frame_index,
    reason: frame.possession.reason,
    confidence: frame.smoothed_possession.confidence,
    source: frame.smoothed_possession.source,
    playerBbox: isBox(frame.smoothed_possession.player_bbox_xyxy)
      ? frame.smoothed_possession.player_bbox_xyxy
      : isBox(frame.possession.player_bbox_xyxy)
        ? frame.possession.player_bbox_xyxy
        : null,
    playerFoot: isPoint(frame.smoothed_possession.player_foot_xy)
      ? frame.smoothed_possession.player_foot_xy
      : null,
    playerFootCourt: isPoint(frame.smoothed_possession.player_foot_court_xy)
      ? frame.smoothed_possession.player_foot_court_xy
      : isPoint(frame.paint_homography.player_foot_court_xy)
        ? frame.paint_homography.player_foot_court_xy
        : null,
  };
};

const getOverlayFrame = (
  results: InferenceSummary | null,
  frameIndex: number,
): OverlayFrame | null => {
  if (!results || results.frames.length === 0) {
    return null;
  }

  const clampedIndex = Math.min(
    results.frames.length - 1,
    Math.max(0, Math.trunc(frameIndex)),
  );

  if (isPossessionOnlySummary(results)) {
    return getPossessionOnlyOverlayFrame(results, clampedIndex);
  }

  return getFullOverlayFrame(results, clampedIndex);
};

const VideoOverlayPlayer = ({
  videoUrl,
  results,
  title = 'Clip Preview',
  defaultFps = DEFAULT_FPS,
  sharedPlaybackEnabled = false,
  sharedFrameIndex,
  onSharedFrameIndexChange,
  sharedFps,
  onSharedFpsChange,
  playerShotGrid,
}: VideoOverlayPlayerProps) => {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const [videoSize, setVideoSize] = useState({ width: 1280, height: 720 });
  const [currentTime, setCurrentTime] = useState(0);
  const [localFps, setLocalFps] = useState(defaultFps);
  const [showBoundingBox, setShowBoundingBox] = useState(true);
  const [showFootMarker, setShowFootMarker] = useState(true);
  const [isPlayerTooltipVisible, setIsPlayerTooltipVisible] = useState(false);
  const fps = sharedPlaybackEnabled
    ? Math.max(1, sharedFps ?? defaultFps)
    : localFps;

  const maxFrameIndex = useMemo(
    () => Math.max(0, (results?.frames.length ?? 1) - 1),
    [results],
  );

  const computedFrameIndex = useMemo(() => {
    if (!results) {
      return 0;
    }

    return Math.min(maxFrameIndex, Math.max(0, Math.round(currentTime * fps)));
  }, [currentTime, fps, maxFrameIndex, results]);

  const overlayFrameIndex = sharedPlaybackEnabled
    ? Math.min(
        maxFrameIndex,
        Math.max(0, sharedFrameIndex ?? computedFrameIndex),
      )
    : computedFrameIndex;

  const overlayFrame = useMemo(
    () => getOverlayFrame(results, overlayFrameIndex),
    [overlayFrameIndex, results],
  );

  const playerTooltipPosition = useMemo(() => {
    if (!overlayFrame?.playerBbox) {
      return null;
    }

    const [x1, y1, x2] = overlayFrame.playerBbox;
    const left = ((x1 + x2) / 2 / videoSize.width) * 100;
    const top = (y1 / videoSize.height) * 100;

    return {
      left: `${left}%`,
      top: `${top}%`,
      isNearTop: y1 < videoSize.height * 0.12,
    };
  }, [overlayFrame, videoSize.height, videoSize.width]);

  const playerFootPosition = useMemo(() => {
    return overlayFrame?.playerFootCourt ?? null;
  }, [overlayFrame]);

  const shotGridCell = useMemo(
    () => findShotGridCell(playerFootPosition, playerShotGrid),
    [playerFootPosition, playerShotGrid],
  );

  const stopAnimationLoop = () => {
    if (animationFrameRef.current !== null) {
      window.cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }
  };

  useEffect(() => stopAnimationLoop, []);

  useEffect(() => {
    setCurrentTime(0);
  }, [videoUrl]);

  useEffect(() => {
    setIsPlayerTooltipVisible(false);
  }, [overlayFrameIndex]);

  useEffect(() => {
    if (!sharedPlaybackEnabled || !videoRef.current) {
      return;
    }

    const expectedTime = overlayFrameIndex / Math.max(fps, 1);
    const currentVideoTime = videoRef.current.currentTime;

    if (Math.abs(currentVideoTime - expectedTime) > 0.05) {
      videoRef.current.currentTime = expectedTime;
      setCurrentTime(expectedTime);
    }
  }, [fps, overlayFrameIndex, sharedPlaybackEnabled]);

  const syncCurrentTime = () => {
    if (!videoRef.current) {
      return;
    }

    const nextTime = videoRef.current.currentTime;
    setCurrentTime(nextTime);
    if (sharedPlaybackEnabled) {
      onSharedFrameIndexChange?.(
        Math.min(maxFrameIndex, Math.max(0, Math.round(nextTime * fps))),
      );
    }
    if (!videoRef.current.paused && !videoRef.current.ended) {
      animationFrameRef.current = window.requestAnimationFrame(syncCurrentTime);
    } else {
      animationFrameRef.current = null;
    }
  };

  const startAnimationLoop = () => {
    stopAnimationLoop();
    animationFrameRef.current = window.requestAnimationFrame(syncCurrentTime);
  };

  const handleLoadedMetadata = () => {
    if (!videoRef.current) {
      return;
    }

    setVideoSize({
      width: videoRef.current.videoWidth || 1280,
      height: videoRef.current.videoHeight || 720,
    });
  };

  const handleFrameScrub = (nextFrameIndex: number) => {
    if (!videoRef.current) {
      return;
    }

    const clampedFrameIndex = Math.min(
      maxFrameIndex,
      Math.max(0, Math.trunc(nextFrameIndex)),
    );
    const nextTime = clampedFrameIndex / Math.max(fps, 1);
    videoRef.current.currentTime = nextTime;
    setCurrentTime(nextTime);
    if (sharedPlaybackEnabled) {
      onSharedFrameIndexChange?.(clampedFrameIndex);
    }
  };

  return (
    <Stack gap={3}>
      <div>
        <h2 className='h4 mb-1'>{title}</h2>
        <p className='text-body-secondary mb-0'>
          Render inference results as an overlay on the input video.
        </p>
      </div>

      {!videoUrl ? (
        <Alert variant='secondary' className='mb-0'>
          Select a video file to preview it here.
        </Alert>
      ) : (
        <>
          <div className='video-overlay-shell'>
            <video
              ref={videoRef}
              className='video-overlay-player'
              src={videoUrl}
              controls
              playsInline
              onLoadedMetadata={handleLoadedMetadata}
              onPlay={() => {
                startAnimationLoop();
              }}
              onPause={() => {
                stopAnimationLoop();
              }}
              onEnded={() => {
                stopAnimationLoop();
              }}
              onSeeked={() => {
                if (videoRef.current) {
                  const nextTime = videoRef.current.currentTime;
                  setCurrentTime(nextTime);
                  if (sharedPlaybackEnabled) {
                    onSharedFrameIndexChange?.(
                      Math.min(
                        maxFrameIndex,
                        Math.max(0, Math.round(nextTime * fps)),
                      ),
                    );
                  }
                }
              }}
              onTimeUpdate={() => {
                if (videoRef.current) {
                  const nextTime = videoRef.current.currentTime;
                  setCurrentTime(nextTime);
                  if (sharedPlaybackEnabled) {
                    onSharedFrameIndexChange?.(
                      Math.min(
                        maxFrameIndex,
                        Math.max(0, Math.round(nextTime * fps)),
                      ),
                    );
                  }
                }
              }}
              disablePictureInPicture
            />
            <svg
              className='video-overlay-layer'
              viewBox={`0 0 ${videoSize.width} ${videoSize.height}`}
              aria-hidden='true'
            >
              {overlayFrame?.playerBbox && shotGridCell && (
                <text
                  x={overlayFrame.playerBbox[0]}
                  y={overlayFrame.playerBbox[1] - 5}
                  className='video-overlay-label'
                >
                  {formatShotGridPercent(shotGridCell.fg)}
                </text>
              )}
              {overlayFrame?.playerBbox && (
                <rect
                  x={overlayFrame.playerBbox[0]}
                  y={overlayFrame.playerBbox[1]}
                  width={Math.max(
                    0,
                    overlayFrame.playerBbox[2] - overlayFrame.playerBbox[0],
                  )}
                  height={Math.max(
                    0,
                    overlayFrame.playerBbox[3] - overlayFrame.playerBbox[1],
                  )}
                  className='video-overlay-hitbox'
                  onMouseEnter={() => {
                    setIsPlayerTooltipVisible(true);
                  }}
                  onMouseLeave={() => {
                    setIsPlayerTooltipVisible(false);
                  }}
                />
              )}
              {showBoundingBox && overlayFrame?.playerBbox && (
                <rect
                  x={overlayFrame.playerBbox[0]}
                  y={overlayFrame.playerBbox[1]}
                  width={Math.max(
                    0,
                    overlayFrame.playerBbox[2] - overlayFrame.playerBbox[0],
                  )}
                  height={Math.max(
                    0,
                    overlayFrame.playerBbox[3] - overlayFrame.playerBbox[1],
                  )}
                  className='video-overlay-bbox'
                />
              )}
              {showFootMarker && overlayFrame?.playerFoot && (
                <circle
                  cx={overlayFrame.playerFoot[0]}
                  cy={overlayFrame.playerFoot[1]}
                  r='9'
                  className='video-overlay-foot'
                />
              )}
            </svg>
            {isPlayerTooltipVisible &&
              overlayFrame &&
              playerTooltipPosition && (
                <PlayerStatsTooltip
                  playerTooltipPosition={playerTooltipPosition}
                  playerFootPosition={playerFootPosition}
                  playerShotGrid={playerShotGrid}
                />
              )}
          </div>

          <Stack
            direction='horizontal'
            gap={3}
            className='flex-wrap align-items-end'
          >
            <Form.Group>
              <Form.Label className='fw-semibold mb-1'>Overlay FPS</Form.Label>
              <Form.Control
                type='number'
                min={1}
                value={fps}
                onChange={(event) =>
                  (sharedPlaybackEnabled ? onSharedFpsChange : setLocalFps)?.(
                    Math.max(1, Number.parseInt(event.target.value, 10) || 1),
                  )
                }
              />
            </Form.Group>

            <Form.Group className='flex-grow-1'>
              <Form.Label className='fw-semibold mb-1'>Scrub</Form.Label>
              <Form.Range
                min={0}
                max={maxFrameIndex}
                value={overlayFrameIndex}
                onChange={(event) =>
                  handleFrameScrub(Number.parseInt(event.target.value, 10))
                }
              />
            </Form.Group>

            <Form.Group>
              <Form.Label className='fw-semibold mb-1'>
                Overlay Layers
              </Form.Label>
              <div className='d-flex flex-wrap gap-3'>
                <Form.Check
                  type='checkbox'
                  id='show-bounding-box'
                  label='Bounding Box'
                  checked={showBoundingBox}
                  onChange={() => setShowBoundingBox((value) => !value)}
                />
                <Form.Check
                  type='checkbox'
                  id='show-foot-marker'
                  label='Foot Marker'
                  checked={showFootMarker}
                  onChange={() => setShowFootMarker((value) => !value)}
                />
              </div>
            </Form.Group>
          </Stack>

          <div className='video-overlay-metadata'>
            <span>
              Time <strong>{currentTime.toFixed(2)}s</strong>
            </span>
            <span>
              Frame <strong>{overlayFrameIndex}</strong>
            </span>
            {overlayFrame?.source && (
              <span>
                Possession <strong>{overlayFrame.source}</strong>
              </span>
            )}
            {overlayFrame?.reason && (
              <span>
                Ball <strong>{overlayFrame.reason}</strong>
              </span>
            )}
            {overlayFrame?.playerFootCourt && (
              <span>
                Foot{' '}
                <strong>
                  {`(${overlayFrame.playerFootCourt[0].toFixed(
                    1,
                  )}, ${overlayFrame.playerFootCourt[1].toFixed(1)})`}
                </strong>
              </span>
            )}
          </div>
        </>
      )}
    </Stack>
  );
};

export default VideoOverlayPlayer;
