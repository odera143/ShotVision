import { useEffect, useMemo, useRef, useState } from 'react';
import { HalfCourt } from '../halfcourt';
import type {
  FullInferenceSummary,
  InferenceSummary,
  PossessionOnlyInferenceSummary,
} from '../inference/types/InferenceSummary';
import Form from 'react-bootstrap/esm/Form';
import { Alert, Button, Stack } from 'react-bootstrap';

type RenderTopDownProps = {
  results: InferenceSummary | null;
  basketSide: 'LEFT' | 'RIGHT';
  sharedPlaybackEnabled?: boolean;
  sharedFrameIndex?: number;
  onSharedFrameIndexChange?: (frameIndex: number) => void;
};

const isPossessionOnlySummary = (
  results: InferenceSummary,
): results is PossessionOnlyInferenceSummary =>
  'summary_type' in results && results.summary_type === 'POSSESSION_ONLY';

const getCourtPositionForFrame = (
  results: InferenceSummary | null,
  frameIndex: number,
): [number, number] | null => {
  if (!results || results.frames.length === 0) {
    return null;
  }

  const clampedFrameIndex = Math.min(
    results.frames.length - 1,
    Math.max(0, Math.trunc(frameIndex)),
  );

  if (isPossessionOnlySummary(results)) {
    return (results.frames[clampedFrameIndex].possession.player_foot_court_xy ??
      null) as [number, number] | null;
  }

  return (
    (results as FullInferenceSummary).frames[clampedFrameIndex].smoothed_possession
      .player_foot_court_xy ?? null
  ) as [number, number] | null;
};

const RenderTopDown = ({
  results,
  basketSide,
  sharedPlaybackEnabled = false,
  sharedFrameIndex,
  onSharedFrameIndexChange,
}: RenderTopDownProps) => {
  const frames = results?.frames ?? [];
  const maxFrameIndex = Math.max(0, frames.length - 1);
  const [currentFrameIndex, setCurrentFrameIndex] = useState(0);

  //might or might not use this, can be useful for smoothing out animations if too jittery.
  const [frameStep, _] = useState(1);

  const playbackIntervalRef = useRef<number | null>(null);

  const clampFrameIndex = (value: number) => {
    if (!Number.isFinite(value)) {
      return 0;
    }

    return Math.min(maxFrameIndex, Math.max(0, Math.trunc(value)));
  };

  const stopPlayback = () => {
    if (playbackIntervalRef.current !== null) {
      window.clearInterval(playbackIntervalRef.current);
      playbackIntervalRef.current = null;
    }
  };

  useEffect(() => {
    setCurrentFrameIndex((previousFrameIndex) =>
      clampFrameIndex(previousFrameIndex),
    );
  }, [maxFrameIndex]);

  useEffect(() => stopPlayback, []);

  useEffect(() => {
    if (sharedPlaybackEnabled) {
      stopPlayback();
    }
  }, [sharedPlaybackEnabled]);

  const playThroughFrames = () => {
    if (frames.length === 0 || playbackIntervalRef.current !== null) {
      return;
    }

    if (currentFrameIndex >= maxFrameIndex) {
      setCurrentFrameIndex(0);
    }

    playbackIntervalRef.current = window.setInterval(() => {
      let shouldStop = false;
      setCurrentFrameIndex((previousFrameIndex) => {
        const nextFrameIndex = Math.min(
          maxFrameIndex,
          previousFrameIndex + frameStep,
        );
        shouldStop = nextFrameIndex >= maxFrameIndex;
        return nextFrameIndex;
      });

      if (shouldStop) {
        stopPlayback();
      }
    }, 33 * frameStep); //approx 30 fps
  };

  const resolvedFrameIndex = sharedPlaybackEnabled
    ? clampFrameIndex(sharedFrameIndex ?? 0)
    : clampFrameIndex(currentFrameIndex);

  const currentFrame = useMemo(() => {
    if (frames.length === 0) {
      return null;
    }

    return frames[resolvedFrameIndex];
  }, [frames, resolvedFrameIndex]);

  const currentCourtPosition = useMemo(
    () => getCourtPositionForFrame(results, resolvedFrameIndex),
    [resolvedFrameIndex, results],
  );

  const displayCourtPosition = useMemo(() => {
    if (!currentCourtPosition) {
      return null;
    }

    const [x, y] = currentCourtPosition;
    return [basketSide === 'RIGHT' ? -x : x, y] as [number, number];
  }, [basketSide, currentCourtPosition]);

  const setFrameIndex = (nextFrameIndex: number) => {
    const clamped = clampFrameIndex(nextFrameIndex);

    if (sharedPlaybackEnabled) {
      onSharedFrameIndexChange?.(clamped);
      return;
    }

    setCurrentFrameIndex(clamped);
  };

  return (
    <Stack gap={3} className='topdown-view align-items-center'>
      <div className='text-center'>
        <h2 className='h4 mb-1'>Top-Down Court</h2>
        <p className='text-body-secondary mb-0'>
          {sharedPlaybackEnabled
            ? 'This view is following the same playback as the video overlay.'
            : 'Use the controls below to step through possession positions.'}
        </p>
      </div>
      <HalfCourt
        handlerXY={displayCourtPosition}
      />
      {!currentFrame && (
        <Alert variant='secondary' className='mb-0'>
          No court positions are available yet.
        </Alert>
      )}
      <div className='d-flex gap-1'>
        {!sharedPlaybackEnabled && (
          <>
            <Button onClick={() => playThroughFrames()}>Play</Button>
            <Button onClick={() => stopPlayback()}>Pause</Button>
          </>
        )}
        <Button
          onClick={() => {
            stopPlayback();
            setFrameIndex(resolvedFrameIndex - frameStep);
          }}
        >
          Previous
        </Button>
        <Button
          onClick={() => {
            stopPlayback();
            setFrameIndex(resolvedFrameIndex + frameStep);
          }}
        >
          Next
        </Button>
        <Form.Control
          type='number'
          value={resolvedFrameIndex}
          min={0}
          max={maxFrameIndex}
          onChange={(e) => {
            stopPlayback();
            const parsedValue = Number.parseInt(e.target.value, 10);
            setFrameIndex(parsedValue);
          }}
        />
      </div>
    </Stack>
  );
};
export default RenderTopDown;
