import { useEffect, useMemo, useRef, useState } from 'react';
import { HalfCourt } from '../halfcourt';
import type { PossessionOnlyInferenceSummary } from '../inference/types/InferenceSummary';
import Form from 'react-bootstrap/esm/Form';
import { Button } from 'react-bootstrap';

const RenderTopDown = ({
  results,
}: {
  results: PossessionOnlyInferenceSummary;
}) => {
  const frames = results.frames;
  const maxFrameIndex = Math.max(0, frames.length - 1);
  const [currentFrameIndex, setCurrentFrameIndex] = useState(0);
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
        const nextFrameIndex = Math.min(maxFrameIndex, previousFrameIndex + 1);
        shouldStop = nextFrameIndex >= maxFrameIndex;
        return nextFrameIndex;
      });

      if (shouldStop) {
        stopPlayback();
      }
    }, 33); //approx 30 fps
  };

  const currentFrame = useMemo(() => {
    if (frames.length === 0) {
      return null;
    }

    return frames[clampFrameIndex(currentFrameIndex)];
  }, [clampFrameIndex, currentFrameIndex, frames]);

  return (
    <div className='d-flex flex-column align-items-center gap-3'>
      <HalfCourt
        handlerXY={
          currentFrame?.possession.player_foot_court_xy as
            | [number, number]
            | null
            | undefined
        }
      />
      <div className='d-flex gap-1'>
        <Button onClick={() => playThroughFrames()}>Play</Button>
        <Button onClick={() => stopPlayback()}>Pause</Button>
        <Button
          onClick={() => {
            stopPlayback();
            setCurrentFrameIndex((i) => Math.max(0, i - 1));
          }}
        >
          Previous
        </Button>
        <Button
          onClick={() => {
            stopPlayback();
            setCurrentFrameIndex((i) => Math.min(maxFrameIndex, i + 1));
          }}
        >
          Next
        </Button>
        <Form.Control
          type='number'
          value={currentFrameIndex}
          min={0}
          max={maxFrameIndex}
          onChange={(e) => {
            stopPlayback();
            const parsedValue = Number.parseInt(e.target.value, 10);
            setCurrentFrameIndex(clampFrameIndex(parsedValue));
          }}
        />
      </div>
    </div>
  );
};
export default RenderTopDown;
