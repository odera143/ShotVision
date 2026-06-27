import { findShotGridCell, formatShotGridPercent } from './shotGridStats';
import type { PlayerShotGrid } from './types/PlayerShotGrid';

const PlayerStatsTooltip = ({
  playerTooltipPosition,
  playerFootPosition,
  playerShotGrid,
}: {
  playerTooltipPosition: { left: string; top: string; isNearTop: boolean };
  playerFootPosition: [number, number] | null;
  playerShotGrid: PlayerShotGrid | null;
}) => {
  const shotGridCell = findShotGridCell(playerFootPosition, playerShotGrid);

  return (
    <div
      className={`video-overlay-tooltip ${
        playerTooltipPosition.isNearTop ? 'video-overlay-tooltip-below' : ''
      }`}
      style={{
        left: playerTooltipPosition.left,
        top: playerTooltipPosition.top,
      }}
    >
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          gap: 12,
        }}
      >
        <span>Distance:</span>
        <span>
          {playerFootPosition
            ? Math.sqrt(
                playerFootPosition[0] * playerFootPosition[0] +
                  playerFootPosition[1] * playerFootPosition[1],
              ).toFixed(1)
            : 0}
          {' ft'}
        </span>
      </div>
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
        }}
      >
        <span>FG%:</span>
        <span>
          {shotGridCell ? formatShotGridPercent(shotGridCell.fg) : '--'}
        </span>
      </div>
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
        }}
      >
        <span>Made/Att:</span>
        <span>
          {shotGridCell
            ? `${shotGridCell.made}/${shotGridCell.att}`
            : '--'}
        </span>
      </div>
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
        }}
      >
        <span>Expected pts:</span>
        <span>{shotGridCell ? shotGridCell.pts.toFixed(2) : '--'}</span>
      </div>
    </div>
  );
};
export default PlayerStatsTooltip;
