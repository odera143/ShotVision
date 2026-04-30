const PlayerStatsTooltip = ({
  playerTooltipPosition,
  playerFootPosition,
}: {
  playerTooltipPosition: { left: string; top: string; isNearTop: boolean };
  playerFootPosition: [number, number] | null;
}) => {
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
        <span>40%</span>
      </div>
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
        }}
      >
        <span>Made/Att:</span>
        <span>4/10</span>
      </div>
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
        }}
      >
        <span>Expected pts:</span>
        <span>6.8</span>
      </div>
    </div>
  );
};
export default PlayerStatsTooltip;
