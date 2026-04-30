const PlayerStatsTooltip = ({
  playerTooltipPosition,
}: {
  playerTooltipPosition: { left: string; top: string; isNearTop: boolean };
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
      <span>Distance from basket:</span>
      <span>FG%:</span>
      <span>Makes / Attempts:</span>
      <span>Expected pts:</span>
    </div>
  );
};
export default PlayerStatsTooltip;
