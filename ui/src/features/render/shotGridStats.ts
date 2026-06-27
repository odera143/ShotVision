import type {
  PlayerShotGrid,
  PlayerShotGridCell,
} from './types/PlayerShotGrid';

export const formatShotGridPercent = (value: number) =>
  `${Math.round(value * 100)}%`;

export const findShotGridCell = (
  playerFootPosition: [number, number] | null,
  playerShotGrid: PlayerShotGrid | null,
): PlayerShotGridCell | null => {
  if (!playerFootPosition || !playerShotGrid?.cells.length) {
    return null;
  }

  const [x, y] = playerFootPosition;
  const gridFt = Math.max(1, playerShotGrid.gridFt || 1);
  const gridX = Math.round(x / gridFt) * gridFt;
  const gridY = Math.round(y / gridFt) * gridFt;
  return (
    playerShotGrid.cells.find((cell) => cell.x === gridX && cell.y === gridY) ??
    null
  );
};
