import type {
  PlayerShotGrid,
  PlayerShotGridCell,
} from './types/PlayerShotGrid';

export const formatShotGridPercent = (value: number) =>
  `${Math.round(value * 100)}%`;

export const findShotGridCell = (
  playerFootPosition: [number, number] | null,
  playerShotGrid: PlayerShotGrid | null,
  radiusFt: number,
  minAttempts: number,
): PlayerShotGridCell | null => {
  if (!playerFootPosition || !playerShotGrid?.cells.length) {
    return null;
  }

  const [x, y] = playerFootPosition;
  const radius = Math.max(0, radiusFt);
  const minAtt = Math.max(1, minAttempts);
  const matchingCells = playerShotGrid.cells.filter(
    (cell) => Math.hypot(cell.x - x, cell.y - y) <= radius,
  );

  if (matchingCells.length === 0) {
    return null;
  }

  const att = matchingCells.reduce((total, cell) => total + cell.att, 0);

  if (att < minAtt) {
    return null;
  }

  const made = matchingCells.reduce((total, cell) => total + cell.made, 0);
  const expectedPoints = matchingCells.reduce(
    (total, cell) => total + cell.pts * cell.att,
    0,
  );

  return {
    x,
    y,
    att,
    made,
    fg: made / att,
    pts: expectedPoints / att,
    is3: matchingCells.every((cell) => cell.is3),
  };
};
