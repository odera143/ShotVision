export type PlayerShotGridCell = {
  x: number;
  y: number;
  att: number;
  made: number;
  fg: number;
  pts: number;
  is3: boolean;
};

export type PlayerShotGrid = {
  playerId: number;
  season: string;
  seasonType: string;
  gridFt: number;
  cells: PlayerShotGridCell[];
};
