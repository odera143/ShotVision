import { useQuery } from '@tanstack/react-query';
import React, { useState } from 'react';
import { Alert, Button, Card, Form, ListGroup, Stack } from 'react-bootstrap';

type PlayerShotGridSubmitValue = {
  requestParams: Record<string, string>;
  radiusFt: number;
  minAttempts: number;
};

const PlayerForm = ({
  onSubmit,
}: {
  onSubmit: (value: PlayerShotGridSubmitValue) => void;
}) => {
  const API_BASE_URL = 'http://localhost:8080';
  const [playerQuery, setPlayerQuery] = useState('');
  const [selectedPlayer, setSelectedPlayer] = useState<{
    id: string;
    name: string;
  } | null>(null);
  const [selectedSeason, setSelectedSeason] = useState<string>('2022-23');
  const [selectedSeasonType, setSelectedSeasonType] =
    useState<string>('Regular Season');
  const [minAtt, setMinAtt] = useState<number>(3);
  const [radiusFt, setRadiusFt] = useState<number>(5);

  const {
    data: playerData,
    isLoading,
    error,
  } = useQuery({
    queryKey: ['players', playerQuery],
    enabled: playerQuery.trim().length > 2 && !selectedPlayer,
    queryFn: () =>
      fetch(
        `${API_BASE_URL}/players?q=${encodeURIComponent(playerQuery)}`,
      ).then((res) => res.json()),
  });
  const showPlayerSuggestions =
    playerQuery.trim().length > 2 && !selectedPlayer;

  const buildRequestParams = (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedPlayer) return;
    const requestParams = {
      player_id: selectedPlayer.id,
      season: selectedSeason,
      season_type: selectedSeasonType,
      grid: '1',
      min_att: '1',
    };
    onSubmit({
      requestParams,
      radiusFt,
      minAttempts: minAtt,
    });
  };

  return (
    <Card className='w-100 shadow-sm'>
      <Card.Body className='p-3 p-md-4'>
        <Form onSubmit={buildRequestParams}>
          <Stack gap={3}>
            <Form.Group controlId='player-search'>
              <Form.Label className='fw-semibold'>Player Name</Form.Label>
              <Form.Control
                type='text'
                placeholder='Search player name'
                value={playerQuery}
                onChange={(e) => {
                  setSelectedPlayer(null);
                  setPlayerQuery(e.target.value);
                }}
              />
            </Form.Group>

            {showPlayerSuggestions && (
              <div>
                {error && (
                  <Alert className='mb-2' variant='danger'>
                    Something went wrong
                  </Alert>
                )}
                <ListGroup
                  className='border rounded'
                  style={{
                    maxHeight: '14rem',
                    minHeight: '3rem',
                    overflowY: 'auto',
                  }}
                >
                  {isLoading && (
                    <ListGroup.Item disabled>Loading...</ListGroup.Item>
                  )}
                  {!isLoading && playerData?.length === 0 && (
                    <ListGroup.Item disabled>No players found.</ListGroup.Item>
                  )}
                  {!isLoading &&
                    playerData?.map((player: { id: string; name: string }) => (
                      <ListGroup.Item
                        action
                        key={player.id}
                        onClick={() => {
                          setSelectedPlayer(player);
                          setPlayerQuery(player.name);
                        }}
                      >
                        {player.name}
                      </ListGroup.Item>
                    ))}
                </ListGroup>
              </div>
            )}

            <Form.Group controlId='season-select'>
              <Form.Label className='fw-semibold'>Season</Form.Label>
              <Form.Select
                name='season'
                value={selectedSeason}
                onChange={(e) => setSelectedSeason(e.target.value)}
              >
                <option value='2018-19'>2018-19</option>
                <option value='2019-20'>2019-20</option>
                <option value='2020-21'>2020-21</option>
                <option value='2021-22'>2021-22</option>
                <option value='2022-23'>2022-23</option>
                <option value='2023-24'>2023-24</option>
                <option value='2024-25'>2024-25</option>
                <option value='2025-26'>2025-26</option>
              </Form.Select>
            </Form.Group>

            <Form.Group controlId='season-type-select'>
              <Form.Label className='fw-semibold'>Season Type</Form.Label>
              <Form.Select
                name='seasonType'
                value={selectedSeasonType}
                onChange={(e) => setSelectedSeasonType(e.target.value)}
              >
                <option value='Regular Season'>Regular Season</option>
                <option value='Playoffs'>Playoffs</option>
              </Form.Select>
            </Form.Group>

            <Form.Group controlId='min-attempts'>
              <Form.Label className='fw-semibold'>
                Minimum Attempts
              </Form.Label>
              <Form.Control
                type='number'
                min={1}
                value={minAtt}
                onChange={(e) => setMinAtt(Math.max(1, Number(e.target.value)))}
              />
            </Form.Group>

            <Form.Group controlId='shot-radius'>
              <Form.Label className='fw-semibold'>Shot Radius (ft)</Form.Label>
              <Form.Control
                type='number'
                min={1}
                value={radiusFt}
                onChange={(e) =>
                  setRadiusFt(Math.max(1, Number(e.target.value)))
                }
              />
            </Form.Group>

            <Button type='submit' variant='primary'>
              Submit
            </Button>
          </Stack>
        </Form>
      </Card.Body>
    </Card>
  );
};
export default PlayerForm;
