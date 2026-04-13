import { useState } from 'react';
import './App.css';
import { Button, Card, Col, Container, Form, Row, Stack } from 'react-bootstrap';
import type { ChangeEvent } from 'react';
import type { RunInferenceOptions } from './types/RunInferenceOptions';

function App() {
  const [file, setFile] = useState<File | null>(null);
  const [options, setOptions] = useState<RunInferenceOptions>({
    saveOverlays: true,
    basketSide: 'LEFT',
    device: 'GPU',
    frameStep: 1,
    holdFrames: 8,
  });

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    setFile(event.target.files?.[0] || null);
  };

  return (
    <Container fluid='lg' className='py-4 py-md-5 text-start'>
      <Stack gap={1} className='mb-4 mb-md-5'>
        <h1 className='mb-0'>Shot Vision</h1>
        <p className='text-body-secondary mb-0'>
          Upload a clip and adjust the inference settings before running the
          pipeline.
        </p>
      </Stack>

      <Row className='g-4 align-items-start'>
        <Col lg={5} xl={4}>
          <Card className='h-100 shadow-sm'>
            <Card.Body className='p-4'>
              <Stack gap={4}>
                <div>
                  <h2 className='h4 mb-1'>Options</h2>
                  <p className='text-body-secondary mb-0'>
                    Configure the video inference run.
                  </p>
                </div>

                <Form>
                  <Form.Group className='mb-3'>
                    <Form.Check
                      type='checkbox'
                      label='Save overlays'
                      checked={options.saveOverlays}
                      onChange={(e) =>
                        setOptions({ ...options, saveOverlays: e.target.checked })
                      }
                    />
                  </Form.Group>

                  <Form.Group className='mb-3'>
                    <Form.Label className='fw-semibold'>Basket Side</Form.Label>
                    <Form.Select
                      value={options.basketSide}
                      onChange={(e) =>
                        setOptions({
                          ...options,
                          basketSide: e.target.value as 'LEFT' | 'RIGHT',
                        })
                      }
                    >
                      <option value='LEFT'>Left</option>
                      <option value='RIGHT'>Right</option>
                    </Form.Select>
                  </Form.Group>

                  <Form.Group className='mb-3'>
                    <Form.Label className='fw-semibold'>Device</Form.Label>
                    <Form.Select
                      value={options.device}
                      onChange={(e) =>
                        setOptions({
                          ...options,
                          device: e.target.value as 'CPU' | 'GPU',
                        })
                      }
                    >
                      <option value='CPU'>CPU</option>
                      <option value='GPU'>GPU</option>
                    </Form.Select>
                  </Form.Group>

                  <Form.Group className='mb-3'>
                    <Form.Label className='fw-semibold'>Frame Step</Form.Label>
                    <Form.Control
                      type='number'
                      min={1}
                      value={options.frameStep}
                      onChange={(e) =>
                        setOptions({
                          ...options,
                          frameStep: parseInt(e.target.value) || 1,
                        })
                      }
                    />
                  </Form.Group>

                  <Form.Group>
                    <Form.Label className='fw-semibold'>Hold Frames</Form.Label>
                    <Form.Control
                      type='number'
                      min={0}
                      value={options.holdFrames}
                      onChange={(e) =>
                        setOptions({
                          ...options,
                          holdFrames: parseInt(e.target.value) || 0,
                        })
                      }
                    />
                  </Form.Group>
                </Form>
              </Stack>
            </Card.Body>
          </Card>
        </Col>

        <Col lg={7} xl={8}>
          <Card className='shadow-sm'>
            <Card.Body className='p-4'>
              <Stack gap={4}>
                <div>
                  <h2 className='h4 mb-1'>Upload Video</h2>
                  <p className='text-body-secondary mb-0'>
                    Select an MP4 file to run the current inference pipeline.
                  </p>
                </div>

                <Form.Group>
                  <Form.Label className='fw-semibold'>Video File</Form.Label>
                  <Form.Control
                    type='file'
                    accept='.mp4'
                    onChange={(e) => handleFileChange(e as ChangeEvent<HTMLInputElement>)}
                  />
                  <Form.Text className='text-body-secondary'>
                    {file ? `Selected: ${file.name}` : 'No file selected yet.'}
                  </Form.Text>
                </Form.Group>

                <div className='d-flex gap-2'>
                  <Button
                    disabled={!file}
                    onClick={() =>
                      file ? alert(`Processing ${file.name}...`) : undefined
                    }
                  >
                    Process Video
                  </Button>
                </div>
              </Stack>
            </Card.Body>
          </Card>
        </Col>
      </Row>
    </Container>
  );
}

export default App;
