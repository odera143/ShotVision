import { useEffect, useState } from 'react';
import './App.css';
import {
  Alert,
  Button,
  Card,
  Col,
  Container,
  Form,
  OverlayTrigger,
  Row,
  Spinner,
  Stack,
  Tooltip,
} from 'react-bootstrap';
import type { ChangeEvent } from 'react';
import type { RunInferenceOptions } from './types/RunInferenceOptions';
import type { InferenceJob } from './types/InferenceJob';
import type { InferenceSummary } from './types/InferenceSummary';

const API_BASE_URL = 'http://localhost:8080';

function App() {
  const [file, setFile] = useState<File | null>(null);
  const [options, setOptions] = useState<RunInferenceOptions>({
    saveOverlays: false,
    basketSide: 'LEFT',
    device: 'GPU',
    frameStep: 1,
    holdFrames: 8,
    resultsMode: 'POSSESSION_ONLY',
  });
  const [job, setJob] = useState<InferenceJob | null>(null);
  const [jobStatus, setJobStatus] = useState<InferenceJob['status']>();
  const [results, setResults] = useState<InferenceSummary | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [overlayArtifactName, setOverlayArtifactName] = useState<string | null>(
    null,
  );

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    setFile(event.target.files?.[0] || null);
  };

  const handleSubmit = async () => {
    if (!file) return;

    setOverlayArtifactName(null);
    setResults(null);
    setError(null);
    setJob(null);
    setJobStatus(undefined);

    const formData = new FormData();
    formData.append('video', file);
    formData.append('save_overlays', String(options.saveOverlays));
    formData.append('basket_side', options.basketSide);
    formData.append('device', options.device);
    formData.append('frame_step', String(options.frameStep));
    formData.append('hold_frames', String(options.holdFrames));
    formData.append('results_mode', options.resultsMode);

    const response = await fetch(`${API_BASE_URL}/jobs`, {
      method: 'POST',
      body: formData,
    });

    const data: InferenceJob = await response.json();
    setJob(data);
    setJobStatus(data.status);
    console.log(data);
  };

  useEffect(() => {
    if (!job?.job_id) return;
    if (jobStatus === 'completed' || jobStatus === 'failed') return;

    const interval = setInterval(async () => {
      const res = await fetch(`${API_BASE_URL}${job.status_url}`);
      const status: InferenceJob = await res.json();
      setJob(status);
      setJobStatus(status.status);

      if (status.status === 'completed') {
        window.clearInterval(interval);
        const resultsRes = await fetch(`${API_BASE_URL}${status.results_url}`);
        const resultsJson = await resultsRes.json();
        setResults(resultsJson);
        if (status.overlay_video_url) {
          await fetchOverlayVideo(status.overlay_video_url);
        }
      }

      if (status.status === 'failed') {
        window.clearInterval(interval);
        setError('Job failed.');
      }
    }, 3000);

    return () => clearInterval(interval);
  }, [job, jobStatus]);
  const extractDownloadName = (headers: Headers) => {
    const disposition = headers.get('content-disposition');
    if (!disposition) return null;

    const utf8Match = disposition.match(/filename\*=UTF-8''([^;]+)/i);
    if (utf8Match?.[1]) {
      return decodeURIComponent(utf8Match[1]);
    }

    const filenameMatch = disposition.match(/filename="?([^"]+)"?/i);
    return filenameMatch?.[1] ?? null;
  };

  const fetchOverlayVideo = async (overlayPath: string) => {
    const res = await fetch(`${API_BASE_URL}${overlayPath}`);

    if (!res.ok) {
      setError('Failed to fetch overlay video.');
      return;
    }

    const blob = await res.blob();
    const filename =
      extractDownloadName(res.headers) ??
      `${job?.job_id ?? 'shot-vision'}_overlay.mp4`;
    const objectUrl = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = objectUrl;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    link.remove();
    window.setTimeout(() => URL.revokeObjectURL(objectUrl), 1000);
    setOverlayArtifactName(filename);
  };

  const renderLabelWithTooltip = (label: string, tooltip: string) => (
    <span className='d-inline-flex align-items-center gap-2'>
      <span>{label}</span>
      <OverlayTrigger
        placement='right'
        overlay={
          <Tooltip id={`tooltip-${label.toLowerCase().replace(/\s+/g, '-')}`}>
            {tooltip}
          </Tooltip>
        }
      >
        <svg
          xmlns='http://www.w3.org/2000/svg'
          width='0.8rem'
          height='0.8rem'
          fill='currentColor'
          className='bi bi-question-circle'
          viewBox='0 0 18 18'
        >
          <path d='M8 15A7 7 0 1 1 8 1a7 7 0 0 1 0 14m0 1A8 8 0 1 0 8 0a8 8 0 0 0 0 16' />
          <path d='M5.255 5.786a.237.237 0 0 0 .241.247h.825c.138 0 .248-.113.266-.25.09-.656.54-1.134 1.342-1.134.686 0 1.314.343 1.314 1.168 0 .635-.374.927-.965 1.371-.673.489-1.206 1.06-1.168 1.987l.003.217a.25.25 0 0 0 .25.246h.811a.25.25 0 0 0 .25-.25v-.105c0-.718.273-.927 1.01-1.486.609-.463 1.244-.977 1.244-2.056 0-1.511-1.276-2.241-2.673-2.241-1.267 0-2.655.59-2.75 2.286m1.557 5.763c0 .533.425.927 1.01.927.609 0 1.028-.394 1.028-.927 0-.552-.42-.94-1.029-.94-.584 0-1.009.388-1.009.94' />
        </svg>
      </OverlayTrigger>
    </span>
  );

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
                      label={renderLabelWithTooltip(
                        'Save overlays',
                        'Returns an output video with detections, paint geometry, and court coordinates drawn on each frame.',
                      )}
                      checked={options.saveOverlays}
                      onChange={(e) =>
                        setOptions({
                          ...options,
                          saveOverlays: e.target.checked,
                        })
                      }
                    />
                  </Form.Group>

                  <Form.Group className='mb-3'>
                    <Form.Label className='fw-semibold'>
                      {renderLabelWithTooltip(
                        'Basket Side',
                        'Choose which side of the frame contains the visible hoop so the paint homography is oriented correctly.',
                      )}
                    </Form.Label>
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
                    <Form.Label className='fw-semibold'>
                      {renderLabelWithTooltip(
                        'Device',
                        'Select GPU for faster inference when CUDA is available, or CPU if you want to run without the graphics card.',
                      )}
                    </Form.Label>
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
                    <Form.Label className='fw-semibold'>
                      {renderLabelWithTooltip(
                        'Frame Step',
                        'Processes every Nth frame. Higher values run faster but skip more frames.',
                      )}
                    </Form.Label>
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

                  <Form.Group className='mb-3'>
                    <Form.Label className='fw-semibold'>
                      {renderLabelWithTooltip(
                        'Results Mode',
                        'Choose whether the API should return the full run summary or only the possession data needed for the court-view UI.',
                      )}
                    </Form.Label>
                    <Form.Select
                      value={options.resultsMode}
                      onChange={(e) =>
                        setOptions({
                          ...options,
                          resultsMode: e.target.value as
                            | 'FULL'
                            | 'POSSESSION_ONLY',
                        })
                      }
                    >
                      <option value='POSSESSION_ONLY'>Possession only</option>
                      <option value='FULL'>Full summary</option>
                    </Form.Select>
                  </Form.Group>

                  <Form.Group>
                    <Form.Label className='fw-semibold'>
                      {renderLabelWithTooltip(
                        'Hold Frames',
                        'Keeps the current ball handler for this many uncertain frames before dropping possession.',
                      )}
                    </Form.Label>
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
          <Alert show={!!error} variant='danger'>
            {error}
          </Alert>
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
                    onChange={(e) =>
                      handleFileChange(e as ChangeEvent<HTMLInputElement>)
                    }
                  />
                  <Form.Text className='text-body-secondary'>
                    {file ? `Selected: ${file.name}` : 'No file selected yet.'}
                  </Form.Text>
                </Form.Group>

                <div className='d-flex gap-2'>
                  <Button
                    disabled={
                      !file || jobStatus === 'queued' || jobStatus === 'running'
                    }
                    onClick={handleSubmit}
                  >
                    {jobStatus === 'queued' || jobStatus === 'running' ? (
                      <>
                        <Spinner size='sm' animation='border' as='span' />
                        &nbsp;<span>{jobStatus}</span>
                      </>
                    ) : (
                      'Process Video'
                    )}
                  </Button>
                </div>
              </Stack>
            </Card.Body>
          </Card>
          {results && (
            <Card className='mt-4 shadow-sm'>
              <Card.Body className='p-4'>
                <Stack gap={3}>
                  <div>
                    <h2 className='h4 mb-1'>Results</h2>
                    <p className='text-body-secondary mb-0'>
                      View the inference summary JSON returned by the API.
                    </p>
                  </div>
                  <pre
                    style={{
                      maxHeight: '400px',
                      overflowY: 'auto',
                      padding: '1rem',
                      borderRadius: '0.25rem',
                    }}
                  >
                    {JSON.stringify(results, null, 2)}
                  </pre>
                </Stack>
              </Card.Body>
            </Card>
          )}
          {overlayArtifactName && (
            <Card className='mt-4 shadow-sm'>
              <Card.Body className='p-4'>
                <Stack gap={3}>
                  <div>
                    <h2 className='h4 mb-1'>Overlay Debug Artifact</h2>
                    <p className='text-body-secondary mb-0'>
                      The overlay video was downloaded for local debugging.
                    </p>
                  </div>
                  <Alert variant='secondary' className='mb-0'>
                    Downloaded <strong>{overlayArtifactName}</strong>.
                  </Alert>
                </Stack>
              </Card.Body>
            </Card>
          )}
        </Col>
      </Row>
    </Container>
  );
}

export default App;
