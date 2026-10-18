# SecureVision — Final Hackathon Submission

## 1. Overview

SecureVision is a local AI-powered video analytics platform
for reviewing cash-discipline events and workplace activity.

The system accepts MOV recordings and produces structured
analysis results, timestamps, visual evidence, and
workplace activity indicators.

Automated findings are intended for human review.
They are not final legal conclusions.

## 2. Architecture

Browser
  |
  v
Next.js frontend (port 3000)
  |
  v
NestJS backend (port 4000)
  |
  v
FastAPI AI service (port 8000)
  |
  +-- Cash Discipline AI
  |     X3D-XS pretrained video features
  |     scikit-learn classification
  |     temporal candidate detection
  |     JPEG evidence extraction
  |
  +-- Workplace Activity Analytics
        TorchVision SSDLite person detection
        anonymous centroid tracking
        cashier work-zone occupancy
        sitting/standing heuristic
        absence calculation

## 3. Main Features

- MOV upload and asynchronous background analysis
- Job ID and job-status polling
- Suspicious event timestamps
- Base64 JPEG evidence frames
- Kazakhstan legal-reference context
- Presence and absence statistics
- Sitting and standing activity
- Anonymous object tracking
- Video bounding boxes and work-zone overlay
- Annotated frame gallery
- JSON report download

## 4. Technologies

Frontend: Next.js, React, TypeScript
Backend: NestJS, Node.js
AI API: Python, FastAPI
AI/ML: PyTorch, TorchVision, PyTorchVideo, scikit-learn
Video processing: OpenCV
Monorepo: Turborepo

## 5. API

Primary competition API:

POST /api/v1/analyze
GET  /api/v1/jobs/{job_id}

Additional AI APIs:

POST /api/v1/workplace/analyze
POST /api/v1/full/analyze
GET  /api/v1/full/jobs/{job_id}

Frontend-facing NestJS API:

POST /api/analysis/full
GET  /api/analysis/full/{job_id}

The internal FastAPI endpoints use X-API-Key authentication.

## 6. Dataset

The development dataset contained 42 MOV entries,
including duplicated video content.

After content deduplication:
- 40 unique videos
- 32 training videos
- 8 development-validation videos

The labels primarily describe binary violation status
and event timing, not specific legal violation categories.

Private original video recordings are excluded from
public source-code releases.

## 7. Validation and Limitations

The software pipeline has been tested end-to-end
on real MOV recordings.

On the eight-video development validation set,
the cash classifier found all four positive videos
but also flagged all four negative videos.

Therefore the current prototype has a significant
false-positive limitation and is not suitable for
automatic legal enforcement.

Workplace posture uses camera-calibrated bounding-box
geometry and should not be treated as a universal
posture-recognition model.

Anonymous Track IDs can change when people cross paths.
They are not personal identities.

## 8. Legal Context

Reference source:
National Bank of Kazakhstan Resolution No. 231,
29 November 2019.

https://adilet.zan.kz/rus/docs/V1900019680

Legal context is informational.
A qualified reviewer must verify the current applicable
legal text and examine each specific event.

## 9. Privacy and Deployment

- No facial recognition or biometric identification
- AI inference executes locally
- No external AI inference service is required
- Source videos should remain under controlled access
- Temporary uploaded videos are removed after jobs
- API keys must not be included in frontend bundles

Offline use requires installed dependencies and
pre-provisioned model weights.

## 10. Known Operational Limitations

- In-memory job stores are lost on server restart
- GPU/CPU runtime depends on the target machine
- All model licenses and weight provenance must be
  verified before claiming unrestricted commercial use
- Broader testing is needed before production deployment

## 11. Demo

Use the demo start script, open localhost:3000,
upload a permitted MOV file, wait for results,
inspect evidence, workplace analytics, and JSON export.
