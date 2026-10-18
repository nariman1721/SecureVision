# SecureVision

### AI-Powered Cash Discipline Monitoring & Workplace Activity Analytics

**SecureVision** is an intelligent video analytics platform developed for the **ForteBank Datathon Hackathon**. It combines Computer Vision, Machine Learning, and modern web technologies to help organizations analyze cashier operations, detect potentially suspicious activities, and monitor workplace activity through video recordings.

The system processes videos locally, identifies events requiring human attention, and presents results through an interactive web dashboard.

**Our mission:** Make video monitoring smarter, more transparent, and more efficient through AI-powered analysis.

## The Problem

Financial organizations and businesses handle numerous cash transactions every day. Monitoring compliance with established operational procedures is essential, but manual video review can be time-consuming.

Traditional CCTV systems record events but usually require employees to review footage manually to identify suspicious behavior or workplace-related issues.

SecureVision addresses these challenges by transforming existing video recordings into structured, searchable analytical results.

## Our Solution

SecureVision combines two complementary AI modules.

### 1. Cash Discipline AI

Analyzes cashier video recordings to identify potentially suspicious visual events.

**Key capabilities:**
- Automated video analysis using Machine Learning.
- Detection of potentially suspicious cash-related activities.
- Identification of event timestamps.
- Extraction of visual evidence frames.
- Model-generated confidence scores.
- Structured event reports.
- References to relevant Kazakhstan regulatory information.
- Human-review recommendations for detected events.

The system acts as a decision-support tool and does not independently determine whether a legal violation occurred.

### 2. Workplace Activity Analytics

Uses Computer Vision to analyze human activity in a defined workplace zone.

**Key capabilities:**
- Person detection using a pretrained object detection model.
- Anonymous tracking with temporary Track IDs.
- Workplace presence monitoring.
- Absence detection and duration calculation.
- Sitting and standing posture estimation.
- Interactive activity timelines.
- Workplace zone monitoring.
- Bounding-box visualization over video recordings.
- Automatic generation of annotated evidence frames.

The system does not use facial recognition or biometric identification.

## Main Features

| Feature | Description |
| Video Upload | Upload MOV recordings through a web interface |
| AI Video Analysis | Process recordings using local ML and Computer Vision models |
| Suspicious Event Detection | Identify potentially suspicious video segments |
| Timestamp Localization | Display the approximate time of detected events |
| Evidence Frames | Extract visual evidence for human review |
| Person Detection | Detect people and draw bounding boxes |
| Anonymous Tracking | Assign temporary identifiers to detected objects |
| Workplace Presence | Measure occupancy of the configured work zone |
| Posture Analytics | Estimate sitting, standing, and unknown postures |
| Activity Timeline | Visualize changes in workplace activity |
| Annotated Video Player | Display detections over the original video |
| Frame Gallery | Generate annotated snapshots at selected timestamps |
| JSON Reporting | Export structured analysis results |
| Local AI Processing | Run model inference without external cloud AI services |

---

## System Architecture

SecureVision follows a full-stack, service-oriented architecture.


                  SECUREVISION
                        |
                        v
              Next.js Frontend
               localhost:3000
                        |
                        v
               NestJS REST API
               localhost:4000
                        |
                        v
               FastAPI AI Engine
               localhost:8000
                        |
           +------------+------------+
           |                         |
           v                         v
    Cash Discipline AI       Workplace Analytics
           |                         |
         X3D-XS                TorchVision
       scikit-learn              SSDLite
       Temporal Scan          Person Detection
       Evidence Frames        Anonymous Tracking
                                     |
                              Zone Monitoring
                              Posture Estimation
           |                         |
           +------------+------------+
                        |
                        v
                Unified JSON
                        |
                        v
                Web Dashboard


### Frontend — Next.js

The frontend provides an interactive dashboard for uploading videos, displaying analysis results, reviewing visual evidence, and exporting reports.

Technologies:
- Next.js 16
- React
- TypeScript
- CSS
- Browser Canvas API

### Backend — NestJS

The backend manages communication between the frontend and the AI engine.

Responsibilities:
- Video upload handling.
- REST API endpoints.
- Job management and status polling.
- Communication with FastAPI.
- Server-side API authentication.

### AI Engine — FastAPI

The AI service performs the main video analysis.

Technologies:
- Python
- FastAPI
- PyTorch
- PyTorchVideo
- TorchVision
- scikit-learn
- OpenCV
- NumPy

The system uses pretrained neural networks and machine-learning classifiers for video feature extraction, suspicious event screening, and person detection.

## AI Models

### Cash Discipline

- **X3D-XS:** Pretrained video feature extraction.
- **scikit-learn:** Classification of extracted features.
- **Temporal analysis:** Detection of candidate segments.
- **OpenCV:** Frame extraction and video processing.

### Workplace Analytics

- **SSDLite320 MobileNetV3 Large:** Person detection.
- **Centroid-based tracking:** Anonymous object tracking.
- **Geometric posture estimation:** Sitting and standing detection.
- **Zone occupancy analysis:** Presence and absence monitoring.

The models operate on video content without requiring facial recognition.

## Project Structure

SecureVision/
|
├── apps/
│   ├── web/                 # Next.js frontend
│   │   └── src/
│   │       ├── app/
│   │       ├── components/
│   │       └── lib/
│   │
│   ├── api/                 # NestJS backend
│   │   └── src/
│   │       └── ai/
│   │
│   └── ai/                  # Python AI engine
│       ├── app/
│       │   ├── model/
│       │   ├── routes/
│       │   ├── services/
│       │   ├── video/
│       │   └── workplace/
│       ├── models/
│       ├── training/
│       └── requirements.txt
│
├── docs/                    # Documentation
├── scripts/                 # Development and test scripts
├── packages/                # Shared Turborepo packages
├── package.json
├── package-lock.json
├── turbo.json
└── README.md


## Installation and Local Setup

### Prerequisites

Before running SecureVision, install:

- Node.js and npm
- Python 3.11
- Git

The AI service requires PyTorch and related Python dependencies. Available memory and processing speed will affect inference performance.

### Step 1 — Clone the Repository

```bash
git clone https://github.com/nariman1721/SecureVision.git
cd SecureVision
```

*The repository is private, so GitHub access is required.*

### Step 2 — Install Node.js Dependencies

```bash
npm ci
```

### Step 3 — Set Up Python

On macOS or Linux:

```bash
python3.11 -m venv apps/ai/.venv
```

Activate the virtual environment:

```bash
source apps/ai/.venv/bin/activate
```

Install Python dependencies:

```bash
pip install -r apps/ai/requirements.txt
```

Deactivate the environment when installation is complete:

```bash
deactivate
```

### Step 4 — Configure Environment Variables

Create the required local `.env` files and configure the API URL, internal AI API key, and service settings according to the backend configuration.

Environment files containing secrets must not be committed to GitHub.

### Step 5 — Prepare Local Models

Ensure that the required model files are available under:

```text
apps/ai/models/
```

The model files include the X3D-XS checkpoint, the trained classification model, and the TorchVision person-detector weights.

The local startup script configures `TORCH_HOME` for the bundled TorchVision weights.

### Step 6 — Run the Project

From the repository root:

```bash
chmod +x scripts/start-demo.sh
./scripts/start-demo.sh
```

The script starts the three applications through Turborepo.

| Service | URL |
| Frontend | http://localhost:3000 |
| Backend API | http://localhost:4000 |
| FastAPI AI | http://localhost:8000 |
| FastAPI Documentation | http://localhost:8000/docs |

Alternatively, after configuring the environment, start the monorepo with:

```bash
npm run dev
```

### Step 7 — Check Service Availability

In another terminal:

```bash
./scripts/check-services.sh
```

All three services should return successful HTTP responses.

---

## API Endpoints

### Cash Discipline Analysis

```http
POST /api/v1/analyze
GET  /api/v1/jobs/{job_id}
```

### Workplace Analytics

```http
POST /api/v1/workplace/analyze
GET  /api/v1/workplace/jobs/{job_id}
```

### Unified Analysis

```http
POST /api/v1/full/analyze
GET  /api/v1/full/jobs/{job_id}
```

### NestJS Frontend API

```http
POST /api/analysis/full
GET  /api/analysis/full/{job_id}
```

The internal FastAPI endpoints use API-key authentication. Refer to the source code and FastAPI documentation for the complete request and response schemas.

## Testing

SecureVision includes scripts for local verification and end-to-end testing.

Check service availability:

```bash
./scripts/check-services.sh
```

Run end-to-end tests:

```bash
python3 scripts/e2e_smoke.py
```

Run the extended test:

```bash
python3 scripts/e2e_smoke.py all
```

These tests require the corresponding local test video files, which are not included in the public source-code package.

### Development Testing

During development, the system was successfully tested with multiple MOV recordings.

It demonstrated:
- Successful video upload and processing.
- Working Cash Discipline inference.
- Working Workplace Analytics.
- Valid structured JSON results.
- Person detection and bounding-box visualization.
- Successful Next.js production builds.

**Important limitation:** The current Cash Discipline classifier has a high false-positive rate on the small development-validation set. Its results must be treated as candidates for human review rather than confirmed violations.

## Privacy and Responsible AI

SecureVision was developed with privacy-conscious design principles.

- Video inference can run locally.
- No facial recognition is used.
- Tracking IDs are temporary and anonymous.
- Uploaded recordings are not sent to external AI inference services during local operation.
- Original private video datasets and annotations are excluded from the repository.
- Internal API keys are managed server-side.
- Automated findings require human verification.

SecureVision is a prototype for analytical assistance and is not an automated legal enforcement or employee-identification system.

## Kazakhstan Regulatory Context

SecureVision includes an informational reference to the National Bank of Kazakhstan's Resolution No. 231 dated November 29, 2019.

Official source:

The applicable legal requirements and current regulatory text must be verified by a qualified specialist. The system does not automatically identify specific legal violations.

## Future Improvements

Planned improvements include:

- Larger and more diverse labeled video datasets.
- Improved classification specificity and fewer false positives.
- More advanced temporal event detection.
- Improved multi-person tracking.
- Posture models that generalize across camera angles.
- Persistent job and report storage.
- Cloud deployment and scalable inference.
- User authentication and role-based permissions.
- Automated report management and analytics.
