# 🎭 Video Swap Pipeline

Hyper-realistic Video-to-Video and Image-to-Video character swapping system.
Upload any video (up to 10 minutes), provide a reference character image, and get back
a video with the character's identity seamlessly transferred — including lip sync,
micro-expressions, and temporal consistency.

## Architecture

```
Mobile Client (Android/Kotlin)          Cloud Backend (Python/FastAPI)
┌────────────────────────────┐          ┌────────────────────────────┐
│  Record/pick video         │          │  FastAPI REST API          │
│  Pick reference character  │──HTTP──▶ │  Celery + Redis workers    │
│  Chunk video (FFmpeg)      │          │  ControlNet + IP-Adapter   │
│  Upload chunks in parallel │          │  Temporal consistency      │
│  Track progress            │◀─HTTP── │  Wav2Lip lip sync          │
│  Download & play result    │          │  FFmpeg stitch             │
└────────────────────────────┘          └────────────────────────────┘
```

## Quick Start

### Option A: Google Colab (Free GPU)

1. Open `VideoSwapPipeline_Colab.ipynb` in Google Colab
2. Select **T4 GPU** runtime
3. Run all cells
4. Copy the public ngrok URL into your Android app

### Option B: Local / Cloud Server

```bash
# 1. Clone and setup
git clone <repo-url> && cd video-swap-pipeline
cp .env.example .env  # Edit as needed

# 2. Start with Docker
docker compose up --build

# 3. Or run manually
pip install -r requirements.txt
redis-server &
celery -A workers.celery_app worker --loglevel=info -Q chunks,stitch &
uvicorn main:app --host 0.0.0.0 --port 8000
```

API docs available at `http://localhost:8000/docs`

## API Endpoints

| Method   | Path                                    | Purpose                    |
|----------|-----------------------------------------|----------------------------|
| `GET`    | `/health`                               | System health + GPU info   |
| `POST`   | `/api/v1/jobs/create`                   | Upload video + reference   |
| `POST`   | `/api/v1/jobs/{id}/chunks/{idx}`        | Upload individual chunk    |
| `POST`   | `/api/v1/jobs/{id}/reference`           | Add reference images       |
| `GET`    | `/api/v1/jobs/{id}/status`              | Job + per-chunk progress   |
| `GET`    | `/api/v1/jobs/{id}/download`            | Download final video       |
| `GET`    | `/api/v1/jobs/{id}/chunks/{idx}/download` | Download processed chunk |
| `DELETE` | `/api/v1/jobs/{id}`                     | Cancel & cleanup job       |

## ML Pipeline Stack

| Component              | Model                                    | Purpose                        |
|------------------------|------------------------------------------|--------------------------------|
| Face Detection         | InsightFace (buffalo_l)                  | Face bbox, landmarks, embeddings |
| Pose Extraction        | ControlNet OpenPose                      | Skeleton structure preservation  |
| Depth Mapping          | ControlNet Depth (MiDaS)                 | 3D spatial structure             |
| Identity Transfer      | IP-Adapter-FaceID Plus                   | Character face/texture injection |
| Temporal Consistency   | Optical Flow + Gaussian Smoothing        | Anti-flicker across frames       |
| Lip Synchronization    | Wav2Lip                                  | Mouth movement to audio sync     |
| Artifact Removal       | Bilateral Filter + Alpha Blending        | Invisible edge compositing       |
| Base Diffusion         | Stable Diffusion v1.5                    | Frame generation backbone        |

## Hardware Requirements

| Component | Minimum        | Recommended      |
|-----------|----------------|------------------|
| GPU       | NVIDIA 8GB VRAM | NVIDIA 16GB VRAM |
| RAM       | 16GB           | 32GB             |
| Storage   | 50GB           | 100GB            |
| CUDA      | 11.8+          | 12.2+            |

## Environment Variables

All prefixed with `VSWAP_`. See `.env.example` for full list.

## License

MIT
