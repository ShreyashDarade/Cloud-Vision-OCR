# Cloud Vision OCR Pipeline

Production-grade OCR pipeline powered by **Google Cloud Vision API** with advanced preprocessing, caching, and multiple interfaces.

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## ✨ Features

- 🔍 **Google Cloud Vision API** - Accurate text extraction with DOCUMENT_TEXT_DETECTION
- 🖼️ **Advanced Preprocessing** - Deskewing, denoising, contrast enhancement, auto-crop
- ⚡ **High Performance** - Redis caching (optional), content-hash deduplication
- 🔄 **Batch Processing** - Process multiple images concurrently
- 📊 **Multiple Output Formats** - JSON, plain text, hOCR
- 🌐 **REST API** - FastAPI with OpenAPI docs
- 🎨 **Streamlit UI** - Beautiful web interface
- 🖥️ **CLI** - Command-line interface with rich output
- 🐳 **Docker Ready** - Production deployment with Docker Compose
- 📈 **Monitoring** - Prometheus metrics and structured logging

## 🚀 Quick Start

### Prerequisites

1. **Python 3.11+**
2. **Google Cloud Account** with Vision API enabled
3. **Service Account Key** (JSON file)

### Installation

```bash
# Clone the repository
cd OCR

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Configure Credentials

```bash
# Copy example configuration
copy .env.example .env

# Edit .env and populate GOOGLE_CREDENTIALS_JSON with your service account key content
# The content should be a single-line JSON string like '{"type": "service_account", ...}'
```

### Start Services

Run both the API server and Frontend with a single command:

```bash
python run.py
```

Or start individually:

```bash
python run.py api       # API on port 8000
python run.py frontend  # Frontend on port 8501
```

**Access:**

- **API Docs**: http://localhost:8000/docs
- **Streamlit UI**: http://localhost:8501
- **Metrics**: http://localhost:8000/metrics

## 📖 Usage

### REST API

```bash
# Sync OCR (single image)
curl -X POST "http://localhost:8000/ocr/sync" \
  -F "file=@document.png" \
  -F "detection_type=DOCUMENT_TEXT_DETECTION"

# Async OCR (submit job)
curl -X POST "http://localhost:8000/ocr/async" \
  -F "file=@document.png"

# Check job status
curl "http://localhost:8000/ocr/jobs/{job_id}"
```

### CLI

```bash
# Process single image
python -m src.cli.main process image.png --output result.txt

# Batch processing
python -m src.cli.main batch ./images/ --output-dir ./results/ --workers 4
```

### Python SDK

```python
from src.vision import VisionClient
from src.preprocessing import PreprocessingPipeline

# Preprocess image
pipeline = PreprocessingPipeline()
with open("document.png", "rb") as f:
    processed = pipeline.process_bytes(f.read())

# Extract text
with VisionClient() as client:
    result = client.detect_text(processed)
    print(result.text)
    print(f"Confidence: {result.confidence:.2%}")
```

## 🐳 Docker Deployment

```bash
# Build and run all services
docker-compose up -d

# View logs
docker-compose logs -f api

# Stop services
docker-compose down
```

## ⚙️ Configuration

| Variable                  | Default | Description                     |
| ------------------------- | ------- | ------------------------------- |
| `GOOGLE_CREDENTIALS_JSON` | -       | Raw JSON of service account key |
| `API_PORT`                | 8000    | API server port                 |
| `CACHE_ENABLED`           | true    | Enable result caching           |
| `PREPROCESSING_ENABLED`   | true    | Enable preprocessing            |
| `LOG_LEVEL`               | INFO    | Logging level                   |

## 🧪 Testing

```bash
# Run tests
pytest tests/ -v
```

## 📄 License

MIT License - see LICENSE file for details.
