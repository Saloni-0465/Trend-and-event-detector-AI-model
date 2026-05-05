# TrendPulse: AI-Powered Trend & Event Detection SaaS

TrendPulse is a production-grade platform that detects trending topics and events in real-time using a hybrid AI approach (Symbolic ML + Neural Deep Learning).

## 🚀 Features
- **Real-time Dashboard**: Modern, glassmorphic UI with smooth animations.
- **AI Processing**: Sentiment analysis (RoBERTa), topic clustering (BERTopic), and keyword extraction (KeyBERT).
- **Hybrid Scoring**: A weighted fusion of lexical, probabilistic, and semantic signals.
- **Scalable Architecture**: FastAPI backend with background Celery workers and PostgreSQL.

## 🛠️ Tech Stack
- **Frontend**: Next.js 14, TypeScript, Tailwind CSS, Framer Motion, Recharts.
- **Backend**: FastAPI, SQLAlchemy, Pydantic, Celery, Redis.
- **AI**: HuggingFace Transformers, BERTopic, SentenceTransformers.
- **Infrastructure**: Docker, Docker Compose.

## 🚦 Quick Start

### 1. Clone and Setup
```bash
# Clone the repository
git clone <repo-url>
cd Trend-and-event-detector-AI-model
```

### 2. Run with Docker
The easiest way to run the entire stack is using Docker Compose:
```bash
docker-compose up --build
```
- **Frontend**: `http://localhost:3000`
- **Backend API**: `http://localhost:8000`
- **API Docs**: `http://localhost:8000/docs`

### 3. Local Development (Optional)

#### Backend
```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

#### Frontend
```bash
cd frontend
npm install
npm run dev
```

## 📂 Project Structure
- `/backend`: FastAPI application and AI services.
- `/frontend`: Next.js dashboard and components.
- `/research`: Original research scripts and documentation.
- `docker-compose.yml`: Full stack orchestration.

## 🧪 Testing
```bash
# Backend tests
cd backend
pytest

# Frontend linting
cd frontend
npm run lint
```
