from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1.endpoints import trends, events, predictions

app = FastAPI(
    title="TrendPulse API",
    description="Production-grade AI Trend & Event Detection API",
    version="1.0.0",
)

# CORS setup for Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Welcome to TrendPulse API", "status": "operational"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


# Include routers
app.include_router(trends.router, prefix="/api/v1/trends", tags=["trends"])
app.include_router(events.router, prefix="/api/v1/events", tags=["events"])
app.include_router(predictions.router, prefix="/api/v1/predictions", tags=["predictions"])
