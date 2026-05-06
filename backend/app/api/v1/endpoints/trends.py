from fastapi import APIRouter
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime

router = APIRouter()


# --- Pydantic response schemas (inline for simplicity) ---
class TrendResponse(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    score: float
    velocity: float
    keywords: List[str]
    articles_count: int
    sentiment: str
    updated_at: datetime


# Mock data for demonstration
MOCK_TRENDS = [
    TrendResponse(
        id=1,
        name="Large Language Models",
        description="Explosive growth in LLM applications and research.",
        score=98.4,
        velocity=12.5,
        keywords=["AI", "OpenAI", "DeepLearning"],
        articles_count=1540,
        sentiment="Positive",
        updated_at=datetime.utcnow(),
    ),
    TrendResponse(
        id=2,
        name="Sustainable Aviation Fuel",
        description="New green energy solutions for the aviation industry.",
        score=85.2,
        velocity=8.2,
        keywords=["GreenTech", "Aviation", "ESG"],
        articles_count=820,
        sentiment="Positive",
        updated_at=datetime.utcnow(),
    ),
    TrendResponse(
        id=3,
        name="Central Bank Digital Currencies",
        description="Governments exploring digital currency frameworks.",
        score=76.8,
        velocity=-2.1,
        keywords=["Finance", "Crypto", "Policy"],
        articles_count=430,
        sentiment="Neutral",
        updated_at=datetime.utcnow(),
    ),
    TrendResponse(
        id=4,
        name="Autonomous Vehicle Ethics",
        description="Safety concerns reshape self-driving regulations.",
        score=92.1,
        velocity=15.4,
        keywords=["Safety", "Auto", "Ethics"],
        articles_count=670,
        sentiment="Negative",
        updated_at=datetime.utcnow(),
    ),
]


@router.get("/", response_model=List[TrendResponse])
async def get_trends():
    """Get all trending topics."""
    return MOCK_TRENDS


@router.get("/{trend_id}", response_model=TrendResponse)
async def get_trend(trend_id: int):
    """Get details for a specific trend."""
    for t in MOCK_TRENDS:
        if t.id == trend_id:
            return t
    return {"error": "Trend not found"}
