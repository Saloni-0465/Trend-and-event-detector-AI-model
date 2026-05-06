from fastapi import APIRouter
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime

router = APIRouter()


class PredictedTopic(BaseModel):
    id: int
    topic: str
    predicted_score: float       # 0-100 likelihood of trending
    velocity_forecast: float     # predicted velocity change
    current_mentions: int
    predicted_mentions: int
    confidence: float            # model confidence 0-1
    horizon: str                 # e.g. "24h", "48h", "7d"
    drivers: List[str]           # keywords driving the prediction
    category: str
    direction: str               # "rising", "breakout", "cooling"


MOCK_PREDICTIONS = [
    PredictedTopic(
        id=1,
        topic="Quantum Error Correction",
        predicted_score=94.2,
        velocity_forecast=28.5,
        current_mentions=120,
        predicted_mentions=890,
        confidence=0.91,
        horizon="48h",
        drivers=["Google Willow", "logical qubit", "error rate"],
        category="Technology",
        direction="breakout",
    ),
    PredictedTopic(
        id=2,
        topic="US-China Semiconductor Tariffs",
        predicted_score=88.7,
        velocity_forecast=19.3,
        current_mentions=340,
        predicted_mentions=1200,
        confidence=0.87,
        horizon="24h",
        drivers=["CHIPS Act", "export controls", "TSMC"],
        category="Geopolitics",
        direction="rising",
    ),
    PredictedTopic(
        id=3,
        topic="Weight Loss Drug Patent Wars",
        predicted_score=82.1,
        velocity_forecast=14.8,
        current_mentions=210,
        predicted_mentions=720,
        confidence=0.83,
        horizon="48h",
        drivers=["Ozempic", "GLP-1", "FDA approval"],
        category="Health",
        direction="rising",
    ),
    PredictedTopic(
        id=4,
        topic="Arctic Shipping Route Expansion",
        predicted_score=71.5,
        velocity_forecast=22.1,
        current_mentions=45,
        predicted_mentions=380,
        confidence=0.76,
        horizon="7d",
        drivers=["ice melt", "Northern Sea Route", "cargo volume"],
        category="Climate",
        direction="breakout",
    ),
    PredictedTopic(
        id=5,
        topic="Creator Economy Regulation",
        predicted_score=67.3,
        velocity_forecast=9.4,
        current_mentions=180,
        predicted_mentions=410,
        confidence=0.72,
        horizon="7d",
        drivers=["influencer tax", "FTC guidelines", "disclosure"],
        category="Business",
        direction="rising",
    ),
    PredictedTopic(
        id=6,
        topic="Satellite Internet Coverage Gap",
        predicted_score=58.9,
        velocity_forecast=-4.2,
        current_mentions=290,
        predicted_mentions=210,
        confidence=0.68,
        horizon="48h",
        drivers=["Starlink", "rural broadband", "spectrum"],
        category="Technology",
        direction="cooling",
    ),
]


@router.get("/", response_model=List[PredictedTopic])
async def get_predictions():
    """Get AI-predicted future trending topics with confidence scores."""
    return MOCK_PREDICTIONS
