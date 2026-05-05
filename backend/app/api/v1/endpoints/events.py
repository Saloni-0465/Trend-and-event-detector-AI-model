from fastapi import APIRouter
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime

router = APIRouter()


class EventResponse(BaseModel):
    id: int
    title: str
    summary: str
    type: str  # "spike", "emerging", "declining"
    confidence: float
    trend_id: int
    created_at: datetime


MOCK_EVENTS = [
    EventResponse(
        id=1,
        title="LLM Efficiency Breakthrough",
        summary="New architecture reduces inference cost by 90%, triggering a massive spike in AI research headlines.",
        type="spike",
        confidence=0.95,
        trend_id=1,
        created_at=datetime.utcnow(),
    ),
    EventResponse(
        id=2,
        title="EU Green Aviation Mandate",
        summary="European Union announces mandatory sustainable fuel quotas for all commercial flights by 2030.",
        type="emerging",
        confidence=0.82,
        trend_id=2,
        created_at=datetime.utcnow(),
    ),
    EventResponse(
        id=3,
        title="Digital Yuan Pilot Expansion",
        summary="China expands CBDC pilot program to 10 additional provinces, covering 300M users.",
        type="emerging",
        confidence=0.78,
        trend_id=3,
        created_at=datetime.utcnow(),
    ),
]


@router.get("/", response_model=List[EventResponse])
async def get_events():
    """Get recent significant events."""
    return MOCK_EVENTS
