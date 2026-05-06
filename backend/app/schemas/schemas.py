from pydantic import BaseModel, HttpUrl
from typing import List, Optional
from datetime import datetime

class ArticleBase(BaseModel):
    title: str
    content: str
    url: Optional[HttpUrl] = None
    source: Optional[str] = None
    category: Optional[str] = None

class ArticleCreate(ArticleBase):
    pass

class Article(ArticleBase):
    id: int
    sentiment_score: float
    sentiment_label: str
    published_at: datetime

    class Config:
        from_attributes = True

class TrendBase(BaseModel):
    name: str
    description: Optional[str] = None
    keywords: List[str]

class Trend(TrendBase):
    id: int
    score: float
    velocity: float
    articles_count: int
    updated_at: datetime

    class Config:
        from_attributes = True

class EventBase(BaseModel):
    title: str
    summary: str
    type: str

class Event(EventBase):
    id: int
    confidence: float
    trend_id: int
    created_at: datetime

    class Config:
        from_attributes = True
