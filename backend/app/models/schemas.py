from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from backend.app.core.database import Base

class Article(Base):
    __tablename__ = "articles"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    content = Column(String)
    url = Column(String)
    source = Column(String)
    category = Column(String)
    sentiment_score = Column(Float)
    sentiment_label = Column(String)
    published_at = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)

class Trend(Base):
    __tablename__ = "trends"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    description = Column(String)
    score = Column(Float)
    velocity = Column(Float)
    keywords = Column(JSON)  # List of related keywords
    articles_count = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class Event(Base):
    __tablename__ = "events"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    summary = Column(String)
    type = Column(String)  # e.g., "spike", "emerging"
    confidence = Column(Float)
    trend_id = Column(Integer, ForeignKey("trends.id"))
    
    trend = relationship("Trend")
    created_at = Column(DateTime, default=datetime.utcnow)
