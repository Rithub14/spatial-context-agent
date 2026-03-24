"""FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.middleware.auth import APIKeyMiddleware
from src.api.middleware.rate_limiter import RateLimitMiddleware
from src.api.routes import agent, health
from src.db.session import create_tables
from src.pipeline.scene_classifier import SceneClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: create DB tables and warm up the CLIP model."""
    logger.info("Starting Spatial Context Agent...")
    create_tables()
    logger.info("Pre-loading CLIP model...")
    SceneClassifier()   # initialises singleton
    logger.info("Ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Spatial Context Agent",
    description="Location-aware AI backend for spatial tour guide experiences.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(APIKeyMiddleware)

app.include_router(health.router)
app.include_router(agent.router)
