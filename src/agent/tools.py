"""LangChain tools that wrap the existing spatial pipeline.

Each tool is decorated with @tool so LangGraph nodes can call them
directly. Docstrings are used by the LLM to decide when to invoke each tool.
"""

import logging

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


@tool
def classify_scene_tool(image_b64: str) -> dict:
    """Classify the scene in a base64-encoded image using CLIP zero-shot vision.

    Returns a dict with 'primary' (category + confidence) and 'alternatives'.
    Use this first whenever an image is provided.
    """
    import base64
    import io

    from PIL import Image

    from src.pipeline.scene_classifier import SceneClassifier

    image_bytes = base64.b64decode(image_b64)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    result = SceneClassifier().classify(image, top_k=5)
    logger.info("classify_scene_tool: primary=%s", result["primary"]["category"])
    return result


@tool
def find_landmark_tool(latitude: float, longitude: float) -> dict:
    """Find the nearest known landmark within 5km of the given GPS coordinates.

    Returns landmark details including name, district, category, distance_meters,
    and narration_template. Returns {'found': False} if nothing is within radius.
    Use this after getting GPS coordinates to identify the location.
    """
    from src.db.session import SessionLocal
    from src.pipeline.context_retriever import ContextRetriever

    db = SessionLocal()
    try:
        landmark = ContextRetriever(db).find_nearest_landmark(latitude, longitude)
        if landmark:
            logger.info("find_landmark_tool: found %r (%.0fm)", landmark["name"],
                        landmark["distance_meters"])
            return {**landmark, "found": True}
        logger.info("find_landmark_tool: no landmark within radius")
        return {"found": False}
    finally:
        db.close()


@tool
def retrieve_knowledge_tool(landmark_name: str, query: str, landmark_id: str) -> str:
    """Retrieve relevant knowledge chunks about a landmark using semantic search.

    Searches the RAG knowledge base (DB descriptions + Wikipedia) for context
    relevant to the query. Use this to ground narrations in factual content.

    Args:
        landmark_name: Name of the landmark (for logging).
        query: Natural language query about the landmark.
        landmark_id: UUID string of the landmark in the database.
    """
    from src.db.session import SessionLocal
    from src.pipeline.rag_retriever import RAGRetriever

    db = SessionLocal()
    try:
        chunks = RAGRetriever(db).retrieve(query=query, landmark_id=landmark_id, top_k=3)
        result = "\n\n".join(chunks) if chunks else "No knowledge chunks found."
        logger.info("retrieve_knowledge_tool: %d chunks for %r", len(chunks), landmark_name)
        return result
    finally:
        db.close()


@tool
def get_nearby_places_tool(latitude: float, longitude: float, category: str = "") -> str:
    """Find other landmarks near the given coordinates, optionally filtered by category.

    Returns a formatted list of nearby places with distances.
    Use this to suggest what else the user can visit nearby.

    Args:
        latitude: User's latitude.
        longitude: User's longitude.
        category: Optional category filter e.g. 'museum', 'memorial', 'park'.
    """
    from src.db.models import Landmark
    from src.db.session import SessionLocal
    from src.pipeline.context_retriever import ContextRetriever

    db = SessionLocal()
    try:
        landmarks = db.query(Landmark).all()
        results = []
        for lm in landmarks:
            dist = ContextRetriever.haversine_distance_m(
                latitude, longitude, lm.latitude, lm.longitude
            )
            if dist < 2000 and (not category or category.lower() in lm.category.lower()):
                results.append((dist, lm.name, lm.district, lm.category))

        if not results:
            return "No nearby places found within 2km."

        results.sort()
        lines = [f"- {name} ({dist:.0f}m, {district}, {cat})"
                 for dist, name, district, cat in results[:5]]
        return "Nearby places:\n" + "\n".join(lines)
    finally:
        db.close()
