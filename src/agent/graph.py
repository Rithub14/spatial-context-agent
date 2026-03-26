"""LangGraph orchestration graph for the spatial tour guide agent.

Flow: vision → context → narration → END
      (context node skips RAG if no landmark found)
"""

import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from src.config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    """Shared state passed between graph nodes."""

    # Inputs
    image_b64: str
    latitude: float
    longitude: float
    persona: str
    session_id: str | None
    conversation_history: list[dict]

    # Intermediate results
    scene_result: dict
    landmark: dict
    knowledge_chunks: str
    nearby_places: str

    # Output
    narration: str
    step_trace: list[str]


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def vision_node(state: AgentState) -> dict[str, Any]:
    """Run CLIP scene classification on the uploaded image."""
    from src.agent.tools import classify_scene_tool

    logger.info("vision_node: classifying scene")
    result = classify_scene_tool.invoke({"image_b64": state["image_b64"]})
    trace = state.get("step_trace", [])
    trace.append(f"👁️ Scene identified: {result['primary']['category']} "
                 f"({result['primary']['confidence']:.0%} confidence)")
    return {"scene_result": result, "step_trace": trace}


def context_node(state: AgentState) -> dict[str, Any]:
    """Find the nearest landmark, reverse geocode via Foursquare, and retrieve RAG chunks."""
    from src.agent.tools import find_landmark_tool, retrieve_knowledge_tool, reverse_geocode_tool

    trace = state.get("step_trace", [])

    # Find nearest landmark in local DB
    landmark = find_landmark_tool.invoke({
        "latitude": state["latitude"],
        "longitude": state["longitude"],
    })

    if landmark.get("found"):
        trace.append(f"📍 Landmark found: {landmark['name']} "
                     f"({landmark['distance_meters']:.0f}m away)")

        # Retrieve RAG knowledge
        scene_type = state["scene_result"]["primary"]["category"]
        knowledge = retrieve_knowledge_tool.invoke({
            "landmark_name": landmark["name"],
            "query": f"{landmark['name']} {scene_type} history significance",
            "landmark_id": landmark["id"],
        })
        trace.append(f"📚 Retrieved knowledge context ({len(knowledge)} chars)")
    else:
        # Fall back to Foursquare reverse geocoding for any location in the world
        trace.append("📍 No local landmark found — querying Foursquare Places API...")
        knowledge = reverse_geocode_tool.invoke({
            "latitude": state["latitude"],
            "longitude": state["longitude"],
        })
        trace.append(f"🌍 Foursquare: {knowledge[:80]}...")

    return {"landmark": landmark, "knowledge_chunks": knowledge, "step_trace": trace}


def narration_node(state: AgentState) -> dict[str, Any]:
    """Generate tour guide narration using GPT-4o-mini via LangChain."""
    from langchain_openai import ChatOpenAI

    trace = state.get("step_trace", [])
    persona = state.get("persona", settings.default_persona)
    landmark = state.get("landmark", {})
    scene_result = state["scene_result"]
    knowledge = state.get("knowledge_chunks", "")

    from src.pipeline.narration_engine import PERSONA_PROMPTS
    system_prompt = PERSONA_PROMPTS.get(persona, PERSONA_PROMPTS["historian"])
    system_prompt += "\n\nKeep your response under 300 characters. Be vivid but concise."

    # Build context block
    context_parts = []
    if landmark.get("found"):
        context_parts.append(
            f"Landmark: {landmark['name']}, {landmark.get('district', '')}, "
            f"{landmark.get('city', 'Berlin')}. "
            f"Category: {landmark.get('category', '')}. "
            f"Historical period: {landmark.get('historical_period', '')}. "
            f"Distance: {landmark.get('distance_meters', 0):.0f}m from user."
        )
    if knowledge:
        context_parts.append(f"Knowledge base context:\n{knowledge[:800]}")

    scene_type = scene_result["primary"]["category"]
    confidence = scene_result["primary"]["confidence"]
    user_content = (
        f"Location: ({state['latitude']:.4f}, {state['longitude']:.4f})\n"
        f"Scene: {scene_type} ({confidence:.0%} confidence)\n"
        + ("\n".join(context_parts) if context_parts else "No landmark data available.")
        + "\n\nGenerate a tour guide narration."
    )

    messages: list[Any] = [SystemMessage(content=system_prompt)]
    for turn in state.get("conversation_history", [])[-4:]:
        if turn["role"] == "user":
            messages.append(HumanMessage(content=turn["content"]))
        else:
            messages.append(AIMessage(content=turn["content"]))
    messages.append(HumanMessage(content=user_content))

    if settings.openai_api_key:
        llm = ChatOpenAI(
            model=settings.gpt_model,
            api_key=settings.openai_api_key,
            max_tokens=300,
        )
        response = llm.invoke(messages)
        narration = response.content
        trace.append(f"🎙️ Narration generated by {settings.gpt_model} ({persona} persona)")
    else:
        # Fallback: use template engine
        from src.pipeline.narration_engine import NarrationEngine
        lm = landmark if landmark.get("found") else None
        narration = NarrationEngine()._template_narration(
            scene_result, lm, (state["latitude"], state["longitude"]), 300
        )
        trace.append("🎙️ Narration generated via template (no OpenAI key)")

    return {"narration": narration, "step_trace": trace}


def _should_fetch_knowledge(state: AgentState) -> str:
    """Route: skip knowledge retrieval if no landmark was found."""
    return "context" if True else "narration"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph():
    """Build and compile the LangGraph StateGraph.

    Returns:
        Compiled LangGraph graph ready for .invoke() or .stream().
    """
    builder = StateGraph(AgentState)

    builder.add_node("vision", vision_node)
    builder.add_node("context", context_node)
    builder.add_node("narration", narration_node)

    builder.add_edge(START, "vision")
    builder.add_edge("vision", "context")
    builder.add_edge("context", "narration")
    builder.add_edge("narration", END)

    return builder.compile()


# Compiled graph singleton — built once, reused per request
_graph = None


def get_graph():
    """Return the compiled graph singleton."""
    global _graph
    if _graph is None:
        _graph = build_graph()
        logger.info("LangGraph compiled.")
    return _graph
