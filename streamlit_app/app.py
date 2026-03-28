"""Streamlit demo dashboard for Spatial Context Agent — agentic tour guide."""

import base64
import io
import json
import uuid

import httpx
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Spatial Context Agent",
    page_icon="🗺️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------

if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_analysis" not in st.session_state:
    st.session_state.last_analysis = None
if "audio_cache" not in st.session_state:
    st.session_state.audio_cache = {}  # key → MP3 bytes

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("⚙️ Settings")
    api_url = st.text_input("API URL", value="http://localhost:8001")
    api_key = st.text_input("API Key (optional)", value="", type="password")
    st.caption("Leave blank if ENABLE_AUTH=false.")
    st.divider()

    persona = st.selectbox(
        "Tour Guide Persona",
        ["historian", "storyteller", "local", "child_friendly"],
        format_func=lambda x: {
            "historian": "🏛️ Historian",
            "storyteller": "📖 Storyteller",
            "local": "🍺 Local",
            "child_friendly": "🧒 Child Friendly",
        }[x],
    )

    if st.button("🔄 New Session", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.chat_history = []
        st.session_state.last_analysis = None
        st.session_state.audio_cache = {}
        st.rerun()

    if st.session_state.session_id:
        st.caption(f"Session: `{st.session_state.session_id[:8]}…`")

    st.divider()
    st.markdown("**About**")
    st.caption(
        "Multi-agent spatial tour guide: CLIP vision + pgvector RAG + "
        "GPT-4o-mini narration + LangGraph orchestration."
    )

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("🗺️ Spatial Context Agent")
st.subheader("AI Tour Guide — Agentic Edition")

# ---------------------------------------------------------------------------
# Image upload
# ---------------------------------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a photo (JPG or PNG)",
    type=["jpg", "jpeg", "png"],
    help="Images with GPS EXIF will have coordinates auto-extracted.",
)

if uploaded_file:
    raw_bytes = uploaded_file.read()          # keep original bytes (EXIF intact)
    uploaded_file.seek(0)                     # reset for PIL
    image = Image.open(uploaded_file)
    col_img, col_meta = st.columns([1, 1])

    with col_img:
        st.image(ImageOps.exif_transpose(image), caption="Uploaded image", use_column_width=True)

    # Try to extract EXIF GPS
    exif_lat, exif_lng = None, None
    try:
        exif_data = image._getexif() or {}
        gps_info = exif_data.get(0x8825)
        if gps_info:
            def _rat(v):
                return float(v) if not isinstance(v, tuple) else v[0] / v[1]
            lat_d, lat_m, lat_s = _rat(gps_info[2][0]), _rat(gps_info[2][1]), _rat(gps_info[2][2])
            lng_d, lng_m, lng_s = _rat(gps_info[4][0]), _rat(gps_info[4][1]), _rat(gps_info[4][2])
            exif_lat = lat_d + lat_m / 60 + lat_s / 3600
            exif_lng = lng_d + lng_m / 60 + lng_s / 3600
            if gps_info.get(1) == b"S":
                exif_lat = -exif_lat
            if gps_info.get(3) == b"W":
                exif_lng = -exif_lng
    except Exception:
        pass

    with col_meta:
        st.markdown("**GPS Coordinates**")
        if exif_lat and exif_lng:
            st.success(f"EXIF GPS detected: {exif_lat:.5f}, {exif_lng:.5f}")
            st.map(pd.DataFrame({"lat": [exif_lat], "lon": [exif_lng]}), zoom=13)
        else:
            st.warning("No EXIF GPS — enter coordinates below.")

        manual_lat = st.number_input("Latitude", value=exif_lat or 0.0,
                                     format="%.6f", step=0.0001)
        manual_lng = st.number_input("Longitude", value=exif_lng or 0.0,
                                     format="%.6f", step=0.0001)
        use_manual = st.checkbox("Use manual GPS (overrides EXIF)", value=(exif_lat is None))

    # --- Analyze button ---
    st.divider()
    analyze_clicked = st.button("🔍 Analyze with AI Agent", type="primary",
                                use_container_width=True)

    if analyze_clicked:
        b64_image = base64.b64encode(raw_bytes).decode()

        payload: dict = {
            "image": b64_image,
            "persona": persona,
            "session_id": st.session_state.session_id,
        }
        if use_manual or (exif_lat is None):
            payload["latitude"] = manual_lat
            payload["longitude"] = manual_lng

        headers = {"X-API-Key": api_key} if api_key else {}

        # --- Streaming SSE response ---
        steps_box = st.empty()
        narration_box = st.empty()

        step_trace: list[str] = []
        full_narration = ""
        final_data: dict | None = None

        try:
            with httpx.Client(timeout=90) as client:
                with client.stream(
                    "POST",
                    f"{api_url.rstrip('/')}/api/v1/agent/stream",
                    json=payload,
                    headers=headers,
                ) as resp:
                    if resp.status_code != 200:
                        resp.read()
                        st.error(f"API error {resp.status_code}: {resp.text}")
                        st.stop()

                    for line in resp.iter_lines():
                        if not line.startswith("data: "):
                            continue
                        event = json.loads(line[6:])

                        if event["type"] == "step":
                            step_trace.append(event["content"])
                            steps_box.markdown(
                                "**Agent thinking…**\n"
                                + "\n".join(f"- {s}" for s in step_trace)
                            )

                        elif event["type"] == "token":
                            full_narration += event["content"]
                            narration_box.markdown(
                                f"**Narration**\n\n{full_narration}▌"
                            )

                        elif event["type"] == "done":
                            final_data = event
                            narration_box.markdown(
                                f"**Narration**\n\n{full_narration}"
                            )
                            steps_box.empty()

        except Exception as e:
            st.error(f"Could not reach API at {api_url}: {e}")
            st.stop()

        if final_data:
            # Persist session and trigger full results render
            st.session_state.session_id = final_data["session_id"]
            # Merge streamed narration into final_data for results section
            final_data["narration"] = full_narration
            final_data["step_trace"] = step_trace
            st.session_state.last_analysis = final_data
            st.session_state.chat_history = [
                {"role": "assistant", "content": full_narration}
            ]
            st.rerun()

# ---------------------------------------------------------------------------
# Results + Chat (shown after analysis)
# ---------------------------------------------------------------------------

if st.session_state.last_analysis:
    data = st.session_state.last_analysis
    st.success("Analysis complete!")

    res_scene, res_location = st.columns(2)

    with res_scene:
        st.markdown("### 🎨 Scene Classification")
        scene = data["scene"]
        primary = scene.get("primary", {})
        primary_name = primary.get("category", "") if isinstance(primary, dict) else primary
        primary_conf = primary.get("confidence", 0) if isinstance(primary, dict) else 0
        st.metric("Primary scene", primary_name, f"{primary_conf:.1%} confidence")
        alternatives = scene.get("alternatives", [])
        all_cats = [{"category": primary_name, "confidence": primary_conf}] + alternatives
        chart_df = pd.DataFrame(all_cats).sort_values("confidence")
        st.bar_chart(chart_df.set_index("category")["confidence"])

    with res_location:
        st.markdown("### 📍 Nearest Landmark")
        loc = data["location"]
        if loc.get("nearest_landmark"):
            st.metric("Landmark", loc["nearest_landmark"])
            c1, c2 = st.columns(2)
            c1.metric("Distance", f"{loc['distance_meters']:.0f} m")
            c2.metric("District", loc.get("district", "—"))
            meta_coords = data["metadata"]["coordinates"]
            st.map(pd.DataFrame({
                "lat": [meta_coords["latitude"]],
                "lon": [meta_coords["longitude"]],
            }), zoom=14)
        else:
            st.info("No known landmark within 5 km radius.")

    # Agent thinking trace
    with st.expander("🧠 Agent Thinking Steps"):
        for step in data.get("step_trace", []):
            st.markdown(f"- {step}")

    # Metadata
    with st.expander("📊 Metadata & Raw Response"):
        meta = data["metadata"]
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Inference time", f"{meta['inference_time_ms']} ms")
        mc2.metric("Model", meta["model_version"])
        mc3.metric("Persona", meta.get("persona", "—"))
        st.json(data)

    # ---------------------------------------------------------------------------
    # Chat interface
    # ---------------------------------------------------------------------------
    st.divider()
    st.markdown("### 💬 Ask the Tour Guide")

    _INTENT_ICONS = {
        "nearby_places": "📍", "historical_facts": "🏛️",
        "tell_me_more": "📖", "opening_hours": "🕐",
        "directions": "🗺️", "translation": "🌐",
        "photo_tip": "📷", "moved": "🚶",
        "current_events": "🎭", "general": "💬",
    }

    def _speak_button(text: str, cache_key: str) -> None:
        """Render a 🔊 button that fetches TTS audio and caches it."""
        if st.button("🔊", key=f"btn_{cache_key}", help="Listen to this response"):
            with st.spinner("Generating audio…"):
                try:
                    r = httpx.post(
                        f"{api_url.rstrip('/')}/api/v1/agent/speak",
                        json={"text": text, "language": "en"},
                        headers={"X-API-Key": api_key} if api_key else {},
                        timeout=30,
                    )
                    r.raise_for_status()
                    st.session_state.audio_cache[cache_key] = r.content
                except Exception as e:
                    st.error(f"Audio failed: {e}")
        if st.session_state.audio_cache.get(cache_key):
            st.audio(st.session_state.audio_cache[cache_key], format="audio/mp3")

    for i, msg in enumerate(st.session_state.chat_history):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                intent = msg.get("intent")
                if intent:
                    icon = _INTENT_ICONS.get(intent, "💬")
                    st.caption(f"{icon} Intent: `{intent}`")
                if intent == "moved":
                    st.info("Upload a new photo above and click **Analyze** to update your location.")
                _speak_button(msg["content"], f"msg_{i}")

    if question := st.chat_input("Ask a follow-up question about this location…"):
        st.session_state.chat_history.append({"role": "user", "content": question})

        headers = {"X-API-Key": api_key} if api_key else {}
        with st.spinner("Thinking…"):
            try:
                r = httpx.post(
                    f"{api_url.rstrip('/')}/api/v1/agent/followup",
                    json={
                        "session_id": st.session_state.session_id,
                        "question": question,
                        "persona": persona,
                    },
                    headers=headers,
                    timeout=30,
                )
                r.raise_for_status()
                resp_data = r.json()
                answer = resp_data["answer"]
                detected_intent = resp_data.get("intent")
            except Exception as e:
                answer = f"Error: {e}"
                detected_intent = None

        msg_meta = {}
        if detected_intent:
            msg_meta["intent"] = detected_intent
        st.session_state.chat_history.append(
            {"role": "assistant", "content": answer, **msg_meta}
        )
        st.rerun()

else:
    # No analysis yet — show landmark map
    st.info("Upload an image above and click Analyze to start.")
    st.divider()
    st.markdown("### 🏛️ Available Landmarks in Database")
    headers = {"X-API-Key": api_key} if api_key else {}
    try:
        r = httpx.get(f"{api_url.rstrip('/')}/api/v1/locations",
                      headers=headers, params={"limit": 18}, timeout=5)
        if r.status_code == 200:
            items = r.json().get("items", [])
            if items:
                df = pd.DataFrame(items)[["name", "district", "category", "latitude", "longitude"]]
                st.dataframe(df, use_container_width=True, hide_index=True)
                st.map(pd.DataFrame({
                    "lat": [i["latitude"] for i in items],
                    "lon": [i["longitude"] for i in items],
                }), zoom=11)
            else:
                st.caption("Database empty — run `python -m src.db.seed` to populate.")
        else:
            st.caption(f"API returned {r.status_code}. Is the server running?")
    except Exception:
        st.caption(f"API not reachable at {api_url}. Start with `docker-compose up -d db`.")
