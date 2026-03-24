"""Streamlit demo dashboard for Spatial Context Agent."""

import base64
import io

import httpx
import pandas as pd
import streamlit as st
from PIL import Image

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Spatial Context Agent",
    page_icon="🗺️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Sidebar — connection settings
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("⚙️ Settings")
    api_url = st.text_input("API URL", value="http://localhost:8000")
    api_key = st.text_input("API Key (optional)", value="", type="password")
    st.caption("Leave API key blank if ENABLE_AUTH=false.")
    st.divider()
    st.markdown("**About**")
    st.caption(
        "Spatial Context Agent combines CLIP zero-shot vision with "
        "geospatial context to classify scenes and narrate landmarks."
    )

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("🗺️ Spatial Context Agent")
st.subheader("AI Tour Guide Demo")
st.markdown(
    "Upload a photo from Berlin, optionally provide GPS coordinates, "
    "and get a scene classification + landmark narration."
)

# ---------------------------------------------------------------------------
# Image upload
# ---------------------------------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload an image (JPG or PNG)",
    type=["jpg", "jpeg", "png"],
    help="Images with GPS EXIF metadata will have coordinates auto-extracted.",
)

if uploaded_file:
    image = Image.open(uploaded_file)

    col_img, col_meta = st.columns([1, 1])

    with col_img:
        st.image(image, caption="Uploaded image", use_column_width=True)

    # --- Try to extract EXIF GPS for display ---
    exif_lat, exif_lng = None, None
    try:
        exif_data = image._getexif() or {}
        gps_info = exif_data.get(0x8825)
        if gps_info:
            def _rat(v):
                return float(v) if not isinstance(v, tuple) else v[0] / v[1]

            lat_d = _rat(gps_info[2][0])
            lat_m = _rat(gps_info[2][1])
            lat_s = _rat(gps_info[2][2])
            lng_d = _rat(gps_info[4][0])
            lng_m = _rat(gps_info[4][1])
            lng_s = _rat(gps_info[4][2])
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

        st.markdown("**Override / enter GPS manually**")
        manual_lat = st.number_input(
            "Latitude",
            value=exif_lat if exif_lat else 52.5163,
            format="%.6f",
            step=0.0001,
        )
        manual_lng = st.number_input(
            "Longitude",
            value=exif_lng if exif_lng else 13.3777,
            format="%.6f",
            step=0.0001,
        )
        use_manual = st.checkbox(
            "Use manual GPS (overrides EXIF)",
            value=(exif_lat is None),
        )

    # --- Analyze button ---
    st.divider()
    analyze_clicked = st.button("🔍 Analyze", type="primary")

    if analyze_clicked:
        # Encode image to base64
        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        b64_image = base64.b64encode(buf.getvalue()).decode()

        payload: dict = {"image": b64_image}
        if use_manual or (exif_lat is None):
            payload["latitude"] = manual_lat
            payload["longitude"] = manual_lng

        headers = {"X-API-Key": api_key} if api_key else {}

        with st.spinner("Analyzing…"):
            try:
                response = httpx.post(
                    f"{api_url.rstrip('/')}/api/v1/analyze",
                    json=payload,
                    headers=headers,
                    timeout=60,
                )
                response.raise_for_status()
                data = response.json()
            except httpx.HTTPStatusError as e:
                st.error(f"API error {e.response.status_code}: {e.response.text}")
                st.stop()
            except Exception as e:
                st.error(f"Could not reach API at {api_url}: {e}")
                st.stop()

        # ── Results ───────────────────────────────────────────────────────
        st.success("Analysis complete!")

        res_scene, res_location = st.columns(2)

        # Scene classification
        with res_scene:
            st.markdown("### 🎨 Scene Classification")
            scene = data["scene"]
            st.metric("Primary scene", scene["primary"], f"{scene['confidence']:.1%} confidence")

            all_categories = [
                {"category": scene["primary"], "confidence": scene["confidence"]},
                *scene.get("alternatives", []),
            ]
            chart_df = pd.DataFrame(all_categories).sort_values("confidence")
            st.bar_chart(chart_df.set_index("category")["confidence"])

        # Landmark info
        with res_location:
            st.markdown("### 📍 Nearest Landmark")
            loc = data["location"]
            if loc.get("nearest_landmark"):
                st.metric("Landmark", loc["nearest_landmark"])
                col1, col2 = st.columns(2)
                col1.metric("Distance", f"{loc['distance_meters']:.0f} m")
                col2.metric("District", loc.get("district", "—"))
                st.caption(f"City: {loc.get('city', '—')}")

                # Map with user location + landmark
                meta_coords = data["metadata"]["coordinates"]
                map_df = pd.DataFrame({
                    "lat": [meta_coords["latitude"]],
                    "lon": [meta_coords["longitude"]],
                })
                st.map(map_df, zoom=14)
            else:
                st.info("No known landmark within 5 km radius.")

        # Narration
        st.markdown("### 🎙️ Tour Guide Narration")
        st.info(data["narration"])

        # Metadata
        with st.expander("📊 Metadata & Raw Response"):
            meta = data["metadata"]
            mcol1, mcol2, mcol3 = st.columns(3)
            mcol1.metric("Inference time", f"{meta['inference_time_ms']} ms")
            mcol2.metric("Model", meta["model_version"])
            mcol3.metric("Timestamp", meta["timestamp"])
            st.json(data)

else:
    st.info("Upload an image above to get started.")

    # Show a sample of available landmarks
    st.divider()
    st.markdown("### 🏛️ Available Landmarks in Database")
    headers = {"X-API-Key": api_key} if api_key else {}
    try:
        r = httpx.get(
            f"{api_url.rstrip('/')}/api/v1/locations",
            headers=headers,
            params={"limit": 18},
            timeout=5,
        )
        if r.status_code == 200:
            items = r.json().get("items", [])
            if items:
                df = pd.DataFrame(items)[["name", "district", "category", "latitude", "longitude"]]
                st.dataframe(df)
                st.map(
                    pd.DataFrame({"lat": [i["latitude"] for i in items],
                                  "lon": [i["longitude"] for i in items]}),
                    zoom=11,
                )
            else:
                st.caption("Database is empty — run `python -m src.db.seed` to populate it.")
        else:
            st.caption(f"API returned {r.status_code}. Is the server running?")
    except Exception:
        st.caption(f"API not reachable at {api_url}. Start it with `docker-compose up -d`.")
