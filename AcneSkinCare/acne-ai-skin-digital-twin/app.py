import streamlit as st
import base64
import io
import os
import uuid
import json
from datetime import datetime
from PIL import Image
import plotly.express as px
import pandas as pd



PRIMARY = "#A8E6CF"
SECONDARY = "#B3E5FC"
ACCENT = "#F8BBD0"
BG = "#FAFAFA"
TEXT = "#263238"
HIGHLIGHT = "#4DB6AC"

STYLE_PATH = os.path.join("src", "components", "style.css")

def load_css():
    try:
        with open(STYLE_PATH, "r", encoding="utf-8") as f:
            css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    except Exception:
        # silent fail if no css
        pass

# import skin passport renderer
from src.components.skin_passport import render_skin_passport

def sidebar_nav():
    st.sidebar.title("Acne AI")
    st.sidebar.markdown("### Navigation")

    options = ["Home", "Upload", "Skin Passport", "Progress"]
    default = st.session_state.get("page", "Home")
    idx = options.index(default) if default in options else 0

    page = st.sidebar.radio(
        "Navigation",
        options,
        index=idx,
        key="page",                   # seçimi state’te tut
        label_visibility="collapsed"  # erişilebilir, ama gizli
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown("Daily tips")
    st.sidebar.info("Use sunscreen • Hydrate • Keep streaks")
    return page

def show_header():
    col1, col2 = st.columns([3,1])
    with col1:
        st.markdown(f"<h1 style='color:{TEXT}'>Acne AI — Skin Digital Twin</h1>", unsafe_allow_html=True)
        st.markdown("<div class='small-muted'>Visual-first prototype — upload selfies, track your skin score over time.</div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='badge center'>Streak: 4 days</div>", unsafe_allow_html=True)

def _encode_image_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def uploader_card():
    st.markdown("<div class='skin-card'>", unsafe_allow_html=True)
    st.subheader("Upload a selfie")
    uploaded = st.file_uploader("Choose a photo (jpg/png)", type=["jpg","jpeg","png"])
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Preview", use_column_width=False, width=320, output_format="PNG", clamp=True)

        # convert to base64 for gateway / backend call
        b64 = _encode_image_to_b64(image)
        st.session_state["last_image_b64"] = b64

        col1, col2 = st.columns([1,1])
        with col1:
            if st.button("Run Analysis"):
                st.session_state.setdefault("analysis_log", [])
                payload = {
                    "request_id": str(uuid.uuid4()),
                    "image_b64": b64,
                    "capture_ts": datetime.utcnow().isoformat() + "Z",
                    "preprocess": {"face_align": True, "illum_norm": True}
                }
                gateway_url = st.secrets.get("GATEWAY_URL", "http://127.0.0.1:8000/infer")
                try:
                    import requests
                except Exception as e:
                    st.error("requests package not installed. Run: pip install requests")
                    st.exception(e)
                    return

                with st.spinner("Running analysis..."):
                    try:
                        resp = requests.post(gateway_url, json=payload, timeout=30)
                        try:
                            body = resp.json()
                            st.json(body)
                        except Exception:
                            st.text(resp.text[:2000])
                        if resp.status_code == 200:
                            result = resp.json()
                            metrics = {
                                "overall_skin_score": result.get("skin_score", 0),
                                "acne_index": result.get("scores", {}).get("acne", 0),
                                "pores": result.get("scores", {}).get("pores", 0),
                                "pigmentation": result.get("scores", {}).get("pigmentation", 0),
                                "hydration": result.get("scores", {}).get("hydration", 0)
                            }
                            st.session_state["last_metrics"] = metrics
                            st.success("Analysis complete — open Skin Passport")
                            st.session_state["analysis_log"].append({"time": datetime.utcnow().isoformat(), "status": "ok", "trace_id": result.get("trace_id")})
                        else:
                            st.error(f"Analysis failed: {resp.status_code}")
                            st.session_state["analysis_log"].append({"time": datetime.utcnow().isoformat(), "status": "error", "code": resp.status_code})
                    except Exception as e:
                        st.error("Error calling gateway (network/exception). See details below.")
                        st.exception(e)
                        st.session_state["analysis_log"].append({"time": datetime.utcnow().isoformat(), "status": "exception", "error": str(e)})

        with col2:
            if st.button("Use placeholder metrics"):
                metrics = {"overall_skin_score": 68, "acne_index": 0.32, "pores": 0.45, "pigmentation": 0.28, "hydration": 0.6}
                st.session_state["last_metrics"] = metrics
                st.success("Placeholder metrics set — open Skin Passport")

    st.markdown("</div>", unsafe_allow_html=True)

def skin_passport_card(metrics=None):
    st.markdown("<div class='skin-card'>", unsafe_allow_html=True)
    st.markdown("<h3>Skin Passport</h3>", unsafe_allow_html=True)
    if not metrics:
        st.write("No analysis available. Upload a selfie first.")
        st.markdown("</div>", unsafe_allow_html=True)
        return
    col1, col2 = st.columns([1.4,1])
    with col1:
        st.markdown(f"<div style='font-size:30px; font-weight:700; color:{PRIMARY}'>{metrics.get('overall_skin_score', '--')}</div>", unsafe_allow_html=True)
        st.markdown("<div class='small-muted'>Overall Skin Score</div>", unsafe_allow_html=True)
        st.write("")
        st.markdown("### Key Metrics")
        st.write(f"- Acne index: {metrics.get('acne_index', 0):.2f}")
        st.write(f"- Pores: {metrics.get('pores', 0):.2f}")
        st.write(f"- Pigmentation: {metrics.get('pigmentation', 0):.2f}")
        st.write(f"- Hydration: {metrics.get('hydration', 0):.2f}")
    with col2:
        df = pd.DataFrame({
            "metric": ["acne", "pores", "pigmentation", "hydration"],
            "value": [
                metrics.get("acne_index", 0),
                metrics.get("pores", 0),
                metrics.get("pigmentation", 0),
                metrics.get("hydration", 0)
            ]
        })
        fig = px.bar(df, x="metric", y="value", range_y=[0,1], color="metric",
                     color_discrete_sequence=[PRIMARY, SECONDARY, ACCENT, HIGHLIGHT])
        fig.update_layout(margin=dict(l=0,r=0,t=0,b=0), showlegend=False, height=220)
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

def progress_chart(timeseries=None):
    st.markdown("<div class='skin-card'>", unsafe_allow_html=True)
    st.write("### Progress")
    if timeseries is None:
        times = pd.date_range(end=datetime.today(), periods=8).tolist()
        scores = [55,58,60,62,63,65,66,68]
        df = pd.DataFrame({"date": times, "score": scores})
    else:
        df = timeseries
    fig = px.line(df, x="date", y="score", markers=True, line_shape="spline",
                  color_discrete_sequence=[HIGHLIGHT])
    fig.update_layout(margin=dict(l=0,r=0,t=10,b=0), height=260, xaxis_title=None, yaxis_title="Skin Score")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="Acne AI — Skin Digital Twin", layout="wide", page_icon="🩺")
    load_css()
    page = sidebar_nav()
    show_header()
    st.write("")  # spacing

    # removed login guard so UI is always accessible
    if page == "Home":
        st.markdown("## Welcome")
        st.write("Focus on calm visuals and simple UX.")
        uploader_card()
    elif page == "Upload":
        uploader_card()
    elif page == "Skin Passport":
        metrics = st.session_state.get("last_metrics")
        last_b64 = st.session_state.get("last_image_b64")
        if metrics is None:
            st.info("No analysis available. Upload a selfie and run analysis first.")
        else:
            render_skin_passport(metrics, before_b64=last_b64, after_b64=last_b64)
    elif page == "Progress":
        progress_chart()

if __name__ == "__main__":
    main()