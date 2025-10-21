import base64
import io
import json
from typing import Dict, Optional
from PIL import Image
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go

_PLACEHOLDER_PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII="

def _b64_or_placeholder(b64: Optional[str]) -> str:
    return b64 if b64 and isinstance(b64, str) else _PLACEHOLDER_PNG

def render_before_after(before_b64: Optional[str], after_b64: Optional[str], height:int=360):
    before = _b64_or_placeholder(before_b64)
    after = _b64_or_placeholder(after_b64)

    # Use str.format and double braces for CSS blocks to avoid f-string brace issues
    html = """
    <style>
    .ba-wrapper{{position:relative; width:100%; max-width:900px; margin:0 auto; user-select:none;}}
    .ba-img{{display:block; width:100%; height:{height}px; object-fit:cover; border-radius:12px; box-shadow:0 8px 30px rgba(0,0,0,0.08);}}
    .ba-overlay{{position:absolute; top:0; left:0; height:{height}px; overflow:hidden; border-radius:12px;}}
    .ba-handle{{position:absolute; left:50%; top:50%; transform:translate(-50%,-50%); z-index:5;
                 width:48px; height:48px; border-radius:24px; background:#fff; box-shadow:0 6px 18px rgba(0,0,0,0.12);
                 display:flex;align-items:center;justify-content:center; font-weight:700; color:#4DB6AC}}
    .ba-range{{width:100%; margin-top:8px}}
    </style>
    <div class="ba-wrapper">
      <img src="data:image/png;base64,{before}" class="ba-img"/>
      <div class="ba-overlay" id="overlay" style="width:50%">
        <img src="data:image/png;base64,{after}" class="ba-img"/>
      </div>
      <div class="ba-handle" id="handle">◀▶</div>
    </div>
    <input id="range" type="range" min="0" max="100" value="50" class="ba-range">
    <script>
      const overlay = document.getElementById('overlay');
      const handle = document.getElementById('handle');
      const range = document.getElementById('range');
      function update(val){{ overlay.style.width = val + '%'; handle.style.left = val + '%'; }}
      range.addEventListener('input', (e)=> update(e.target.value));
      update(50);
    </script>
    """.format(before=before, after=after, height=height)

    components.html(html, height=height+70, scrolling=False)

def render_radar_chart(metrics: Dict[str, float], title="Skin Metrics"):
    labels = list(metrics.keys())
    values = [float(metrics.get(k, 0.0)) for k in labels]
    # close the loop for radar
    vals = values + [values[0]] if labels else [0,0]
    thetas = labels + [labels[0]] if labels else ["",""]
    fig = go.Figure(
        data=go.Scatterpolar(r=vals, theta=thetas, fill='toself',
                             marker=dict(color="#4DB6AC"))
    )
    fig.update_layout(polar=dict(radialaxis=dict(range=[0,1])), showlegend=False, title=title, margin=dict(t=40,b=10,l=10,r=10), height=300)
    st.plotly_chart(fig, use_container_width=True)

def render_badges(score:int):
    if score >= 85:
        badge = ("Platinum", "#4DB6AC")
    elif score >= 70:
        badge = ("Gold", "#A8E6CF")
    elif score >= 50:
        badge = ("Silver", "#B3E5FC")
    else:
        badge = ("Bronze", "#F8BBD0")
    st.markdown(f"<div style='display:inline-block;padding:8px 12px;border-radius:999px;background:{badge[1]};color:#fff;font-weight:700'>{badge[0]}</div>", unsafe_allow_html=True)

def _pil_from_b64(b64: str) -> Image.Image:
    data = base64.b64decode(_b64_or_placeholder(b64))
    return Image.open(io.BytesIO(data)).convert("RGB")

def export_passport_pdf(metrics: Dict, before_b64: Optional[str], after_b64: Optional[str], filename="skin_passport.pdf"):
    """
    Try to create a simple PDF using reportlab. If reportlab not installed, return None.
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
        from reportlab.lib.utils import ImageReader
    except Exception:
        return None

    buf = io.BytesIO()
    try:
        c = canvas.Canvas(buf, pagesize=A4)
        w, h = A4
        left = 40
        y = h - 60
        c.setFont("Helvetica-Bold", 20)
        c.drawString(left, y, "Skin Passport")
        c.setFont("Helvetica", 11)
        y -= 28
        # Scores table
        for k, v in metrics.items():
            try:
                c.drawString(left, y, f"{k}: {v}")
            except Exception:
                c.drawString(left, y, f"{k}: {str(v)}")
            y -= 16
            if y < 120:
                c.showPage()
                y = h - 60
        # Add small images
        try:
            if before_b64:
                before_data = base64.b64decode(_b64_or_placeholder(before_b64))
                before_img = ImageReader(io.BytesIO(before_data))
                c.drawImage(before_img, w - 260, h - 260, width=220, height=220)
            if after_b64:
                after_data = base64.b64decode(_b64_or_placeholder(after_b64))
                after_img = ImageReader(io.BytesIO(after_data))
                c.drawImage(after_img, w - 260, h - 500, width=220, height=220)
        except Exception:
            pass
        c.save()
        buf.seek(0)
        return buf
    except Exception:
        try:
            buf.close()
        except Exception:
            pass
        return None

def render_skin_passport(metrics: Dict[str, float], before_b64: Optional[str]=None, after_b64: Optional[str]=None):
    st.header("Skin Passport")
    col1, col2 = st.columns([1.4,1])
    with col1:
        st.markdown("<div style='background:#fff;border-radius:12px;padding:18px;box-shadow:0 8px 30px rgba(0,0,0,0.04)'>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:36px;font-weight:800;color:#263238'>{int(metrics.get('overall_skin_score',0))}</div>", unsafe_allow_html=True)
        st.markdown("<div style='color:#62727b'>Overall Skin Score</div>", unsafe_allow_html=True)
        render_badges(int(metrics.get('overall_skin_score',0)))
        st.markdown("<hr/>", unsafe_allow_html=True)
        st.markdown("### Key indicators", unsafe_allow_html=True)
        st.write(f"- Acne index: {metrics.get('acne_index', 0):.2f}")
        st.write(f"- Pores: {metrics.get('pores', 0):.2f}")
        st.write(f"- Pigmentation: {metrics.get('pigmentation', 0):.2f}")
        st.write(f"- Hydration: {metrics.get('hydration', 0):.2f}")
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        render_radar_chart({
            "acne": metrics.get("acne_index", 0),
            "pores": metrics.get("pores", 0),
            "pigmentation": metrics.get("pigmentation", 0),
            "hydration": metrics.get("hydration", 0),
        }, title="Metric Radar")
        st.markdown("### Visual Comparison")
        render_before_after(before_b64, after_b64, height=260)
    # export button
    st.markdown("---")
    col3, col4 = st.columns([1,1])
    with col3:
        buf = export_passport_pdf(metrics, before_b64, after_b64)
        if buf:
            st.download_button("Download Passport (PDF)", data=buf.getvalue(), file_name="skin_passport.pdf", mime="application/pdf")
        else:
            # fallback: download JSON
            st.download_button("Download Passport (JSON)", data=json.dumps(metrics, indent=2), file_name="skin_passport.json", mime="application/json")
    with col4:
        st.markdown("### Personalized Tips")
        # simple rule-based tips
        tips = []
        if metrics.get("acne_index",0) > 0.6:
            tips.append("Consider a gentle BPO or salicylic acid treatment.")
        if metrics.get("hydration",0) < 0.4:
            tips.append("Increase daily hydration; use humectants.")
        if metrics.get("pigmentation",0) > 0.5:
            tips.append("Add vitamin C and sunscreen to reduce pigmentation.")
        if not tips:
            tips = ["Maintain current routine. Use SPF daily."]
        for t in tips:
            st.write(f"- {t}")