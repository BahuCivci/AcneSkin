import streamlit as st
import base64
import io
import os
import uuid
import json
from datetime import datetime, timedelta, timezone
import pytz
from PIL import Image, ImageDraw, ImageFont, ImageStat, ImageFilter
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from fpdf import FPDF
import tempfile
import cv2
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import warnings
warnings.filterwarnings('ignore')



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
    st.sidebar.markdown("""
    <div style='text-align: center; padding: 1rem;'>
        <h2 style='color: #6366f1; margin: 0;'>🔬 AcneAI Pro</h2>
        <p style='color: #64748b; margin: 0.5rem 0;'>AI-Powered Skin Analysis</p>
        <small style='color: #94a3b8;'>Advanced Analytics & Predictions</small>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("### 🧭 Navigation")
    options = ["🏠 Home", "📤 Upload & Analyze", "📊 Skin Report", "📈 Progress", "🎯 Analytics"]
    icons = ["🏠", "📤", "📊", "📈", "🎯"]
    pages = ["Home", "Upload", "Skin Passport", "Progress", "Analytics"]

    default = st.session_state.get("page", "Home")
    try:
        idx = pages.index(default)
    except ValueError:
        idx = 0

    selected = st.sidebar.radio(
        "Navigation",
        options,
        index=idx,
        key="nav_radio",
        label_visibility="collapsed"
    )

    # Map selection back to page names
    page = pages[options.index(selected)]
    st.session_state["page"] = page

    st.sidebar.markdown("---")

    # Quick stats sidebar
    history = st.session_state.get("analysis_history", [])
    if history:
        st.sidebar.markdown("### 📊 Quick Stats")
        latest_score = history[-1]['skin_score']
        total_analyses = len(history)

        st.sidebar.metric(
            label="Latest Score",
            value=f"{latest_score:.1f}",
            delta=f"{latest_score - history[-2]['skin_score']:.1f}" if len(history) >= 2 else None
        )

        st.sidebar.metric(
            label="Total Analyses",
            value=total_analyses
        )

        # Health grade
        current_result = st.session_state.get("last_result")
        health_score = calculate_skin_health_score(history, current_result)
        grade_color = {
            "A": "🟢", "B": "🔵", "C": "🟡", "D": "🟠", "F": "🔴"
        }.get(health_score["grade"], "⚪")

        st.sidebar.markdown(f"""
        **Health Grade:** {grade_color} **{health_score["grade"]}**
        """)

    st.sidebar.markdown("---")

    # Daily tips section
    st.sidebar.markdown("### 💡 Daily Tips")
    tips = [
        "☀️ Always use SPF 30+ sunscreen",
        "💧 Stay hydrated (8+ glasses daily)",
        "🧼 Gentle cleansing twice daily",
        "😴 Get 7-8 hours of quality sleep",
        "🥗 Eat antioxidant-rich foods"
    ]

    import random
    daily_tip = random.choice(tips)
    st.sidebar.info(daily_tip)

    # Model info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🤖 Model Info")
    st.sidebar.markdown("""
    **YOLO Acne Detection**
    - Comedones, Papules
    - Pustules, Nodules
    - Real-time analysis
    """)

    return page

def show_header():
    st.markdown("""
    <div class='main-header'>
        <h1 class='text-gradient'>🔬 AcneAI — Skin Digital Twin</h1>
        <div class='subtitle'>Professional acne detection using advanced YOLO AI technology</div>
    </div>
    """, unsafe_allow_html=True)

def _encode_image_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def draw_detections_on_image(image: Image.Image, detections: list) -> Image.Image:
    """Draw bounding boxes and labels on image for acne detections"""
    if not detections:
        return image

    # Create a copy to avoid modifying original
    img_with_boxes = image.copy()
    draw = ImageDraw.Draw(img_with_boxes)

    # Colors for different acne types
    colors = {
        'comedone': '#FF6B6B',      # Red
        'papules': '#4ECDC4',       # Teal
        'pustules': '#45B7D1',      # Blue
        'nodules': '#96CEB4',       # Green
        'blackhead': '#FF6B6B',     # Red
        'whitehead': '#FFA07A',     # Light salmon
        'cyst': '#8B5CF6',          # Purple
        'unknown': '#6C757D'        # Gray
    }

    # Try to load a font, fallback to default
    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except:
        font = ImageFont.load_default()

    for detection in detections:
        # Get bounding box coordinates (x, y, w, h)
        x = detection.get('x', 0)
        y = detection.get('y', 0)
        w = detection.get('w', 0)
        h = detection.get('h', 0)

        # Calculate box coordinates
        x1, y1 = x, y
        x2, y2 = x + w, y + h

        # Get label and confidence
        label = detection.get('label', 'unknown')
        confidence = detection.get('confidence', 0.0)

        # Choose color
        color = colors.get(label.lower(), colors['unknown'])

        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        # Draw label with confidence
        text = f"{label}: {confidence:.2f}"

        # Get text bounding box for background
        try:
            bbox = draw.textbbox((x1, y1-20), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        except:
            # Fallback for older PIL versions
            text_width, text_height = draw.textsize(text, font=font)

        # Draw background rectangle for text
        draw.rectangle([x1, y1-text_height-4, x1+text_width+4, y1], fill=color)

        # Draw text
        draw.text((x1+2, y1-text_height-2), text, fill='white', font=font)

    return img_with_boxes

def create_pdf_report(history, latest_result=None):
    """Create PDF report from analysis history"""
    class PDFReport(FPDF):
        def header(self):
            self.set_font('Arial', 'B', 15)
            self.cell(0, 10, 'AcneAI - Skin Analysis Report', 0, 1, 'C')
            self.cell(0, 10, f'Generated on {datetime.now().strftime("%Y-%m-%d %H:%M")}', 0, 1, 'C')
            self.ln(10)

        def footer(self):
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    pdf = PDFReport()
    pdf.add_page()

    # Summary statistics
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Summary Statistics', 0, 1)
    pdf.set_font('Arial', '', 12)

    if history:
        total_analyses = len(history)
        avg_skin_score = sum(h['skin_score'] for h in history) / len(history)
        latest_score = history[-1]['skin_score']
        total_detections = sum(h['total_detections'] for h in history)

        pdf.cell(0, 8, f'Total Analyses: {total_analyses}', 0, 1)
        pdf.cell(0, 8, f'Average Skin Score: {avg_skin_score:.1f}', 0, 1)
        pdf.cell(0, 8, f'Latest Skin Score: {latest_score:.1f}', 0, 1)
        pdf.cell(0, 8, f'Total Detections: {total_detections}', 0, 1)
        pdf.ln(10)

        # Analysis history
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, 'Analysis History', 0, 1)
        pdf.set_font('Arial', '', 10)

        # Table header
        pdf.cell(50, 8, 'Date', 1)
        pdf.cell(30, 8, 'Skin Score', 1)
        pdf.cell(30, 8, 'Detections', 1)
        pdf.cell(30, 8, 'Severity', 1)
        pdf.ln()

        # Table data
        for record in history:
            date_str = record['timestamp'][:19].replace('T', ' ')
            pdf.cell(50, 8, date_str, 1)
            pdf.cell(30, 8, f"{record['skin_score']:.1f}", 1)
            pdf.cell(30, 8, str(record['total_detections']), 1)
            pdf.cell(30, 8, f"{record['severity_score']:.1f}", 1)
            pdf.ln()

    # Latest analysis details
    if latest_result:
        pdf.ln(10)
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, 'Latest Analysis Details', 0, 1)
        pdf.set_font('Arial', '', 12)

        detections = latest_result.get('detections', [])
        if detections:
            pdf.cell(0, 8, f'Total Detections: {len(detections)}', 0, 1)

            type_counts = {}
            for det in detections:
                label = det.get('label', 'unknown')
                type_counts[label] = type_counts.get(label, 0) + 1

            pdf.ln(5)
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 8, 'Acne Types Found:', 0, 1)
            pdf.set_font('Arial', '', 12)

            for acne_type, count in type_counts.items():
                pdf.cell(0, 8, f'  {acne_type.title()}: {count}', 0, 1)

    # Add disclaimer
    pdf.ln(20)
    pdf.set_font('Arial', 'I', 10)
    pdf.multi_cell(0, 5, 'Disclaimer: This report is generated by AI analysis and is for informational purposes only. Always consult with a qualified dermatologist for professional medical advice.')

    return pdf

def create_csv_export(history):
    """Create CSV export from analysis history"""
    if not history:
        return pd.DataFrame()

    # Create DataFrame from history
    df = pd.DataFrame(history)

    # Format timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')

    # Flatten acne_types dictionary
    for i, record in enumerate(history):
        acne_types = record.get('acne_types', {})
        for acne_type, confidence in acne_types.items():
            df.loc[i, f'{acne_type}_confidence'] = confidence

    # Select and rename columns
    export_df = df[['timestamp', 'skin_score', 'total_detections', 'severity_score']].copy()
    export_df.columns = ['Date_Time', 'Skin_Score', 'Total_Detections', 'Severity_Score']

    # Add acne type columns if they exist
    acne_cols = [col for col in df.columns if col.endswith('_confidence')]
    for col in acne_cols:
        export_df[col] = df[col].fillna(0)

    return export_df

def assess_photo_quality(image: Image.Image) -> dict:
    """Assess the quality of uploaded photo for acne analysis"""
    quality_score = 0
    issues = []
    recommendations = []

    # Convert to numpy array for analysis
    img_array = np.array(image)

    # 1. Resolution check
    width, height = image.size
    resolution = width * height
    if resolution < 200000:  # Less than 0.2MP
        issues.append("Low resolution")
        recommendations.append("Use higher resolution camera (min 720p)")
    elif resolution > 200000:
        quality_score += 20

    # 2. Brightness analysis
    grayscale = image.convert('L')
    brightness = ImageStat.Stat(grayscale).mean[0]
    if brightness < 50:
        issues.append("Too dark")
        recommendations.append("Improve lighting conditions")
    elif brightness > 200:
        issues.append("Too bright/overexposed")
        recommendations.append("Reduce lighting or move away from direct light")
    else:
        quality_score += 25

    # 3. Blur detection using Laplacian variance
    gray_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    blur_score = cv2.Laplacian(gray_cv, cv2.CV_64F).var()
    if blur_score < 100:
        issues.append("Image appears blurry")
        recommendations.append("Hold camera steady and ensure proper focus")
    else:
        quality_score += 25

    # 4. Color analysis for skin detection
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    # Skin color range in HSV
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
    skin_percentage = (np.sum(skin_mask > 0) / skin_mask.size) * 100

    if skin_percentage < 30:
        issues.append("Limited skin area detected")
        recommendations.append("Ensure face fills most of the frame")
    else:
        quality_score += 15

    # 5. Contrast check
    contrast = ImageStat.Stat(grayscale).stddev[0]
    if contrast < 20:
        issues.append("Low contrast")
        recommendations.append("Improve lighting to increase contrast")
    else:
        quality_score += 15

    # Overall assessment
    if quality_score >= 80:
        assessment = "Excellent"
        color = "success"
    elif quality_score >= 60:
        assessment = "Good"
        color = "info"
    elif quality_score >= 40:
        assessment = "Fair"
        color = "warning"
    else:
        assessment = "Poor"
        color = "error"

    return {
        "score": quality_score,
        "assessment": assessment,
        "color": color,
        "issues": issues,
        "recommendations": recommendations,
        "metrics": {
            "resolution": f"{width}x{height}",
            "brightness": f"{brightness:.1f}",
            "blur_score": f"{blur_score:.1f}",
            "skin_percentage": f"{skin_percentage:.1f}%",
            "contrast": f"{contrast:.1f}"
        }
    }

def calculate_skin_health_score(history: list, current_result: dict = None) -> dict:
    """Calculate comprehensive skin health score"""
    if not history and not current_result:
        return {"score": 0, "grade": "No Data", "factors": {}}

    # Combine history and current result
    all_data = history.copy()
    if current_result:
        all_data.append({
            "timestamp": datetime.now(pytz.timezone('Europe/Istanbul')).isoformat(),
            "skin_score": current_result.get("skin_score", 0),
            "total_detections": len(current_result.get("detections", [])),
            "severity_score": current_result.get("scores", {}).get("severity_score", 0)
        })

    if not all_data:
        return {"score": 0, "grade": "No Data", "factors": {}}

    # Calculate factors
    factors = {}

    # 1. Current severity (40% weight)
    latest_severity = all_data[-1]["severity_score"]
    severity_factor = max(0, 100 - latest_severity)
    factors["Current Severity"] = severity_factor

    # 2. Improvement trend (30% weight)
    if len(all_data) >= 2:
        recent_scores = [d["skin_score"] for d in all_data[-5:]]  # Last 5 scores
        if len(recent_scores) >= 2:
            trend = np.polyfit(range(len(recent_scores)), recent_scores, 1)[0]
            trend_factor = min(100, max(0, 50 + trend * 10))  # Normalize trend
        else:
            trend_factor = 50
    else:
        trend_factor = 50
    factors["Improvement Trend"] = trend_factor

    # 3. Consistency (20% weight)
    skin_scores = [d["skin_score"] for d in all_data]
    if len(skin_scores) > 1:
        consistency = 100 - (np.std(skin_scores) * 2)  # Lower std = higher consistency
        consistency_factor = max(0, min(100, consistency))
    else:
        consistency_factor = 50
    factors["Consistency"] = consistency_factor

    # 4. Analysis frequency (10% weight)
    if len(all_data) >= 7:
        frequency_factor = 100  # Regular monitoring
    elif len(all_data) >= 3:
        frequency_factor = 70   # Moderate monitoring
    else:
        frequency_factor = 30   # Infrequent monitoring
    factors["Monitoring Frequency"] = frequency_factor

    # Calculate weighted score
    weighted_score = (
        severity_factor * 0.4 +
        trend_factor * 0.3 +
        consistency_factor * 0.2 +
        frequency_factor * 0.1
    )

    # Determine grade
    if weighted_score >= 85:
        grade = "A"
    elif weighted_score >= 70:
        grade = "B"
    elif weighted_score >= 55:
        grade = "C"
    elif weighted_score >= 40:
        grade = "D"
    else:
        grade = "F"

    return {
        "score": round(weighted_score, 1),
        "grade": grade,
        "factors": {k: round(v, 1) for k, v in factors.items()}
    }

def generate_treatment_recommendations(detections: list, severity_score: float, history: list = None) -> list:
    """AI-powered treatment recommendations based on analysis"""
    recommendations = []

    # Count acne types
    acne_counts = {}
    for det in detections:
        acne_type = det.get('label', 'unknown')
        acne_counts[acne_type] = acne_counts.get(acne_type, 0) + 1

    total_lesions = len(detections)

    # Severity-based recommendations
    if severity_score >= 80:
        recommendations.extend([
            {
                "category": "Medical",
                "priority": "High",
                "recommendation": "Consult dermatologist immediately for severe acne treatment",
                "details": "May require prescription medications like isotretinoin or antibiotics"
            },
            {
                "category": "Treatment",
                "priority": "High",
                "recommendation": "Professional acne treatments (chemical peels, light therapy)",
                "details": "Consider in-office treatments for faster results"
            }
        ])
    elif severity_score >= 60:
        recommendations.extend([
            {
                "category": "Medical",
                "priority": "Medium",
                "recommendation": "Schedule dermatologist appointment within 2-4 weeks",
                "details": "Moderate acne may benefit from prescription topical treatments"
            },
            {
                "category": "Treatment",
                "priority": "Medium",
                "recommendation": "Combination therapy with benzoyl peroxide and retinoids",
                "details": "Start with lower concentrations to build tolerance"
            }
        ])
    elif severity_score >= 30:
        recommendations.extend([
            {
                "category": "Treatment",
                "priority": "Medium",
                "recommendation": "Over-the-counter acne treatments",
                "details": "Salicylic acid (BHA) or benzoyl peroxide 2.5-5%"
            },
            {
                "category": "Skincare",
                "priority": "Medium",
                "recommendation": "Gentle skincare routine with non-comedogenic products",
                "details": "Cleanse twice daily, moisturize, and use SPF 30+ daily"
            }
        ])

    # Type-specific recommendations
    if 'nodules' in acne_counts or 'cyst' in acne_counts:
        recommendations.append({
            "category": "Medical",
            "priority": "High",
            "recommendation": "Cystic acne requires professional treatment",
            "details": "Never attempt to extract cysts - risk of scarring"
        })

    if 'pustules' in acne_counts and acne_counts['pustules'] > 5:
        recommendations.append({
            "category": "Treatment",
            "priority": "Medium",
            "recommendation": "Antibacterial treatment for pustular acne",
            "details": "Benzoyl peroxide or topical antibiotics may be beneficial"
        })

    if 'comedone' in acne_counts:
        recommendations.append({
            "category": "Treatment",
            "priority": "Low",
            "recommendation": "Exfoliation for comedonal acne",
            "details": "Salicylic acid or gentle physical exfoliation 2-3x per week"
        })

    # Lifestyle recommendations
    recommendations.extend([
        {
            "category": "Lifestyle",
            "priority": "Medium",
            "recommendation": "Avoid touching or picking at skin",
            "details": "Can lead to bacterial spread and scarring"
        },
        {
            "category": "Lifestyle",
            "priority": "Low",
            "recommendation": "Clean pillowcases and phone screens regularly",
            "details": "Reduces bacterial transfer to facial skin"
        },
        {
            "category": "Diet",
            "priority": "Low",
            "recommendation": "Consider reducing dairy and high-glycemic foods",
            "details": "Some studies suggest correlation with acne severity"
        }
    ])

    # Progress-based recommendations
    if history and len(history) >= 3:
        recent_severity = [h.get('severity_score', 0) for h in history[-3:]]
        if all(s >= recent_severity[0] for s in recent_severity):  # Worsening trend
            recommendations.append({
                "category": "Medical",
                "priority": "Medium",
                "recommendation": "Re-evaluate current treatment plan",
                "details": "Condition appears to be worsening - may need stronger treatment"
            })

    return recommendations

def predict_progress(history: list) -> dict:
    """Predict skin improvement trend using machine learning"""
    if len(history) < 3:
        return {"prediction": "Insufficient data", "trend": "unknown"}

    # Prepare data
    df = pd.DataFrame(history)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')

    # Create time features (days since first record)
    df['days'] = (df['timestamp'] - df['timestamp'].iloc[0]).dt.days

    # Fit polynomial regression for skin score prediction
    X = df['days'].values.reshape(-1, 1)
    y = df['skin_score'].values

    # Try linear and polynomial fits
    linear_reg = LinearRegression()
    linear_reg.fit(X, y)

    poly_features = PolynomialFeatures(degree=2)
    X_poly = poly_features.fit_transform(X)
    poly_reg = LinearRegression()
    poly_reg.fit(X_poly, y)

    # Predict next 30 days
    future_days = np.arange(df['days'].iloc[-1] + 1, df['days'].iloc[-1] + 31).reshape(-1, 1)
    linear_pred = linear_reg.predict(future_days)
    poly_pred = poly_reg.predict(poly_features.transform(future_days))

    # Choose better model (less overfitting)
    if len(history) >= 7:
        predictions = poly_pred
        model_type = "polynomial"
    else:
        predictions = linear_pred
        model_type = "linear"

    # Calculate trend
    current_score = df['skin_score'].iloc[-1]
    future_score = np.mean(predictions[:7])  # Average of next week
    trend_change = future_score - current_score

    if trend_change > 5:
        trend = "improving"
        message = f"Predicted {trend_change:.1f} point improvement in next week"
    elif trend_change < -5:
        trend = "worsening"
        message = f"Predicted {abs(trend_change):.1f} point decline - consider treatment review"
    else:
        trend = "stable"
        message = "Skin condition predicted to remain stable"

    return {
        "prediction": message,
        "trend": trend,
        "current_score": current_score,
        "predicted_score": future_score,
        "confidence": min(100, len(history) * 10),  # More data = higher confidence
        "model_type": model_type
    }

def uploader_card():
    st.markdown("<div class='skin-card'>", unsafe_allow_html=True)
    st.markdown("### 📤 Upload & Analyze Your Skin")

    # Advanced settings in expander
    with st.expander("⚙️ Advanced Settings", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            confidence_threshold = st.slider(
                "🎯 Confidence Threshold",
                min_value=0.1,
                max_value=0.9,
                value=0.25,
                step=0.05,
                help="Lower values detect more lesions but may include false positives"
            )

            face_align = st.checkbox("🎭 Face Alignment", value=True, help="Automatically align face for better detection")

        with col2:
            illum_norm = st.checkbox("💡 Illumination Normalization", value=True, help="Normalize lighting conditions")

            enable_preprocessing = st.checkbox("🔧 Enable Image Enhancement", value=False, help="Apply image sharpening and contrast enhancement")

    uploaded = st.file_uploader(
        "Choose a photo (jpg/png)",
        type=["jpg","jpeg","png"],
        help="For best results, use a clear, well-lit photo of your face"
    )

    if uploaded:
        image = Image.open(uploaded).convert("RGB")

        # Photo Quality Assessment
        with st.expander("📋 Photo Quality Assessment", expanded=True):
            try:
                quality_result = assess_photo_quality(image)

                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.markdown(f"""
                    <div class='stat-box'>
                        <span class='stat-number' style='color: var(--{quality_result["color"]});'>{quality_result["score"]}/100</span>
                        <div class='stat-label'>Quality Score - {quality_result["assessment"]}</div>
                    </div>
                    """, unsafe_allow_html=True)

                # Quality metrics
                metrics_col1, metrics_col2 = st.columns(2)
                with metrics_col1:
                    st.markdown("**Quality Metrics:**")
                    for metric, value in quality_result["metrics"].items():
                        st.write(f"• {metric.replace('_', ' ').title()}: {value}")

                with metrics_col2:
                    if quality_result["issues"]:
                        st.markdown("**Issues Found:**")
                        for issue in quality_result["issues"]:
                            st.warning(f"⚠️ {issue}")

                    if quality_result["recommendations"]:
                        st.markdown("**Recommendations:**")
                        for rec in quality_result["recommendations"]:
                            st.info(f"💡 {rec}")

            except Exception as e:
                st.warning(f"Quality assessment failed: {str(e)}")

        col1, col2 = st.columns([2, 1])
        with col1:
            st.image(image, caption="📷 Original Image", use_column_width=True)

        with col2:
            # Image info
            width, height = image.size
            st.markdown(f"""
            <div class='stat-box'>
                <div class='stat-label'>Image Dimensions</div>
                <span class='stat-number' style='font-size: 1.2rem;'>{width}×{height}</span>
            </div>
            """, unsafe_allow_html=True)

            file_size = len(uploaded.getvalue()) / 1024  # KB
            st.markdown(f"**File Size:** {file_size:.1f} KB")

        # Image preprocessing options
        if enable_preprocessing:
            import cv2
            import numpy as np

            # Convert PIL to OpenCV
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            # Apply enhancement
            lab = cv2.cvtColor(cv_image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            enhanced = cv2.merge((cl,a,b))
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)

            # Convert back to PIL
            image = Image.fromarray(enhanced)
            st.image(image, caption="✨ Enhanced Image", use_column_width=True)

        # convert to base64 for gateway / backend call
        b64 = _encode_image_to_b64(image)
        st.session_state["last_image_b64"] = b64
        st.session_state["confidence_threshold"] = confidence_threshold

        col1, col2 = st.columns([1,1])
        with col1:
            if st.button("Run Analysis"):
                st.session_state.setdefault("analysis_log", [])
                confidence_threshold = st.session_state.get("confidence_threshold", 0.25)
                payload = {
                    "request_id": str(uuid.uuid4()),
                    "image_b64": b64,
                    "capture_ts": datetime.now(pytz.timezone('Europe/Istanbul')).isoformat(),
                    "preprocess": {
                        "face_align": face_align,
                        "illum_norm": illum_norm,
                        "confidence_threshold": confidence_threshold
                    }
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

                            # Store detections for visualization
                            detections = result.get("detections", [])
                            st.session_state["last_detections"] = detections
                            st.session_state["last_result"] = result

                            # Show detection results with modern styling
                            if detections:
                                st.markdown(f"""
                                <div class='result-success'>
                                    <strong>🎯 Analysis Complete!</strong><br>
                                    Found {len(detections)} acne lesion(s) in the uploaded image
                                </div>
                                """, unsafe_allow_html=True)

                                # Draw bounding boxes on image
                                img_with_boxes = draw_detections_on_image(image, detections)

                                col1, col2 = st.columns([2, 1])
                                with col1:
                                    st.image(img_with_boxes, caption="🔍 Detected Acne Lesions", use_column_width=True)

                                with col2:
                                    # Severity score display
                                    severity = result.get("scores", {}).get("severity_score", 0)
                                    st.markdown(f"""
                                    <div class='stat-box'>
                                        <span class='stat-number'>{severity:.1f}</span>
                                        <div class='stat-label'>Severity Score</div>
                                    </div>
                                    """, unsafe_allow_html=True)

                                    # Count by type
                                    type_counts = {}
                                    for det in detections:
                                        label = det.get('label', 'unknown')
                                        type_counts[label] = type_counts.get(label, 0) + 1

                                    st.markdown("**Acne Types Found:**")
                                    for acne_type, count in type_counts.items():
                                        st.markdown(f"""
                                        <div class='detection-type type-{acne_type}'>
                                            {acne_type.title()}: {count}
                                        </div>
                                        """, unsafe_allow_html=True)

                                # Detailed detection cards
                                st.markdown("### 📋 Detection Details")
                                for i, det in enumerate(detections):
                                    label = det.get('label', 'unknown')
                                    conf = det.get('confidence', 0.0)
                                    x, y, w, h = det.get('x', 0), det.get('y', 0), det.get('w', 0), det.get('h', 0)

                                    confidence_color = "success" if conf > 0.7 else "warning" if conf > 0.5 else "error"
                                    st.markdown(f"""
                                    <div class='detection-card'>
                                        <strong>#{i+1} {label.title()}</strong>
                                        <br>📍 Position: ({x}, {y}) • 📏 Size: {w}×{h}px
                                        <br>🎯 Confidence: <span class='badge badge-{confidence_color}'>{conf:.1%}</span>
                                    </div>
                                    """, unsafe_allow_html=True)

                                # Severity warning
                                if severity > 60:
                                    st.markdown(f"""
                                    <div class='result-warning'>
                                        ⚠️ <strong>High Severity Detected</strong><br>
                                        Consider consulting a dermatologist for professional treatment
                                    </div>
                                    """, unsafe_allow_html=True)
                                elif severity > 30:
                                    st.markdown(f"""
                                    <div class='result-info'>
                                        ℹ️ <strong>Moderate Acne</strong><br>
                                        Regular skincare routine and monitoring recommended
                                    </div>
                                    """, unsafe_allow_html=True)
                            else:
                                st.markdown("""
                                <div class='result-info'>
                                    😊 <strong>Great News!</strong><br>
                                    No acne lesions detected in this image. Your skin looks clear!
                                </div>
                                """, unsafe_allow_html=True)

                            # Extract metrics for backward compatibility
                            metrics = {
                                "overall_skin_score": result.get("skin_score", 0),
                                "acne_index": result.get("scores", {}).get("total_acne_count", 0),
                                "pores": result.get("scores", {}).get("pores", 0),
                                "pigmentation": result.get("scores", {}).get("pigmentation", 0),
                                "hydration": result.get("scores", {}).get("hydration", 0)
                            }
                            st.session_state["last_metrics"] = metrics

                            # Generate AI-powered treatment recommendations
                            st.markdown("### 🎯 Personalized Treatment Recommendations")
                            try:
                                history = st.session_state.get("analysis_history", [])
                                recommendations = generate_treatment_recommendations(
                                    detections,
                                    result.get("scores", {}).get("severity_score", 0),
                                    history
                                )

                                # Group recommendations by priority
                                high_priority = [r for r in recommendations if r.get("priority") == "High"]
                                medium_priority = [r for r in recommendations if r.get("priority") == "Medium"]
                                low_priority = [r for r in recommendations if r.get("priority") == "Low"]

                                if high_priority:
                                    st.markdown("#### 🚨 High Priority")
                                    for rec in high_priority:
                                        st.markdown(f"""
                                        <div class='result-warning'>
                                            <strong>{rec['category']}: {rec['recommendation']}</strong><br>
                                            <small>{rec['details']}</small>
                                        </div>
                                        """, unsafe_allow_html=True)

                                if medium_priority:
                                    st.markdown("#### ⚠️ Medium Priority")
                                    for rec in medium_priority:
                                        st.markdown(f"""
                                        <div class='result-info'>
                                            <strong>{rec['category']}: {rec['recommendation']}</strong><br>
                                            <small>{rec['details']}</small>
                                        </div>
                                        """, unsafe_allow_html=True)

                                if low_priority:
                                    with st.expander("💡 Additional Recommendations"):
                                        for rec in low_priority:
                                            st.markdown(f"""
                                            <div class='detection-card'>
                                                <strong>{rec['category']}: {rec['recommendation']}</strong><br>
                                                <small>{rec['details']}</small>
                                            </div>
                                            """, unsafe_allow_html=True)

                            except Exception as e:
                                st.warning(f"Could not generate recommendations: {str(e)}")

                            # Save analysis to history
                            analysis_record = {
                                "timestamp": datetime.now(pytz.timezone('Europe/Istanbul')).isoformat(),
                                "skin_score": result.get("skin_score", 0),
                                "total_detections": len(detections),
                                "severity_score": result.get("scores", {}).get("severity_score", 0),
                                "acne_types": {det['label']: det['confidence'] for det in detections},
                                "trace_id": result.get("trace_id")
                            }

                            if "analysis_history" not in st.session_state:
                                st.session_state["analysis_history"] = []
                            st.session_state["analysis_history"].append(analysis_record)

                            st.success("Analysis complete — check Skin Passport for full report")
                            st.session_state["analysis_log"].append({"time": datetime.now(pytz.timezone('Europe/Istanbul')).isoformat(), "status": "ok", "trace_id": result.get("trace_id")})

                            # Quick export for current analysis
                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button("📄 Export Current Analysis", key="export_current"):
                                    # Create a single analysis record for export
                                    current_analysis = {
                                        "timestamp": datetime.now(pytz.timezone('Europe/Istanbul')).isoformat(),
                                        "skin_score": result.get("skin_score", 0),
                                        "total_detections": len(detections),
                                        "severity_score": result.get("scores", {}).get("severity_score", 0),
                                        "acne_types": {det['label']: det['confidence'] for det in detections},
                                        "trace_id": result.get("trace_id"),
                                        "detections": detections
                                    }

                                    json_data = json.dumps(current_analysis, indent=2)
                                    st.download_button(
                                        label="📥 Download Analysis JSON",
                                        data=json_data,
                                        file_name=f"current_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                                        mime="application/json"
                                    )
                        else:
                            st.error(f"Analysis failed: {resp.status_code}")
                            st.session_state["analysis_log"].append({"time": datetime.now(pytz.timezone('Europe/Istanbul')).isoformat(), "status": "error", "code": resp.status_code})
                    except Exception as e:
                        st.error("Error calling gateway (network/exception). See details below.")
                        st.exception(e)
                        st.session_state["analysis_log"].append({"time": datetime.now(pytz.timezone('Europe/Istanbul')).isoformat(), "status": "exception", "error": str(e)})

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
        # Welcome section with stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div class='stat-box'>
                <span class='stat-number'>4</span>
                <div class='stat-label'>Acne Types Detected</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class='stat-box'>
                <span class='stat-number'>98%</span>
                <div class='stat-label'>AI Accuracy</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown("""
            <div class='stat-box'>
                <span class='stat-number'>< 2s</span>
                <div class='stat-label'>Analysis Time</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Educational content about acne types
        st.markdown("## 📚 Understanding Acne Types")

        acne_types = [
            {
                "name": "Comedones",
                "description": "Non-inflammatory acne including blackheads and whiteheads",
                "severity": "Mild",
                "color": "#dcfce7",
                "emoji": "⚪",
                "treatment": "Gentle exfoliation, salicylic acid, retinoids"
            },
            {
                "name": "Papules",
                "description": "Small, inflamed bumps without pus",
                "severity": "Moderate",
                "color": "#dbeafe",
                "emoji": "🔴",
                "treatment": "Benzoyl peroxide, topical antibiotics"
            },
            {
                "name": "Pustules",
                "description": "Inflamed lesions filled with pus",
                "severity": "Moderate",
                "color": "#fef3c7",
                "emoji": "🟡",
                "treatment": "Benzoyl peroxide, topical antibiotics, professional care"
            },
            {
                "name": "Nodules",
                "description": "Deep, painful lumps under the skin",
                "severity": "Severe",
                "color": "#fee2e2",
                "emoji": "🔺",
                "treatment": "Professional dermatological treatment required"
            }
        ]

        cols = st.columns(2)
        for i, acne_type in enumerate(acne_types):
            with cols[i % 2]:
                st.markdown(f"""
                <div class='detection-card' style='background: {acne_type["color"]}; border-color: {acne_type["color"]}'>
                    <h4>{acne_type["emoji"]} {acne_type["name"]}</h4>
                    <p><strong>Severity:</strong> {acne_type["severity"]}</p>
                    <p>{acne_type["description"]}</p>
                    <p><strong>Treatment:</strong> {acne_type["treatment"]}</p>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")

        # Photo Capture Guidelines
        st.markdown("## 📸 Photo Capture Guidelines")
        st.markdown("For optimal acne analysis results, follow these professional guidelines:")

        guidelines_cols = st.columns(3)

        with guidelines_cols[0]:
            st.markdown(f"""
            <div class='detection-card' style='background: #f0f9ff; border-color: #0ea5e9;'>
                <h4>📱 Setup & Positioning</h4>
                <ul style='margin: 0; padding-left: 1.2rem; color: #374151;'>
                    <li>Hold phone 12-18 inches from face</li>
                    <li>Face should fill 70% of frame</li>
                    <li>Keep phone steady and parallel</li>
                    <li>Look directly into camera</li>
                    <li>Use rear camera for best quality</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        with guidelines_cols[1]:
            st.markdown(f"""
            <div class='detection-card' style='background: #f0fdf4; border-color: #22c55e;'>
                <h4>💡 Lighting Requirements</h4>
                <ul style='margin: 0; padding-left: 1.2rem; color: #374151;'>
                    <li>Use natural daylight when possible</li>
                    <li>Face light source directly</li>
                    <li>Avoid harsh shadows</li>
                    <li>No direct flash or overhead lighting</li>
                    <li>Ensure even illumination</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        with guidelines_cols[2]:
            st.markdown(f"""
            <div class='detection-card' style='background: #fefce8; border-color: #eab308;'>
                <h4>🎯 Best Practices</h4>
                <ul style='margin: 0; padding-left: 1.2rem; color: #374151;'>
                    <li>Clean face before photography</li>
                    <li>No makeup or filters</li>
                    <li>Take multiple angles if needed</li>
                    <li>Consistent timing (same time of day)</li>
                    <li>Stable environment</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        # Quality checklist
        st.markdown("### ✅ Pre-Analysis Checklist")
        checklist_cols = st.columns(2)

        with checklist_cols[0]:
            st.markdown("""
            **Image Quality:**
            - ✅ Resolution: Minimum 720p (1280x720)
            - ✅ Focus: Sharp, not blurry
            - ✅ Brightness: Neither too dark nor overexposed
            - ✅ Contrast: Sufficient detail visibility
            """)

        with checklist_cols[1]:
            st.markdown("""
            **Face Coverage:**
            - ✅ Skin area: At least 30% of image
            - ✅ Face position: Centered in frame
            - ✅ Angle: Straight-on view preferred
            - ✅ Expression: Neutral, relaxed
            """)

        st.markdown("---")

        # Upload section
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
        st.markdown("## 📈 Progress Tracking")

        # Check if we have analysis history
        history = st.session_state.get("analysis_history", [])

        if not history:
            st.markdown("""
            <div class='result-info'>
                📊 <strong>No Analysis History Yet</strong><br>
                Upload and analyze images to start tracking your progress
            </div>
            """, unsafe_allow_html=True)
            return

        # Comprehensive Skin Health Score
        current_result = st.session_state.get("last_result")
        health_score = calculate_skin_health_score(history, current_result)

        st.markdown("### 🏆 Comprehensive Skin Health Score")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            grade_color = {
                "A": "#10b981", "B": "#3b82f6", "C": "#f59e0b", "D": "#ef4444", "F": "#dc2626"
            }.get(health_score["grade"], "#6b7280")

            st.markdown(f"""
            <div class='stat-box' style='background: linear-gradient(135deg, {grade_color}20, {grade_color}10);'>
                <span class='stat-number' style='color: {grade_color}; font-size: 3rem;'>{health_score["grade"]}</span>
                <div class='stat-label' style='font-size: 1.1rem;'>Overall Grade</div>
                <div style='color: {grade_color}; font-weight: 600;'>{health_score["score"]}/100 Health Score</div>
            </div>
            """, unsafe_allow_html=True)

        # Health Score Factors
        st.markdown("#### 📊 Health Score Breakdown")
        factor_cols = st.columns(len(health_score["factors"]))
        for i, (factor, score) in enumerate(health_score["factors"].items()):
            with factor_cols[i]:
                color = "#10b981" if score >= 70 else "#f59e0b" if score >= 50 else "#ef4444"
                st.markdown(f"""
                <div class='stat-box' style='padding: 1rem;'>
                    <span class='stat-number' style='color: {color}; font-size: 1.5rem;'>{score:.0f}</span>
                    <div class='stat-label' style='font-size: 0.8rem;'>{factor}</div>
                </div>
                """, unsafe_allow_html=True)

        # AI Predictions
        st.markdown("### 🔮 AI-Powered Progress Predictions")
        try:
            prediction = predict_progress(history)
            prediction_color = {
                "improving": "#10b981",
                "stable": "#3b82f6",
                "worsening": "#ef4444"
            }.get(prediction["trend"], "#6b7280")

            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown(f"""
                <div class='result-info' style='background: linear-gradient(135deg, {prediction_color}20, {prediction_color}10); border-left: 4px solid {prediction_color};'>
                    <strong>7-Day Prediction</strong><br>
                    {prediction["prediction"]}<br>
                    <small>Confidence: {prediction["confidence"]:.0f}% | Model: {prediction["model_type"]}</small>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div class='stat-box'>
                    <span class='stat-number' style='color: {prediction_color};'>{prediction["predicted_score"]:.1f}</span>
                    <div class='stat-label'>Predicted Score</div>
                </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            st.info("🤖 Predictions will be available after more data is collected")

        # Progress statistics with enhanced metrics
        st.markdown("### 📈 Progress Analytics")
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            total_analyses = len(history)
            st.markdown(f"""
            <div class='stat-box'>
                <span class='stat-number'>{total_analyses}</span>
                <div class='stat-label'>Total Analyses</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            avg_skin_score = sum(h['skin_score'] for h in history) / len(history)
            st.markdown(f"""
            <div class='stat-box'>
                <span class='stat-number'>{avg_skin_score:.1f}</span>
                <div class='stat-label'>Avg Skin Score</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            latest_score = history[-1]['skin_score']
            # Calculate trend arrow
            if len(history) >= 2:
                prev_score = history[-2]['skin_score']
                trend = "↗️" if latest_score > prev_score else "↘️" if latest_score < prev_score else "➡️"
            else:
                trend = ""

            st.markdown(f"""
            <div class='stat-box'>
                <span class='stat-number'>{latest_score:.1f} {trend}</span>
                <div class='stat-label'>Latest Score</div>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            total_detections = sum(h['total_detections'] for h in history)
            avg_detections = total_detections / len(history)
            st.markdown(f"""
            <div class='stat-box'>
                <span class='stat-number'>{avg_detections:.1f}</span>
                <div class='stat-label'>Avg Detections</div>
            </div>
            """, unsafe_allow_html=True)

        with col5:
            # Calculate improvement percentage
            if len(history) >= 2:
                first_score = history[0]['skin_score']
                improvement = ((latest_score - first_score) / first_score) * 100
                improvement_color = "#10b981" if improvement > 0 else "#ef4444" if improvement < 0 else "#6b7280"
                improvement_text = f"{improvement:+.1f}%"
            else:
                improvement_color = "#6b7280"
                improvement_text = "N/A"

            st.markdown(f"""
            <div class='stat-box'>
                <span class='stat-number' style='color: {improvement_color};'>{improvement_text}</span>
                <div class='stat-label'>Overall Change</div>
            </div>
            """, unsafe_allow_html=True)

        # Advanced Progress Visualization
        if len(history) > 1:
            df = pd.DataFrame(history)
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            # Multi-metric dashboard
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Skin Score Trend', 'Severity Trend', 'Detection Count', 'Improvement Rate'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )

            # Skin Score with trend line
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=df['skin_score'],
                          mode='lines+markers', name='Skin Score',
                          line=dict(color='#10b981', width=3),
                          marker=dict(size=8)),
                row=1, col=1
            )

            # Add trend line if enough data
            if len(df) >= 3:
                z = np.polyfit(range(len(df)), df['skin_score'], 1)
                trend_line = np.poly1d(z)(range(len(df)))
                fig.add_trace(
                    go.Scatter(x=df['timestamp'], y=trend_line,
                              mode='lines', name='Trend',
                              line=dict(color='#10b981', width=2, dash='dash')),
                    row=1, col=1
                )

            # Severity score
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=df['severity_score'],
                          mode='lines+markers', name='Severity',
                          line=dict(color='#ef4444', width=3),
                          marker=dict(size=8)),
                row=1, col=2
            )

            # Detection count
            fig.add_trace(
                go.Bar(x=df['timestamp'], y=df['total_detections'],
                      name='Detections', marker_color='#3b82f6'),
                row=2, col=1
            )

            # Improvement rate (rolling change)
            if len(df) >= 2:
                improvement_rate = df['skin_score'].pct_change().rolling(window=2).mean() * 100
                fig.add_trace(
                    go.Scatter(x=df['timestamp'], y=improvement_rate,
                              mode='lines+markers', name='Improvement %',
                              line=dict(color='#8b5cf6', width=3),
                              marker=dict(size=8)),
                    row=2, col=2
                )

            fig.update_layout(
                height=600,
                showlegend=False,
                title_text="📊 Comprehensive Progress Analytics",
                title_x=0.5
            )

            # Update y-axes
            fig.update_yaxes(title_text="Score", range=[0, 100], row=1, col=1)
            fig.update_yaxes(title_text="Severity", range=[0, 100], row=1, col=2)
            fig.update_yaxes(title_text="Count", row=2, col=1)
            fig.update_yaxes(title_text="Change %", row=2, col=2)

            st.plotly_chart(fig, use_container_width=True)

            # Acne type distribution over time
            st.markdown("#### 🎯 Acne Type Analysis")
            acne_type_data = []
            for record in history:
                timestamp = record['timestamp']
                for acne_type, confidence in record.get('acne_types', {}).items():
                    acne_type_data.append({
                        'timestamp': timestamp,
                        'acne_type': acne_type,
                        'confidence': confidence,
                        'count': 1
                    })

            if acne_type_data:
                acne_df = pd.DataFrame(acne_type_data)
                acne_df['timestamp'] = pd.to_datetime(acne_df['timestamp'])

                # Group by type and sum counts
                type_summary = acne_df.groupby(['timestamp', 'acne_type']).sum().reset_index()

                fig_types = px.area(type_summary, x='timestamp', y='count',
                                   color='acne_type', title='Acne Type Distribution Over Time',
                                   color_discrete_map={
                                       'comedone': '#fbbf24',
                                       'papules': '#34d399',
                                       'pustules': '#60a5fa',
                                       'nodules': '#f87171'
                                   })
                fig_types.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Count",
                    height=400
                )
                st.plotly_chart(fig_types, use_container_width=True)

        # Analysis history table
        st.markdown("### 📋 Analysis History")
        history_df = pd.DataFrame(history)
        history_df['timestamp'] = pd.to_datetime(history_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')

        # Display table
        st.dataframe(
            history_df[['timestamp', 'skin_score', 'total_detections', 'severity_score']],
            column_config={
                "timestamp": "Date & Time",
                "skin_score": st.column_config.NumberColumn("Skin Score", format="%.1f"),
                "total_detections": "Detections",
                "severity_score": st.column_config.NumberColumn("Severity", format="%.1f")
            },
            use_container_width=True
        )

        # Export functionality
        st.markdown("### 📥 Export Reports")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("📄 Export PDF Report", type="primary"):
                try:
                    # Get latest result for detailed analysis
                    latest_result = st.session_state.get("last_result")

                    # Create PDF
                    pdf = create_pdf_report(history, latest_result)

                    # Save to temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                        pdf.output(tmp_file.name)

                        # Read the file and create download
                        with open(tmp_file.name, 'rb') as f:
                            pdf_data = f.read()

                        # Clean up
                        os.unlink(tmp_file.name)

                        # Create download button
                        st.download_button(
                            label="📥 Download PDF Report",
                            data=pdf_data,
                            file_name=f"acne_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                            mime="application/pdf"
                        )

                        st.success("PDF report generated successfully!")

                except Exception as e:
                    st.error(f"Error generating PDF: {str(e)}")
                    st.info("Note: fpdf2 package is required. Install with: pip install fpdf2")

        with col2:
            if st.button("📊 Export CSV Data", type="secondary"):
                try:
                    # Create CSV
                    csv_df = create_csv_export(history)

                    if not csv_df.empty:
                        # Convert to CSV
                        csv_data = csv_df.to_csv(index=False)

                        # Create download button
                        st.download_button(
                            label="📥 Download CSV Data",
                            data=csv_data,
                            file_name=f"acne_analysis_data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                            mime="text/csv"
                        )

                        st.success("CSV data exported successfully!")

                        # Show preview
                        with st.expander("📋 Data Preview"):
                            st.dataframe(csv_df.head(), use_container_width=True)
                    else:
                        st.warning("No data available for export")

                except Exception as e:
                    st.error(f"Error generating CSV: {str(e)}")

        # Individual analysis export
        if history:
            st.markdown("### 📋 Individual Analysis Export")

            # Select analysis to export
            analysis_options = [f"Analysis {i+1} - {h['timestamp'][:19].replace('T', ' ')}" for i, h in enumerate(history)]
            selected_analysis = st.selectbox("Select analysis to export:", analysis_options)

            if selected_analysis:
                selected_idx = int(selected_analysis.split()[1]) - 1
                selected_record = history[selected_idx]

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Analysis Details:**")
                    st.write(f"**Date:** {selected_record['timestamp'][:19].replace('T', ' ')}")
                    st.write(f"**Skin Score:** {selected_record['skin_score']:.1f}")
                    st.write(f"**Total Detections:** {selected_record['total_detections']}")
                    st.write(f"**Severity Score:** {selected_record['severity_score']:.1f}")

                    if selected_record.get('acne_types'):
                        st.write("**Acne Types:**")
                        for acne_type, confidence in selected_record['acne_types'].items():
                            st.write(f"  - {acne_type}: {confidence:.2f}")

                with col2:
                    # Export single analysis as JSON
                    if st.button("📄 Export as JSON"):
                        json_data = json.dumps(selected_record, indent=2)

                        st.download_button(
                            label="📥 Download JSON",
                            data=json_data,
                            file_name=f"analysis_{selected_idx+1}_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                            mime="application/json"
                        )
    elif page == "Analytics":
        st.markdown("## 🎯 Advanced Analytics Dashboard")

        history = st.session_state.get("analysis_history", [])
        current_result = st.session_state.get("last_result")

        if not history:
            st.markdown("""
            <div class='result-info'>
                📊 <strong>Analytics Dashboard</strong><br>
                Advanced analytics will appear here once you have performed multiple skin analyses
            </div>
            """, unsafe_allow_html=True)
            return

        # Comprehensive health score
        health_score = calculate_skin_health_score(history, current_result)

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            grade_color = {
                "A": "#10b981", "B": "#3b82f6", "C": "#f59e0b", "D": "#ef4444", "F": "#dc2626"
            }.get(health_score["grade"], "#6b7280")

            st.markdown(f"""
            <div class='stat-box' style='background: linear-gradient(135deg, {grade_color}20, {grade_color}10); border: 2px solid {grade_color}40;'>
                <span class='stat-number' style='color: {grade_color}; font-size: 4rem;'>{health_score["grade"]}</span>
                <div class='stat-label' style='font-size: 1.3rem; font-weight: 600;'>Comprehensive Health Grade</div>
                <div style='color: {grade_color}; font-weight: 600; font-size: 1.1rem;'>{health_score["score"]}/100 Overall Score</div>
            </div>
            """, unsafe_allow_html=True)

        # AI Predictions with confidence intervals
        st.markdown("### 🔮 AI-Powered Predictions & Insights")
        try:
            prediction = predict_progress(history)

            col1, col2 = st.columns([3, 1])
            with col1:
                prediction_color = {
                    "improving": "#10b981", "stable": "#3b82f6", "worsening": "#ef4444"
                }.get(prediction["trend"], "#6b7280")

                st.markdown(f"""
                <div class='result-info' style='background: linear-gradient(135deg, {prediction_color}15, {prediction_color}05); border-left: 6px solid {prediction_color};'>
                    <h4 style='margin: 0; color: {prediction_color};'>🎯 7-Day Forecast</h4>
                    <p style='margin: 0.5rem 0; font-size: 1.1rem;'><strong>{prediction["prediction"]}</strong></p>
                    <small>Model Confidence: {prediction["confidence"]:.0f}% • Algorithm: {prediction["model_type"].title()}</small><br>
                    <small>Current Score: {prediction["current_score"]:.1f} → Predicted: {prediction["predicted_score"]:.1f}</small>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                trend_emoji = {"improving": "📈", "stable": "➡️", "worsening": "📉"}.get(prediction["trend"], "❓")
                st.markdown(f"""
                <div class='stat-box' style='text-align: center;'>
                    <span style='font-size: 3rem;'>{trend_emoji}</span>
                    <div class='stat-label'>Trend: {prediction["trend"].title()}</div>
                </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            st.info("🤖 Advanced predictions require at least 3 analysis sessions")

        # Treatment effectiveness analysis
        if len(history) >= 3:
            st.markdown("### 📈 Progress Analysis")

            df = pd.DataFrame(history)
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            # Calculate improvement rate
            first_score = df['skin_score'].iloc[0]
            latest_score = df['skin_score'].iloc[-1]
            total_improvement = latest_score - first_score
            time_span = (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).days

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                improvement_color = "#10b981" if total_improvement > 0 else "#ef4444" if total_improvement < 0 else "#6b7280"
                st.metric(
                    "Total Improvement",
                    f"{total_improvement:+.1f} points",
                    delta=f"{(total_improvement/first_score)*100:+.1f}%" if first_score > 0 else None
                )

            with col2:
                avg_improvement = total_improvement / max(time_span, 1)
                st.metric(
                    "Daily Rate",
                    f"{avg_improvement:+.2f}/day",
                    help="Average daily improvement rate"
                )

            with col3:
                consistency = 100 - (df['skin_score'].std() * 2)
                consistency = max(0, min(100, consistency))
                st.metric(
                    "Consistency",
                    f"{consistency:.0f}%",
                    help="How consistent your progress has been"
                )

            with col4:
                frequency = len(df) / max(time_span, 1) * 7
                st.metric(
                    "Analysis Frequency",
                    f"{frequency:.1f}/week",
                    help="How often you analyze your skin"
                )

        # Detailed factor analysis
        st.markdown("### 📊 Health Score Factors")
        factor_cols = st.columns(len(health_score["factors"]))
        for i, (factor, score) in enumerate(health_score["factors"].items()):
            with factor_cols[i]:
                color = "#10b981" if score >= 70 else "#f59e0b" if score >= 50 else "#ef4444"
                progress = score / 100

                st.markdown(f"""
                <div class='stat-box'>
                    <span class='stat-number' style='color: {color}; font-size: 2.5rem;'>{score:.0f}</span>
                    <div class='stat-label'>{factor}</div>
                    <div style='background: #f1f5f9; border-radius: 10px; height: 8px; margin-top: 0.5rem;'>
                        <div style='background: {color}; height: 100%; width: {progress*100}%; border-radius: 10px;'></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # Advanced recommendations based on analytics
        st.markdown("### 🎯 Data-Driven Recommendations")
        if current_result:
            detections = current_result.get("detections", [])
            severity = current_result.get("scores", {}).get("severity_score", 0)
            recommendations = generate_treatment_recommendations(detections, severity, history)

            # Filter to most relevant recommendations
            top_recommendations = [r for r in recommendations if r.get("priority") in ["High", "Medium"]][:3]

            for i, rec in enumerate(top_recommendations):
                priority_color = {"High": "#ef4444", "Medium": "#f59e0b", "Low": "#10b981"}.get(rec["priority"], "#6b7280")
                st.markdown(f"""
                <div class='detection-card' style='border-left: 4px solid {priority_color};'>
                    <strong>{rec["category"]}: {rec["recommendation"]}</strong><br>
                    <small style='color: #6b7280;'>{rec["details"]}</small>
                </div>
                """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()