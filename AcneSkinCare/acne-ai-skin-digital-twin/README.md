# 🎯 AcneSkin AI - Advanced Acne Detection Platform

AI-powered acne detection and analysis platform using YOLOv8 computer vision and advanced analytics.

## ✨ Features

- **AI Acne Detection** - Custom YOLOv8 model for 4 acne types (comedone, papules, pustules, nodules)
- **Real-time Analysis** - Live bounding box visualization with confidence scores
- **Advanced Analytics** - Progress tracking, trend analysis, and ML predictions
- **Photo Quality Assessment** - Computer vision analysis for optimal image quality
- **Treatment Recommendations** - AI-powered medical-grade suggestions
- **Skin Health Scoring** - Comprehensive multi-factor scoring algorithm
- **Progress Tracking** - Historical analysis with predictive modeling
- **Professional Reports** - PDF generation with detailed analytics

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Custom YOLOv8 acne detection model (place in `models/acne_detection.pt`)

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/melikeZaman04/AcneSkin.git
cd AcneSkin/AcneSkinCare
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
pip install -r acne-ai-skin-digital-twin/requirements.txt
```

3. **Place your custom model:**
```bash
# Copy your trained YOLOv8 model to:
# acne-ai-skin-digital-twin/models/acne_detection.pt
```

### Running the Application

**Option 1: VS Code (Recommended)**

Open VS Code and create 2 terminals:

**Terminal 1 - Backend (Gateway):**
```bash
cd F:\saglik\AcneSkin\AcneSkinCare
python -m uvicorn gateway.main:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 2 - Frontend (Streamlit):**
```bash
cd F:\saglik\AcneSkin\AcneSkinCare\acne-ai-skin-digital-twin
streamlit run app.py --server.port 8502
```

**Option 2: Command Line**

```bash
# Terminal 1
cd AcneSkinCare
python -m uvicorn gateway.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2 (new terminal)
cd AcneSkinCare/acne-ai-skin-digital-twin
streamlit run app.py --server.port 8502
```

### Access the Application

- **Web Interface**: http://localhost:8502
- **API Backend**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## 📁 Project Structure

```
AcneSkin/
├── AcneSkinCare/
│   ├── gateway/                    # FastAPI backend
│   │   ├── adapters/              # Model adapters
│   │   │   ├── acne_adapter.py    # YOLOv8 acne detection
│   │   │   └── base.py            # Base adapter class
│   │   ├── main.py                # FastAPI app
│   │   └── schemas.py             # Pydantic schemas
│   └── acne-ai-skin-digital-twin/ # Streamlit frontend
│       ├── src/
│       │   ├── ai/
│       │   │   └── acne_yolo.py   # YOLO detection logic
│       │   └── components/
│       │       └── style.css      # Custom styling
│       ├── models/                # AI models directory
│       │   └── acne_detection.pt  # Custom YOLOv8 model
│       ├── .streamlit/
│       │   └── secrets.toml       # Streamlit config
│       ├── app.py                 # Main Streamlit app
│       └── requirements.txt       # Frontend dependencies
```

## 🔧 Configuration

### Model Configuration

Place your custom YOLOv8 acne detection model in:
```
acne-ai-skin-digital-twin/models/acne_detection.pt
```

The model should be trained to detect 4 acne classes:
- **Class 0**: comedone (blackheads/whiteheads)
- **Class 1**: nodules (deep inflammatory lesions)
- **Class 2**: papules (inflamed bumps)
- **Class 3**: pustules (pus-filled lesions)

### Streamlit Secrets

Configure in `.streamlit/secrets.toml`:
```toml
[general]
GATEWAY_URL = "http://127.0.0.1:8000/infer"

[app]
title = "AcneSkin AI"
description = "AI-Powered Acne Detection & Analysis"
```

## 🎯 Usage

1. **Upload Image**: Select a clear facial photo (JPG/PNG)
2. **Adjust Sensitivity**: Use the confidence threshold slider (0.1-0.9)
3. **Analyze**: Click "Analyze Image" to run AI detection
4. **View Results**: See detected acne types with bounding boxes
5. **Track Progress**: Use navigation to access analytics and reports

### Navigation Pages

- **🏠 Home**: Guidelines and best practices
- **📸 Upload**: Image upload and analysis
- **🆔 Skin Passport**: Personal detailed report
- **📈 Progress**: Historical tracking and predictions
- **📊 Analytics**: Advanced dashboard and insights

## 🔬 Technical Details

### AI Model
- **Architecture**: YOLOv8 (You Only Look Once)
- **Training Dataset**: Custom Roboflow acne dataset
- **Classes**: 4 acne types with medical accuracy
- **Input**: RGB images (any resolution)
- **Output**: Bounding boxes with confidence scores

### Backend (Gateway)
- **Framework**: FastAPI
- **Features**: Model inference, preprocessing, response formatting
- **Endpoints**: `/infer` for image analysis
- **Validation**: Pydantic schemas for type safety

### Frontend (Streamlit)
- **Framework**: Streamlit with custom CSS
- **Features**: Multi-page navigation, real-time analysis
- **Visualization**: Plotly charts, progress tracking
- **Reports**: PDF generation with fpdf2

### Analytics Engine
- **Photo Quality**: OpenCV-based assessment (brightness, blur, skin detection)
- **Progress Tracking**: Historical data analysis with trend computation
- **ML Predictions**: Scikit-learn linear/polynomial regression
- **Scoring Algorithm**: Multi-factor weighted health scoring (0-100)

## 🛠️ Development

### Adding New Features

1. **Backend**: Extend `gateway/adapters/acne_adapter.py`
2. **Frontend**: Modify `app.py` or add new pages
3. **Models**: Place new `.pt` files in `models/` directory
4. **Styling**: Update `src/components/style.css`

### Testing

```bash
# Test backend
curl -X POST "http://localhost:8000/infer" \
  -H "Content-Type: application/json" \
  -d '{"request_id": "test", "image_b64": "..."}'

# Test frontend
streamlit run app.py --server.port 8502
```

## 📋 Requirements

### Core Dependencies
- `streamlit` - Web interface
- `fastapi` - Backend API
- `uvicorn` - ASGI server
- `ultralytics` - YOLOv8 implementation
- `opencv-python` - Image processing
- `pillow` - Image handling
- `plotly` - Interactive charts
- `pandas` - Data manipulation
- `scikit-learn` - ML predictions
- `fpdf2` - PDF generation

### System Requirements
- **RAM**: Minimum 4GB, Recommended 8GB+
- **Storage**: 2GB free space for models
- **Network**: Internet connection for initial model download

## 🚨 Troubleshooting

### Common Issues

**1. Model Not Found**
```
Error: Model not found at models/acne_detection.pt
```
**Solution**: Place your custom YOLOv8 model in the correct directory

**2. Port Already in Use**
```
Error: Port 8000 is already in use
```
**Solution**: Kill existing processes or use different ports:
```bash
# Kill processes
taskkill /f /im python.exe  # Windows
pkill -f python             # Linux/Mac

# Or use different ports
uvicorn gateway.main:app --port 8001
streamlit run app.py --server.port 8503
```

**3. Gateway Connection Failed**
```
Error: Connection error
```
**Solution**: Ensure backend is running before starting frontend

**4. Unicode Encoding Errors**
```
UnicodeEncodeError: 'charmap' codec can't encode character
```
**Solution**: Already fixed in current version (removed emoji characters)

## 📞 Support

For issues and questions:
- Create an issue on [GitHub](https://github.com/melikeZaman04/AcneSkin/issues)
- Check troubleshooting section above
- Verify all dependencies are installed correctly

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- YOLOv8 by Ultralytics
- Roboflow for dataset tools
- Streamlit for rapid web development
- FastAPI for high-performance APIs

---

**⚠️ Medical Disclaimer**: This AI tool is for educational and research purposes only. Always consult qualified dermatologists for professional medical diagnosis and treatment.