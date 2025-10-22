from typing import Dict, List, Optional, Tuple
import base64
import hashlib
import numpy as np
import os
import io
from PIL import Image
from .base import BaseAdapter

# Import YOLO acne detection functions
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'acne-ai-skin-digital-twin'))
from src.ai.acne_yolo import detect_acne_in_image, get_acne_detector, ACNE_CLASSES, CLASS_DESCRIPTIONS

class AcneAdapter(BaseAdapter):
    """
    YOLO-based acne detection adapter
    """

    def __init__(self):
        self.model_path = os.path.join(
            os.path.dirname(__file__),
            '..', '..',
            'acne-ai-skin-digital-twin',
            'models',
            'acne_detection.pt'
        )
        self.detector = None
        self._load_model()

    def _load_model(self):
        """Load the YOLO acne detection model"""
        try:
            if os.path.exists(self.model_path):
                # Initialize the YOLO detector with our custom model
                from src.ai.acne_yolo import AcneYOLODetector
                self.detector = AcneYOLODetector(self.model_path)
                print(f"YOLO acne model loaded from {self.model_path}")
            else:
                print(f"Model not found at {self.model_path}")
                # Use base detector without custom model
                self.detector = get_acne_detector()
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            # Fallback to base detector
            self.detector = get_acne_detector()

    def infer(
        self,
        image_bytes: bytes,
        preprocess: Dict,
        request_id: str,
        capture_ts: Optional[str] = None,
        image_size: Optional[Tuple[int, int]] = (512, 512)
    ) -> Dict:
        """YOLO acne detection inference"""

        if self.detector is None:
            # Fallback to mock if detector not available
            return self._mock_inference(request_id, image_bytes, preprocess, capture_ts, image_size)

        try:
            # Set confidence threshold if provided
            confidence_threshold = preprocess.get("confidence_threshold", 0.25)
            self.detector.confidence_threshold = confidence_threshold

            # Use YOLO detector for acne detection
            result = self.detector.detect_acne(image_bytes)

            detections = result.get('detections', [])
            total_count = result.get('total_count', 0)
            severity_score = result.get('severity_score', 0)
            recommendations = result.get('recommendations', [])

            # Calculate skin score (inverse of severity for compatibility)
            skin_score = max(0, 100 - severity_score)

            # Convert YOLO detections to gateway format
            formatted_detections = []
            for det in detections:
                bbox = det.get('bbox', [0, 0, 50, 50])
                formatted_detections.append({
                    "label": det.get('class', 'unknown'),
                    "x": int(bbox[0]),
                    "y": int(bbox[1]),
                    "w": int(bbox[2]),
                    "h": int(bbox[3]),
                    "confidence": float(det.get('confidence', 0.0))
                })

            # Generate warnings based on detections
            warnings = self._generate_acne_warnings(detections, severity_score)

            return {
                "model_name": "yolo-acne-detection",
                "model_version": "1.0.0",
                "skin_score": skin_score,
                "scores": {
                    "total_acne_count": total_count,
                    "severity_score": severity_score,
                    "acne_types": {det['class']: det['confidence'] for det in detections}
                },
                "detections": formatted_detections,
                "warnings": warnings + recommendations,
                "meta": {
                    "preprocess": preprocess,
                    "capture_ts": capture_ts,
                    "is_simulation": result.get('is_simulation', False)
                }
            }

        except Exception as e:
            print(f"❌ YOLO inference error: {e}")
            # Fallback to mock
            return self._mock_inference(request_id, image_bytes, preprocess, capture_ts, image_size)

    def _generate_acne_warnings(self, detections: List[Dict], severity_score: float) -> List[str]:
        """Generate acne-specific warnings based on detections"""
        warnings = []

        # Count acne types
        acne_counts = {}
        for det in detections:
            acne_type = det.get('class', 'unknown')
            acne_counts[acne_type] = acne_counts.get(acne_type, 0) + 1

        # Severity-based warnings
        if severity_score > 80:
            warnings.append("🚨 Severe acne detected - Consider dermatological consultation")
        elif severity_score > 60:
            warnings.append("⚠️ Moderate acne - Professional treatment may be needed")
        elif severity_score > 40:
            warnings.append("⚠️ Mild to moderate acne - Consider acne treatment products")

        # Specific acne type warnings
        if 'nodules' in acne_counts or 'cyst' in acne_counts:
            warnings.append("⚠️ Cystic/nodular acne detected - Professional treatment recommended")

        if 'pustules' in acne_counts and acne_counts['pustules'] > 3:
            warnings.append("⚠️ Multiple pustules detected - Monitor for signs of infection")

        # Total count warnings
        total_count = len(detections)
        if total_count > 10:
            warnings.append("⚠️ High acne lesion count - Consider comprehensive treatment plan")

        # General disclaimer
        warnings.append("ℹ️ This is AI assistance only - Consult dermatologist for professional diagnosis")

        return warnings

    def _mock_inference(self, request_id: str, image_bytes: bytes, preprocess: Dict,
                       capture_ts: Optional[str], image_size: Optional[Tuple[int, int]]) -> Dict:
        """Fallback mock inference when model is not available"""
        h = hashlib.sha256()
        h.update(request_id.encode("utf-8"))
        h.update(image_bytes[:1024] if image_bytes else b"")
        seed_int = int.from_bytes(h.digest()[:8], "big") & 0xFFFFFFFF
        rng = np.random.default_rng(seed_int)

        # Mock acne detections
        acne_classes = list(ACNE_CLASSES.keys())[:-1]  # Exclude healthy_skin
        num_detections = int(rng.uniform(1, 6))

        mock_detections = []
        for i in range(num_detections):
            acne_type = rng.choice(acne_classes)
            confidence = float(rng.uniform(0.4, 0.9))

            # Random bounding box
            x = int(rng.uniform(50, 400))
            y = int(rng.uniform(50, 400))
            w = int(rng.uniform(30, 100))
            h = int(rng.uniform(30, 100))

            mock_detections.append({
                "label": ACNE_CLASSES[acne_type],
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "confidence": confidence
            })

        severity_score = float(rng.uniform(20, 80))
        skin_score = max(0, 100 - severity_score)

        return {
            "model_name": "yolo-acne-mock",
            "model_version": "0.1.0",
            "skin_score": skin_score,
            "scores": {
                "total_acne_count": len(mock_detections),
                "severity_score": severity_score,
                "acne_types": {det['label']: det['confidence'] for det in mock_detections}
            },
            "detections": mock_detections,
            "warnings": ["ℹ️ Using mock acne detection - Install real model for accurate results"],
            "meta": {
                "preprocess": preprocess,
                "capture_ts": capture_ts,
                "is_simulation": True
            }
        }
