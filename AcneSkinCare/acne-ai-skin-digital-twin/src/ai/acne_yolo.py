"""
YOLO-based acne detection wrapper
Uses YOLOv8 base model with custom acne detection logic
"""

import torch
import numpy as np
from ultralytics import YOLO
from PIL import Image
import io

# Acne classes based on the trained Roboflow model (4 classes)
ACNE_CLASSES = {
    0: 'comedone',
    1: 'nodules',
    2: 'papules',
    3: 'pustules'
}

CLASS_DESCRIPTIONS = {
    'comedone': 'Comedone (Blackhead/Whitehead)',
    'nodules': 'Nodules (Deep Inflammatory Lesions)',
    'papules': 'Papules (Inflamed Bumps)',
    'pustules': 'Pustules (Pus-filled Lesions)'
}

class AcneYOLODetector:
    def __init__(self, model_path=None):
        """Initialize YOLO-based acne detector"""
        try:
            if model_path and model_path.endswith('.pt'):
                # Try to load custom model first
                self.model = YOLO(model_path)
                print(f"Custom model loaded: {model_path}")
            else:
                # Fallback to base YOLOv8
                self.model = YOLO('yolov8n.pt')
                print("Using base YOLOv8 with acne detection simulation")
                self.use_simulation = True
        except Exception as e:
            print(f"Model loading error: {e}")
            # Ultimate fallback
            self.model = YOLO('yolov8n.pt')
            self.use_simulation = True

    def detect_acne(self, image_bytes):
        """Detect acne in image and return results"""
        try:
            # Load image
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

            # Run YOLO detection
            results = self.model(image, verbose=False)

            print(f"YOLO Results: {len(results)} result objects")
            for i, result in enumerate(results):
                if hasattr(result, 'boxes') and result.boxes is not None:
                    print(f"   Result {i}: {len(result.boxes)} detections")
                    if len(result.boxes) > 0:
                        print(f"   Confidences: {result.boxes.conf.cpu().numpy()}")
                        print(f"   Classes: {result.boxes.cls.cpu().numpy()}")
                else:
                    print(f"   Result {i}: No boxes detected")

            if hasattr(self, 'use_simulation') and self.use_simulation:
                print("Using simulation mode")
                # Simulate acne detection results
                return self._simulate_acne_detection(image, results)
            else:
                print("Using real model results")
                # Real custom model results
                return self._process_real_results(results, image)

        except Exception as e:
            print(f"Detection error: {e}")
            import traceback
            traceback.print_exc()
            return self._empty_result()

    def _simulate_acne_detection(self, image, yolo_results):
        """Simulate acne detection based on general object detection"""
        detections = []

        # Get image dimensions
        width, height = image.size

        # Simulate different acne types based on image characteristics
        # This is a demonstration - in real scenario, you'd use actual trained model

        # Simulate finding some acne lesions
        import random
        random.seed(hash(str(image.size)) % 1000)  # Deterministic based on image

        num_detections = random.randint(2, 8)

        for i in range(num_detections):
            # Random acne type from our 4 classes
            acne_type = random.choice(list(ACNE_CLASSES.keys()))
            confidence = random.uniform(0.4, 0.9)

            # Random location (but avoid edges)
            x = random.randint(int(width * 0.1), int(width * 0.9))
            y = random.randint(int(height * 0.1), int(height * 0.9))
            w = random.randint(20, 80)
            h = random.randint(20, 80)

            detections.append({
                'class': ACNE_CLASSES[acne_type],
                'confidence': confidence,
                'bbox': [x, y, w, h],
                'description': CLASS_DESCRIPTIONS[ACNE_CLASSES[acne_type]]
            })

        # Calculate acne severity score
        severity_score = self._calculate_severity(detections)

        return {
            'detections': detections,
            'total_count': len(detections),
            'severity_score': severity_score,
            'recommendations': self._get_recommendations(severity_score),
            'is_simulation': True
        }

    def _calculate_severity(self, detections):
        """Calculate acne severity score (0-100)"""
        if not detections:
            return 0

        severity_weights = {
            'comedone': 1,
            'papules': 2,
            'pustules': 3,
            'nodules': 4
        }

        total_score = 0
        for det in detections:
            weight = severity_weights.get(det['class'], 1)
            total_score += weight * det['confidence']

        # Normalize to 0-100 scale
        max_possible = len(detections) * 5  # Max weight is 5
        severity = min(100, (total_score / max_possible) * 100)

        return round(severity, 1)

    def _get_recommendations(self, severity_score):
        """Get treatment recommendations based on severity"""
        if severity_score < 20:
            return [
                "Maintain good skincare routine",
                "Use gentle cleanser twice daily",
                "Consider non-comedogenic moisturizer"
            ]
        elif severity_score < 50:
            return [
                "Consider over-the-counter acne treatments",
                "Use salicylic acid or benzoyl peroxide",
                "Maintain consistent skincare routine",
                "Avoid touching face frequently"
            ]
        elif severity_score < 80:
            return [
                "Consult dermatologist for treatment plan",
                "May need prescription medications",
                "Consider topical retinoids",
                "Professional extraction may be needed"
            ]
        else:
            return [
                "Seek immediate dermatological consultation",
                "May require systemic medications",
                "Consider professional acne treatments",
                "Monitor for scarring prevention"
            ]

    def _process_real_results(self, results, image):
        """Process real YOLO model results"""
        detections = []

        # Get image dimensions
        width, height = image.size

        # Process YOLO results
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes
                for i in range(len(boxes)):
                    # Get detection info
                    box = boxes.xyxy[i].cpu().numpy()  # x1, y1, x2, y2
                    conf = float(boxes.conf[i].cpu().numpy())
                    cls = int(boxes.cls[i].cpu().numpy())

                    # Skip low confidence detections (use dynamic threshold)
                    confidence_threshold = getattr(self, 'confidence_threshold', 0.25)
                    if conf < confidence_threshold:
                        print(f"   Skipping low confidence detection: {conf:.3f} < {confidence_threshold}")
                        continue

                    # Convert to x, y, w, h format
                    x1, y1, x2, y2 = box
                    x = int(x1)
                    y = int(y1)
                    w = int(x2 - x1)
                    h = int(y2 - y1)

                    # Get class name
                    class_name = ACNE_CLASSES.get(cls, 'unknown')
                    print(f"   Valid detection: class={class_name}, conf={conf:.3f}, bbox=[{x},{y},{w},{h}]")

                    detections.append({
                        'class': class_name,
                        'confidence': conf,
                        'bbox': [x, y, w, h],
                        'description': CLASS_DESCRIPTIONS.get(class_name, class_name)
                    })

        # Calculate acne severity score
        severity_score = self._calculate_severity(detections)

        return {
            'detections': detections,
            'total_count': len(detections),
            'severity_score': severity_score,
            'recommendations': self._get_recommendations(severity_score),
            'is_simulation': False
        }

    def _empty_result(self):
        """Return empty result on error"""
        return {
            'detections': [],
            'total_count': 0,
            'severity_score': 0,
            'recommendations': ["Unable to analyze image"],
            'error': True
        }

# Global detector instance
_detector = None

def get_acne_detector():
    """Get global detector instance"""
    global _detector
    if _detector is None:
        _detector = AcneYOLODetector()
    return _detector

def detect_acne_in_image(image_bytes):
    """Convenience function for acne detection"""
    detector = get_acne_detector()
    return detector.detect_acne(image_bytes)# Updated
