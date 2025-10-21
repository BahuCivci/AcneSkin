from typing import Dict, List, Optional, Tuple
import base64
import hashlib
import numpy as np
from PIL import Image
from .base import BaseAdapter

class AcneAdapter(BaseAdapter):
    """
    Mock deterministic adapter that returns reproducible pseudo-random scores
    based on the request_id + a hash of the image bytes.
    """

    def _make_rng(self, request_id: str, image_bytes: bytes):
        h = hashlib.sha256()
        h.update(request_id.encode("utf-8"))
        h.update(image_bytes[:1024] if image_bytes else b"")
        seed_int = int.from_bytes(h.digest()[:8], "big") & 0xFFFFFFFF
        return np.random.default_rng(seed_int)

    def infer(
        self,
        image_bytes: bytes,
        preprocess: Dict,
        request_id: str,
        capture_ts: Optional[str] = None,
        image_size: Optional[Tuple[int, int]] = (512, 512)
    ) -> Dict:
        rng = self._make_rng(request_id, image_bytes)

        acne = float(rng.uniform(0.0, 1.0))
        pores = float(rng.uniform(0.0, 1.0))
        pigmentation = float(rng.uniform(0.0, 1.0))
        hydration = float(rng.uniform(0.0, 1.0))

        score_components = (
            0.30 * (1.0 - acne) +
            0.25 * (1.0 - pores) +
            0.20 * (1.0 - pigmentation) +
            0.25 * hydration
        )
        skin_score = round(float(score_components * 100.0), 2)

        num_detections = int(rng.integers(0, 4))
        width, height = image_size if image_size else (512, 512)
        detections: List[Dict] = []
        for i in range(num_detections):
            w = int(rng.integers(max(1, width // 10), max(2, width // 3)))
            h = int(rng.integers(max(1, height // 10), max(2, height // 3)))
            x = int(rng.integers(0, max(1, width - w)))
            y = int(rng.integers(0, max(1, height - h)))
            confidence = float(round(rng.uniform(0.3, 0.99), 3))
            detections.append({
                "label": "lesion",
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "confidence": confidence
            })

        warnings: List[str] = []
        if acne > 0.75:
            warnings.append("High acne severity detected")
        if hydration < 0.25:
            warnings.append("Low skin hydration")
        if pigmentation > 0.7:
            warnings.append("Significant pigmentation detected")

        return {
            "model_name": "acne-mock-adapter",
            "model_version": "0.1.0",
            "skin_score": skin_score,
            "scores": {
                "acne": round(acne, 3),
                "pores": round(pores, 3),
                "pigmentation": round(pigmentation, 3),
                "hydration": round(hydration, 3),
            },
            "detections": detections,
            "warnings": warnings,
            "meta": {
                "preprocess": preprocess,
                "capture_ts": capture_ts
            }
        }
