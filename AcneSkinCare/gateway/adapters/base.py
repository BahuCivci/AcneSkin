from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, List

class BaseAdapter(ABC):
    """
    Abstract adapter interface for inference adapters.
    Implementations must provide an `infer` method that returns a serializable dict.
    """

    @abstractmethod
    def infer(
        self,
        image_bytes: bytes,
        preprocess: Dict,
        request_id: str,
        capture_ts: Optional[str] = None,
        image_size: Optional[Tuple[int, int]] = None
    ) -> Dict:
        """
        Run inference on provided image bytes and return a dictionary with keys:
        - model_name, model_version, skin_score, scores (dict), detections (list), warnings (list)
        """
        raise NotImplementedError
