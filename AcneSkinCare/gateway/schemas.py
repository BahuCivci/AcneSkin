from pydantic import BaseModel, Field
from typing import Dict, List
from datetime import datetime

class PreprocessOptions(BaseModel):
    face_align: bool = False
    illum_norm: bool = False

class InferIn(BaseModel):
    request_id: str
    image_b64: str
    capture_ts: datetime
    preprocess: PreprocessOptions = Field(default_factory=PreprocessOptions)

class Detection(BaseModel):
    label: str
    x: int
    y: int
    w: int
    h: int
    confidence: float

class InferOut(BaseModel):
    model_name: str
    model_version: str
    latency_ms: int
    skin_score: float
    scores: Dict[str, float]
    detections: List[Detection]
    warnings: List[str]
    trace_id: str
