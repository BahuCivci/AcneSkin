from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from time import perf_counter
from .schemas import InferIn, InferOut
from .adapters.acne_adapter import AcneAdapter
import base64
import io
from PIL import Image

app = FastAPI(title="AcneAI Inference Gateway", docs_url=None, redoc_url=None)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

adapter = AcneAdapter()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/ready")
def ready():
    try:
        ok = isinstance(adapter, AcneAdapter)
        return {"ready": ok}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/infer")
def infer(payload: InferIn):
    t0 = perf_counter()

    # Decode image base64 (robust)
    try:
        img_bytes = _decode_base64_image(payload.image_b64)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid image_b64: {e}")

    # Try to get image size (optional, adapter can handle None)
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        image_size = img.size  # (width, height)
    except Exception:
        image_size = (512, 512)

    # Call adapter.infer (use infer, not analyze)
    try:
        result = adapter.infer(
            image_bytes=img_bytes,
            preprocess=payload.preprocess.dict(),
            request_id=payload.request_id,
            capture_ts=payload.capture_ts.isoformat() if hasattr(payload.capture_ts, "isoformat") else str(payload.capture_ts),
            image_size=image_size
        )
    except AttributeError:
        raise HTTPException(status_code=500, detail="Adapter method 'infer' not implemented")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    latency_ms = int((perf_counter() - t0) * 1000)

    # Map adapter result to response model safely
    response = {
        "model_name": result.get("model_name", "unknown"),
        "model_version": result.get("model_version", "0.0"),
        "latency_ms": latency_ms,
        "skin_score": result.get("skin_score", 0.0),
        "scores": result.get("scores", {}),
        "detections": result.get("detections", []),
        "warnings": result.get("warnings", []),
        "trace_id": payload.request_id
    }

    return response

def _decode_base64_image(b64: str) -> bytes:
    """
    Robust base64 decoder:
    - Accepts data:URI (data:image/...), strips prefix
    - Removes whitespace/newlines
    - Fixes missing '=' padding
    """
    if not isinstance(b64, str) or not b64:
        raise ValueError("empty or non-string image_b64")

    # strip data URI prefix if present
    if b64.startswith("data:"):
        try:
            b64 = b64.split(",", 1)[1]
        except Exception:
            raise ValueError("invalid data URI")

    # remove whitespace/newlines that may come from copy/paste
    b64 = b64.strip().replace("\r", "").replace("\n", "").replace(" ", "")

    # fix padding
    pad = len(b64) % 4
    if pad:
        b64 += "=" * (4 - pad)

    try:
        return base64.b64decode(b64)
    except Exception as e:
        raise ValueError(f"base64 decode error: {e}")