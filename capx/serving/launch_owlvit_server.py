import base64
import io
import json
import logging
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import tyro
import uvicorn
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel
from transformers import (
    Owlv2ForObjectDetection,
    Owlv2Processor,
    OwlViTForObjectDetection,
    OwlViTProcessor,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Global state
_PROC: Any | None = None
_MODEL: Any | None = None
_DEVICE: str = "cuda"
_THRESHOLD: float = 0.05


# --- Helper Functions ---


def _encode_image(image: np.ndarray | Image.Image) -> str:
    if isinstance(image, np.ndarray):
        image_u8 = np.clip(image, 0, 255).astype(np.uint8) if image.dtype != np.uint8 else image
        pil_image = Image.fromarray(image_u8).convert("RGB")
    else:
        pil_image = image

    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def _decode_image(image_base64: str) -> Image.Image:
    image_bytes = base64.b64decode(image_base64)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


# --- API Models ---


class DetectRequest(BaseModel):
    image_base64: str
    texts: list[list[str]] | None = None
    threshold: float | None = None


class Detection(BaseModel):
    label: str
    score: float
    box: list[float]


class DetectResponse(BaseModel):
    detections: list[Detection]


# --- API Endpoints ---


@app.post("/detect", response_model=DetectResponse)
async def detect_endpoint(req: DetectRequest):
    if _MODEL is None or _PROC is None:
        raise HTTPException(status_code=503, detail="Model not initialized")

    try:
        # Decode image
        pil_image = _decode_image(req.image_base64)
        rgb = np.array(pil_image)

        # Use request threshold or default
        threshold = req.threshold if req.threshold is not None else _THRESHOLD
        texts = req.texts if req.texts is not None else [["an object"]]

        # Process
        rgb_u8 = np.clip(rgb, 0, 255).astype(np.uint8) if rgb.dtype != np.uint8 else rgb
        inputs = _PROC(text=texts, images=rgb_u8, return_tensors="pt")
        with torch.no_grad():
            outputs = _MODEL(**{k: v.to(_MODEL.device) for k, v in inputs.items()})

        target_sizes = torch.tensor([rgb_u8.shape[:2]], device=_MODEL.device)
        results = _PROC.post_process_grounded_object_detection(
            outputs=outputs, threshold=threshold, target_sizes=target_sizes
        )

        detections = []
        i = 0
        labels = texts[i]
        for box, score, label in zip(
            results[i]["boxes"], results[i]["scores"], results[i]["labels"], strict=False
        ):
            b = box.detach().to("cpu").numpy().tolist()
            detections.append(
                Detection(
                    label=labels[int(label)],
                    score=float(score.item()),
                    box=[round(x, 2) for x in b],
                )
            )

        return DetectResponse(detections=detections)

    except Exception as e:
        logger.error(f"Error during detection: {e}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")


# --- Entrypoint ---


ModelFamily = Literal["auto", "owlv2", "owlvit"]


def _infer_model_family(model_name: str, model_family: ModelFamily) -> Literal["owlv2", "owlvit"]:
    if model_family != "auto":
        return model_family

    model_name_lower = model_name.lower()
    if "owlv2" in model_name_lower:
        return "owlv2"
    if "owlvit" in model_name_lower or "owl-vit" in model_name_lower:
        return "owlvit"

    config_path = Path(model_name).expanduser() / "config.json"
    if config_path.exists():
        try:
            with config_path.open("r", encoding="utf-8") as f:
                config = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid model config JSON at {config_path}: {e}") from e

        config_model_type = str(config.get("model_type", "")).lower()
        architectures = " ".join(config.get("architectures", [])).lower()
        if "owlv2" in config_model_type or "owlv2" in architectures:
            return "owlv2"
        if "owlvit" in config_model_type or "owlvit" in architectures:
            return "owlvit"

    raise ValueError(
        "Could not infer model family from model_name or local config.json. "
        "Pass --model-family owlv2 or --model-family owlvit."
    )


def main(
    model_name: str = "google/owlv2-large-patch14-ensemble",
    model_family: ModelFamily = "auto",
    local_files_only: bool = False,
    device: str = "cuda",
    port: int = 8117,
    host: str = "127.0.0.1",
    threshold: float = 0.05,
):
    global _PROC, _MODEL, _DEVICE, _THRESHOLD

    _DEVICE = device
    _THRESHOLD = threshold

    logger.info(f"Loading OWL-ViT model: {model_name} on {device}...")

    model_kind = _infer_model_family(model_name, model_family)
    is_v2 = model_kind == "owlv2"
    logger.info(
        "Resolved model family: %s%s",
        model_kind,
        " (local files only)" if local_files_only else "",
    )

    try:
        if is_v2:
            _PROC = Owlv2Processor.from_pretrained(
                model_name, local_files_only=local_files_only
            )
            _MODEL = Owlv2ForObjectDetection.from_pretrained(
                model_name, local_files_only=local_files_only
            )
        else:
            _PROC = OwlViTProcessor.from_pretrained(
                model_name, local_files_only=local_files_only
            )
            _MODEL = OwlViTForObjectDetection.from_pretrained(
                model_name, local_files_only=local_files_only
            )

        _MODEL = _MODEL.to(device)
        _MODEL.eval()

        logger.info(f"OWL-ViT model loaded on {device}. Starting server...")
        uvicorn.run(app, host=host, port=port)

    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


if __name__ == "__main__":
    tyro.cli(main)
