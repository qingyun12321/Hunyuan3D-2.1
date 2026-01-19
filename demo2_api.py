import argparse
import base64
import io
import os
import sys
import threading
import uuid

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import uvicorn
from PIL import Image
from pydantic import BaseModel
from typing import Optional

# Ensure local packages are on sys.path
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO_ROOT, "hy3dshape"))

from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

MODEL_PATH = os.environ.get("HY3D_MODEL_PATH", "tencent/Hunyuan3D-2.1")
SAVE_DIR = os.environ.get("HY3D_SAVE_DIR", "./demo2_outputs")

PIPELINE = None
PIPELINE_LOCK = threading.Lock()

app = FastAPI(title="Hunyuan3D Demo2 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def setup_pipeline(model_path: str, save_dir: str) -> None:
    global PIPELINE, MODEL_PATH, SAVE_DIR
    MODEL_PATH = model_path
    SAVE_DIR = save_dir
    os.makedirs(SAVE_DIR, exist_ok=True)
    if PIPELINE is None:
        PIPELINE = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(MODEL_PATH)


@app.on_event("startup")
def _startup() -> None:
    setup_pipeline(MODEL_PATH, SAVE_DIR)


@app.get("/health")
def health_check():
    return {"status": "ok"}


class GenerateRequest(BaseModel):
    image_base64: str
    precision: Optional[str] = "standard"


def _load_image_from_base64(data: str) -> Image.Image:
    if "," in data:
        data = data.split(",", 1)[1]
    try:
        image_bytes = base64.b64decode(data)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid base64 image") from exc
    try:
        return Image.open(io.BytesIO(image_bytes))
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Failed to read image") from exc


@app.post("/generate")
async def generate(image: UploadFile = File(...)):
    if PIPELINE is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image uploads are supported")

    try:
        image_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(image_bytes))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read image: {exc}") from exc

    with PIPELINE_LOCK:
        mesh = PIPELINE(image=pil_image)[0]

    output_name = f"{uuid.uuid4().hex}.glb"
    output_path = os.path.join(SAVE_DIR, output_name)
    mesh.export(output_path)
    return FileResponse(output_path, filename=output_name, media_type="model/gltf-binary")


@app.post("/generate_3d")
async def generate_3d(req: GenerateRequest):
    if PIPELINE is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    pil_image = _load_image_from_base64(req.image_base64)

    with PIPELINE_LOCK:
        mesh = PIPELINE(image=pil_image)[0]

    output_name = f"{uuid.uuid4().hex}.glb"
    output_path = os.path.join(SAVE_DIR, output_name)
    mesh.export(output_path)

    with open(output_path, "rb") as f:
        model_base64 = base64.b64encode(f.read()).decode("utf-8")

    return {"status": "success", "model_data": model_base64, "format": "glb"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=10083)
    parser.add_argument("--model_path", type=str, default=MODEL_PATH)
    parser.add_argument("--save_dir", type=str, default=SAVE_DIR)
    args = parser.parse_args()

    setup_pipeline(args.model_path, args.save_dir)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
