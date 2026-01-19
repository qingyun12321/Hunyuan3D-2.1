import base64
import io
import os
import sys
import tempfile
import threading
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
import uvicorn

API_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(API_ROOT, ".."))
HY3DSHAPE_ROOT = os.path.join(PROJECT_ROOT, "hy3dshape")
for path in (HY3DSHAPE_ROOT, PROJECT_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

HUNYUAN_VISIBLE_DEVICES = os.environ.get("HUNYUAN_CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", HUNYUAN_VISIBLE_DEVICES)

import torch

from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

HUNYUAN_ROOT = PROJECT_ROOT
MODEL_PATH = os.environ.get("HY3D_MODEL_PATH", "tencent/Hunyuan3D-2.1")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16

PIPELINE: Optional[Hunyuan3DDiTFlowMatchingPipeline] = None
PIPELINE_LOCK = threading.Lock()

app = FastAPI(title="Hunyuan3D Shape Service")


class GenerateRequest(BaseModel):
    image_base64: str


class GenerateResponse(BaseModel):
    status: str
    mesh_base64: str


def _load_pipeline() -> None:
    global PIPELINE
    if PIPELINE is not None:
        return
    PIPELINE = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        MODEL_PATH,
        device=DEVICE,
        dtype=DTYPE,
    )


def _decode_base64_image(data: str) -> Image.Image:
    try:
        if "," in data:
            data = data.split(",", 1)[1]
        raw = base64.b64decode(data)
        image = Image.open(io.BytesIO(raw))
        image.load()
        return image
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image base64: {exc}") from exc


def _mesh_to_base64(mesh) -> str:
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as tmp_file:
            tmp_path = tmp_file.name
        mesh.export(tmp_path)
        with open(tmp_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode("utf-8")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.on_event("startup")
def _startup() -> None:
    _load_pipeline()


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    if PIPELINE is None:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")

    image = _decode_base64_image(req.image_base64)
    with PIPELINE_LOCK:
        try:
            with torch.no_grad():
                mesh = PIPELINE(image=image)[0]
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    mesh_base64 = _mesh_to_base64(mesh)
    return {"status": "success", "mesh_base64": mesh_base64}


if __name__ == "__main__":
    host = os.environ.get("HUNYUAN_SERVICE_HOST", "0.0.0.0")
    port = int(os.environ.get("HUNYUAN_SERVICE_PORT", "9084"))
    config = uvicorn.Config(app=app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)
    server.run()
