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

HUNYUAN_VISIBLE_DEVICES = os.environ.get("HUNYUAN_CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", HUNYUAN_VISIBLE_DEVICES)

import torch

from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

HUNYUAN_ROOT = PROJECT_ROOT
MODEL_PATH = os.environ.get("HY3D_MODEL_PATH", "tencent/Hunyuan3D-2.1")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16
LOAD_ON_STARTUP = os.environ.get("HUNYUAN_LOAD_ON_STARTUP", "1") == "1"
STAGED_EXPORT = os.environ.get("HUNYUAN_STAGED_EXPORT", "1") == "1"
IDLE_OFFLOAD_SECS = float(os.environ.get("HUNYUAN_IDLE_OFFLOAD_SECS", "60"))
KEEP_ON_GPU_RAW = os.environ.get("HUNYUAN_KEEP_ON_GPU", "model,conditioner")

PIPELINE: Optional[Hunyuan3DDiTFlowMatchingPipeline] = None
PIPELINE_LOCK = threading.Lock()
IDLE_OFFLOAD_TIMER: Optional[threading.Timer] = None
IDLE_OFFLOAD_LOCK = threading.Lock()

app = FastAPI(title="Hunyuan3D Shape Service")


class GenerateRequest(BaseModel):
    image_base64: str


class GenerateResponse(BaseModel):
    status: str
    mesh_base64: str


def _parse_keep_on_gpu(value: str) -> set:
    value = (value or "").strip().lower()
    if not value or value == "all":
        return {"model", "vae", "conditioner"}
    if value == "none":
        return set()
    return {item.strip() for item in value.split(",") if item.strip()}


KEEP_ON_GPU = _parse_keep_on_gpu(KEEP_ON_GPU_RAW)


def _cancel_idle_offload() -> None:
    global IDLE_OFFLOAD_TIMER
    if IDLE_OFFLOAD_SECS <= 0:
        return
    with IDLE_OFFLOAD_LOCK:
        if IDLE_OFFLOAD_TIMER is not None:
            IDLE_OFFLOAD_TIMER.cancel()
            IDLE_OFFLOAD_TIMER = None


def _offload_pipeline_components(keep_on_gpu: set) -> None:
    if PIPELINE is None or DEVICE != "cuda":
        return
    if keep_on_gpu == {"model", "vae", "conditioner"}:
        return
    if not keep_on_gpu:
        PIPELINE.to("cpu")
    else:
        components = {
            "model": PIPELINE.model,
            "vae": PIPELINE.vae,
            "conditioner": PIPELINE.conditioner,
        }
        for name, module in components.items():
            if name in keep_on_gpu:
                module.to(DEVICE)
            else:
                module.to("cpu")
    torch.cuda.empty_cache()


def _schedule_idle_offload() -> None:
    global IDLE_OFFLOAD_TIMER
    if IDLE_OFFLOAD_SECS <= 0:
        return

    def _do_offload() -> None:
        if PIPELINE is None or DEVICE != "cuda":
            return
        if not PIPELINE_LOCK.acquire(blocking=False):
            _schedule_idle_offload()
            return
        try:
            _offload_pipeline_components(KEEP_ON_GPU)
        finally:
            PIPELINE_LOCK.release()

    with IDLE_OFFLOAD_LOCK:
        if IDLE_OFFLOAD_TIMER is not None:
            IDLE_OFFLOAD_TIMER.cancel()
        IDLE_OFFLOAD_TIMER = threading.Timer(IDLE_OFFLOAD_SECS, _do_offload)
        IDLE_OFFLOAD_TIMER.daemon = True
        IDLE_OFFLOAD_TIMER.start()


def _offload_after_diffusion() -> None:
    if PIPELINE is None or DEVICE != "cuda":
        return
    PIPELINE.model.to("cpu")
    PIPELINE.conditioner.to("cpu")
    torch.cuda.empty_cache()


def _restore_after_export() -> None:
    if PIPELINE is None or DEVICE != "cuda":
        return
    if "model" in KEEP_ON_GPU:
        PIPELINE.model.to(DEVICE)
    if "conditioner" in KEEP_ON_GPU:
        PIPELINE.conditioner.to(DEVICE)


def _load_pipeline() -> None:
    global PIPELINE
    if PIPELINE is not None:
        return
    PIPELINE = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        MODEL_PATH,
        device=DEVICE,
        dtype=DTYPE,
    )
    _schedule_idle_offload()


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
    if LOAD_ON_STARTUP:
        _load_pipeline()


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    _load_pipeline()
    if PIPELINE is None:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")

    image = _decode_base64_image(req.image_base64)
    with PIPELINE_LOCK:
        try:
            with torch.no_grad():
                _cancel_idle_offload()
                PIPELINE.to(DEVICE)
                if STAGED_EXPORT and DEVICE == "cuda":
                    latents = PIPELINE(
                        image=image,
                        output_type="latent",
                    )
                    _offload_after_diffusion()
                    mesh = PIPELINE._export(
                        latents,
                        output_type="trimesh",
                        box_v=1.01,
                        mc_level=0.0,
                        num_chunks=8000,
                        octree_resolution=384,
                        mc_algo=None,
                        enable_pbar=True,
                    )
                    _restore_after_export()
                else:
                    mesh = PIPELINE(image=image)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        finally:
            _schedule_idle_offload()

    if isinstance(mesh, list):
        mesh = mesh[0]
    mesh_base64 = _mesh_to_base64(mesh)
    return {"status": "success", "mesh_base64": mesh_base64}


if __name__ == "__main__":
    host = os.environ.get("HUNYUAN_SERVICE_HOST", "0.0.0.0")
    port = int(os.environ.get("HUNYUAN_SERVICE_PORT", "9084"))
    config = uvicorn.Config(app=app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)
    server.run()
