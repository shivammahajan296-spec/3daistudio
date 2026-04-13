from __future__ import annotations

import re
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from agents.workflow import import_step_model, infer_image_prompt, run_generation


BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR / "outputs"
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

OUTPUTS_DIR.mkdir(exist_ok=True)
IMPORTS_DIR = OUTPUTS_DIR / "imports"
IMPORTS_DIR.mkdir(exist_ok=True)

app = FastAPI(title="3D Studio", version="1.0.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/outputs", StaticFiles(directory=OUTPUTS_DIR), name="outputs")
templates = Jinja2Templates(directory=TEMPLATES_DIR)


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=3, max_length=2000)
    openrouter_api_key: str | None = Field(default=None, max_length=300)
    llm_provider: str | None = Field(default="openrouter", max_length=30)
    llm_api_key: str | None = Field(default=None, max_length=500)
    previous_code: str | None = Field(default=None, max_length=50000)
    previous_run_id: str | None = Field(default=None, max_length=80)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/generate")
async def generate(request: GenerateRequest):
    try:
        api_key = normalize_api_key(request.llm_api_key or request.openrouter_api_key)
        return run_generation(
            request.prompt,
            OUTPUTS_DIR,
            llm_provider=normalize_provider(request.llm_provider),
            llm_api_key=api_key,
            previous_code=request.previous_code,
            previous_run_id=request.previous_run_id,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Generation failed: {exc}") from exc


def normalize_api_key(api_key: str | None) -> str | None:
    if not api_key:
        return None
    cleaned = api_key.strip().strip('"').strip("'")
    return cleaned or None


def normalize_provider(provider: str | None) -> str:
    cleaned = (provider or "openrouter").strip().lower()
    return cleaned if cleaned in {"openrouter", "gemini", "gpt", "claude"} else "openrouter"


def safe_filename(filename: str) -> str:
    name = Path(filename).name
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    return cleaned or "uploaded.step"


@app.post("/chat")
async def chat(request: GenerateRequest):
    return await generate(request)


@app.post("/edit")
async def edit(request: GenerateRequest):
    if not request.previous_code:
        raise HTTPException(status_code=400, detail="No existing model code was provided to edit.")
    return await generate(request)


@app.post("/import-step")
async def import_step(file: UploadFile = File(...)):
    filename = file.filename or "uploaded.step"
    suffix = Path(filename).suffix.lower()
    if suffix not in {".step", ".stp"}:
        raise HTTPException(status_code=400, detail="Please upload a .step or .stp file.")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded STEP file is empty.")
    if len(contents) > 50 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="STEP file is too large. Limit is 50 MB.")

    import_id = uuid.uuid4().hex
    import_dir = IMPORTS_DIR / import_id
    import_dir.mkdir(parents=True, exist_ok=True)
    step_path = import_dir / safe_filename(filename)
    step_path.write_bytes(contents)

    try:
        return import_step_model(step_path, OUTPUTS_DIR, original_filename=filename)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"STEP import failed: {exc}") from exc


@app.post("/infer-image")
async def infer_image(
    file: UploadFile = File(...),
    openrouter_api_key: str | None = Form(default=None),
    llm_api_key: str | None = Form(default=None),
    llm_provider: str | None = Form(default="openrouter"),
):
    filename = file.filename or "uploaded-image.png"
    suffix = Path(filename).suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg", ".webp"}:
        raise HTTPException(status_code=400, detail="Please upload a PNG, JPG, JPEG, or WEBP image.")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")
    if len(contents) > 12 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image is too large. Limit is 12 MB.")

    import_id = uuid.uuid4().hex
    import_dir = IMPORTS_DIR / import_id
    import_dir.mkdir(parents=True, exist_ok=True)
    image_path = import_dir / safe_filename(filename)
    image_path.write_bytes(contents)

    try:
        return infer_image_prompt(
            image_path,
            llm_provider=normalize_provider(llm_provider),
            llm_api_key=normalize_api_key(llm_api_key or openrouter_api_key),
            original_filename=filename,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Image inference failed: {exc}") from exc


@app.get("/models/{run_id}")
async def model_metadata(run_id: str):
    run_dir = OUTPUTS_DIR / run_id
    if not run_dir.exists() or not run_dir.is_dir():
        raise HTTPException(status_code=404, detail="Run not found.")

    attempts = sorted(
        [p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("attempt-")],
        key=lambda p: int(p.name.split("-")[-1]),
    )
    if not attempts:
        raise HTTPException(status_code=404, detail="No attempts found for run.")

    latest = attempts[-1]
    step_path = latest / "model.step"
    stl_path = latest / "model.stl"
    code_path = latest / "generated_model.py"
    return {
        "run_id": run_id,
        "attempt": latest.name,
        "files": {
            "step": f"/outputs/{run_id}/{latest.name}/model.step" if step_path.exists() else None,
            "stl": f"/outputs/{run_id}/{latest.name}/model.stl" if stl_path.exists() else None,
            "code": f"/outputs/{run_id}/{latest.name}/generated_model.py" if code_path.exists() else None,
        },
        "sizes": {
            "step": step_path.stat().st_size if step_path.exists() else 0,
            "stl": stl_path.stat().st_size if stl_path.exists() else 0,
            "code": code_path.stat().st_size if code_path.exists() else 0,
        },
    }
