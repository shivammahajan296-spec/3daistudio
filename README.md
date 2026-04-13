# 3daistudio

A full-stack text-to-CAD app built with FastAPI, LangGraph, OpenRouter/LLM Foundry providers, CadQuery, and a Three.js preview.

## What It Does

- Accepts a product prompt in a chat-style UI.
- Uses a LangGraph pipeline to plan geometry, generate CadQuery code, execute it, validate it, repair failures, and package the result.
- Exports a validated STEP file for CAD workflows.
- Exports an STL preview mesh for the browser viewer.
- Shows attempt logs, generated CadQuery code, success/failure state, and download links.

## Project Structure

```text
app.py
agents/
  workflow.py
cad/
  executor.py
  examples.py
templates/
  index.html
static/
  css/styles.css
  js/app.js
outputs/
requirements.txt
```

## Local Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENROUTER_API_KEY="your-openrouter-key"
uvicorn app:app --reload
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000).

You can also paste the OpenRouter key directly into the UI. The browser stores it in `localStorage` and sends it with each generation request. The backend still supports `OPENROUTER_API_KEY` as the production-friendly default.

The default model is `arcee-ai/trinity-large-preview:free`. To use another OpenRouter model:

```bash
export OPENROUTER_MODEL="openai/gpt-4o-mini"
```

## First Prompt

```text
create a simple cosmetic jar with cylindrical body and lid
```

The app includes a deterministic local CadQuery fallback for this jar prompt, so the first example can still run when `OPENROUTER_API_KEY` is not set or the model call fails. With an API key, the pipeline uses OpenRouter through the OpenAI Python client:

```python
OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)
```

## API

### `POST /generate`

Request:

```json
{
  "prompt": "create a simple cosmetic jar with cylindrical body and lid",
  "openrouter_api_key": "sk-or-v1-..."
}
```

Response fields include:

- `status`
- `attempt_count`
- `plan`
- `code`
- `validation`
- `logs`
- `files.step`
- `files.stl`
- `paths`
- `metadata`

### `POST /chat`

Alias for `/generate`.

### `GET /models/{run_id}`

Returns latest generated file metadata for a run.
