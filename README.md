# 3daistudio

A full-stack text-to-CAD app built with FastAPI, LangGraph, LLM Foundry providers, CadQuery, and a Three.js preview.

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
export LLMFOUNDRY_API_KEY="your-provider-key"
uvicorn app:app --reload
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000).

You can also paste a Gemini, GPT, or Claude provider key directly into the UI. The browser stores it in `localStorage` and sends it with each generation request. The backend also supports provider-specific environment variables:

- `GEMINI_API_KEY`
- `GPT_API_KEY`
- `CLAUDE_API_KEY`
- `LLMFOUNDRY_API_KEY`

## First Prompt

```text
create a simple cosmetic jar with cylindrical body and lid
```

Generation requires a working provider key. No offline generation is provided.

## API

### `POST /generate`

Request:

```json
{
  "prompt": "create a simple cosmetic jar with cylindrical body and lid",
  "llm_provider": "gemini",
  "llm_api_key": "..."
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
