from __future__ import annotations

import json
import os
import re
import uuid
import base64
import mimetypes
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph
from openai import OpenAI

from cad.examples import SIMPLE_COSMETIC_JAR_CODE
from cad.executor import ExecutionResult, execute_cadquery, strip_markdown_fences


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = os.getenv("OPENROUTER_MODEL", "arcee-ai/trinity-large-preview:free")
MAX_ATTEMPTS = int(os.getenv("CADGEN_MAX_ATTEMPTS", "3"))
VISION_MODEL = os.getenv("OPENROUTER_VISION_MODEL", "openai/gpt-4o-mini")
PROVIDER_CONFIGS = {
    "openrouter": {
        "label": "OpenRouter",
        "env_key": "OPENROUTER_API_KEY",
        "base_url": OPENROUTER_BASE_URL,
        "model": DEFAULT_MODEL,
        "vision_model": VISION_MODEL,
        "temperature": 0.2,
    },
    "gemini": {
        "label": "Gemini",
        "env_key": "GEMINI_API_KEY",
        "base_url": "https://llmfoundry.straive.com/gemini/v1beta/openai",
        "model": os.getenv("GEMINI_MODEL", "gemini-3-pro-preview"),
        "vision_model": os.getenv("GEMINI_VISION_MODEL", "gemini-3-pro-preview"),
        "temperature": 0.1,
    },
    "gpt": {
        "label": "GPT",
        "env_key": "GPT_API_KEY",
        "base_url": "https://llmfoundry.straive.com/openai/v1",
        "model": os.getenv("GPT_MODEL", "gpt-5.2"),
        "vision_model": os.getenv("GPT_VISION_MODEL", "gpt-5.2"),
        "temperature": 0.2,
    },
    "claude": {
        "label": "Claude",
        "env_key": "CLAUDE_API_KEY",
        "endpoint": "https://llmfoundry.straive.com/anthropic/v1/messages",
        "model": os.getenv("CLAUDE_MODEL", "claude-opus-4-5-20251101"),
        "vision_model": os.getenv("CLAUDE_VISION_MODEL", "claude-opus-4-5-20251101"),
        "temperature": 0.2,
    },
}
GENERIC_PROVIDER_ENV = "LLMFOUNDRY_API_KEY"


@dataclass
class LLMResult:
    text: str
    source: str
    error: str | None = None


class AgentState(TypedDict, total=False):
    prompt: str
    previous_code: str | None
    previous_run_id: str | None
    mode: str
    openrouter_api_key: str | None
    llm_provider: str
    llm_api_key: str | None
    run_id: str
    plan: str
    code: str
    attempt: int
    max_attempts: int
    execution: dict[str, Any]
    validation: dict[str, Any]
    logs: list[dict[str, Any]]
    llm: dict[str, Any]
    final: dict[str, Any]
    should_retry: bool


def run_generation(
    prompt: str,
    outputs_dir: Path,
    *,
    openrouter_api_key: str | None = None,
    llm_provider: str = "openrouter",
    llm_api_key: str | None = None,
    previous_code: str | None = None,
    previous_run_id: str | None = None,
) -> dict[str, Any]:
    graph = build_graph(outputs_dir)
    provider = normalize_provider(llm_provider)
    selected_key = select_api_key(provider, llm_api_key or openrouter_api_key)
    config = provider_config(provider)
    initial_state: AgentState = {
        "prompt": prompt,
        "previous_code": previous_code,
        "previous_run_id": previous_run_id,
        "mode": "edit" if previous_code else "generate",
        "openrouter_api_key": openrouter_api_key,
        "llm_provider": provider,
        "llm_api_key": llm_api_key or openrouter_api_key,
        "run_id": uuid.uuid4().hex,
        "attempt": 1,
        "max_attempts": MAX_ATTEMPTS,
        "logs": [],
        "llm": {
            "key_received": bool(selected_key),
            "key_fingerprint": key_fingerprint(selected_key),
            "provider": provider,
            "provider_label": config["label"],
            "used_provider": False,
            "used_openrouter": False,
            "fallback_used": False,
            "model": config["model"],
            "calls": [],
            "last_error": None,
        },
    }
    result = graph.invoke(initial_state)
    return result["final"]


def import_step_model(step_path: Path, outputs_dir: Path, *, original_filename: str | None = None) -> dict[str, Any]:
    step_path = step_path.resolve()
    run_id = uuid.uuid4().hex
    code = step_import_code(step_path)
    result = execute_cadquery(code, outputs_dir, run_id=run_id, attempt=1)
    execution = execution_to_dict(result, outputs_dir)
    step_exists = Path(execution["absolute_step_path"]).exists()
    stl_exists = Path(execution["absolute_stl_path"]).exists()
    non_empty = (execution.get("metadata", {}).get("volume") or 0) > 1e-6
    ok = result.ok and step_exists and stl_exists and non_empty
    validation = {
        "ok": ok,
        "step_exists": step_exists,
        "stl_exists": stl_exists,
        "non_empty_geometry": non_empty,
        "message": "STEP file imported and validated." if ok else execution.get("error", "STEP import failed."),
    }
    logs = [
        {
            "node": "import_step",
            "status": "completed",
            "detail": {
                "file_name": original_filename or step_path.name,
                "source_step": str(step_path),
                "message": "Saved the uploaded STEP file and created editable CadQuery import code.",
            },
        },
        {"node": "execute_code", "status": "success" if result.ok else "failed", "detail": execution},
        {"node": "validate_output", "status": "success" if ok else "failed", "detail": validation},
    ]
    return {
        "status": "success" if ok else "failed",
        "run_id": run_id,
        "attempt_count": 1,
        "prompt": f"Import STEP file: {original_filename or step_path.name}",
        "mode": "import",
        "previous_run_id": None,
        "plan": "Import the uploaded STEP file as the editable base model.",
        "code": code,
        "validation": validation,
        "logs": logs,
        "files": {
            "step": execution.get("step_url") if ok else None,
            "stl": execution.get("stl_url") if ok else None,
            "code": execution.get("code_url"),
        },
        "paths": {
            "step": execution.get("absolute_step_path"),
            "stl": execution.get("absolute_stl_path"),
            "code": execution.get("absolute_code_path"),
            "source_step": str(step_path),
        },
        "metadata": execution.get("metadata", {}),
        "llm": {
            "key_received": False,
            "key_fingerprint": None,
            "provider": "local",
            "provider_label": "Local",
            "used_provider": False,
            "used_openrouter": False,
            "fallback_used": False,
            "model": DEFAULT_MODEL,
            "calls": [],
            "last_error": None,
        },
        "message": (
            "STEP file imported. You can now describe an edit and click Edit Current."
            if ok
            else validation["message"]
        ),
    }


def step_import_code(step_path: Path) -> str:
    source = json.dumps(str(step_path))
    return f'''import cadquery as cq
from cadquery import importers


SOURCE_STEP = {source}


def build_model():
    model = importers.importStep(SOURCE_STEP)
    return model
'''


def infer_image_prompt(
    image_path: Path,
    *,
    openrouter_api_key: str | None = None,
    llm_provider: str = "openrouter",
    llm_api_key: str | None = None,
    original_filename: str | None = None,
) -> dict[str, Any]:
    image_path = image_path.resolve()
    provider = normalize_provider(llm_provider)
    config = provider_config(provider)
    selected_key = select_api_key(provider, llm_api_key or openrouter_api_key)
    system = (
        "You are an image-to-CAD interpretation agent. Infer the main object from the image "
        "and write a concise text-to-CAD prompt for CadQuery generation. Include likely "
        "primitive shapes, important features, rough proportions, and manufacturing-friendly "
        "details. Do not mention uncertainty unless the image is ambiguous."
    )
    fallback = (
        "Create a simple CAD model inspired by the uploaded image. Use robust primitive shapes, "
        "preserve the main silhouette, add the most visible functional details, and keep it as "
        "a non-empty manufacturable solid."
    )
    result = call_provider_vision(system, image_path, provider=provider, api_key=llm_api_key or openrouter_api_key, fallback=fallback)
    logs = [
        {
            "node": "image_inference",
            "status": "completed" if result.source == "openrouter" else "fallback",
            "detail": {
                "file_name": original_filename or image_path.name,
                "image_path": str(image_path),
                "llm_source": result.source,
                "llm_error": result.error,
                "provider": config["label"],
                "prompt": result.text,
            },
        }
    ]
    return {
        "status": "success",
        "mode": "image_inference",
        "prompt": result.text,
        "image_url": image_output_url(image_path),
        "logs": logs,
        "llm": {
            "key_received": bool(selected_key),
            "key_fingerprint": key_fingerprint(selected_key),
            "provider": provider,
            "provider_label": config["label"],
            "used_provider": result.source == provider,
            "used_openrouter": result.source == "openrouter",
            "fallback_used": result.source == "fallback",
            "model": config["vision_model"],
            "calls": [{"node": "image_inference", "source": result.source, "error": result.error, "model": config["vision_model"], "provider": provider}],
            "last_error": result.error,
        },
        "message": "Image inferred into a CAD prompt. Review it, then click Generate.",
    }


def call_provider_vision(system: str, image_path: Path, *, provider: str, api_key: str | None, fallback: str) -> LLMResult:
    provider = normalize_provider(provider)
    config = provider_config(provider)
    api_key = select_api_key(provider, api_key)
    if not api_key:
        return LLMResult(text=fallback, source="fallback", error=f"No {config['label']} API key was provided.")
    try:
        mime = mimetypes.guess_type(str(image_path))[0] or "image/png"
        data = base64.b64encode(image_path.read_bytes()).decode("ascii")
        text = "Infer a CAD generation prompt from this image. Return only the prompt text."
        if provider == "claude":
            content = call_claude(
                system,
                text,
                api_key=api_key,
                model=config["vision_model"],
                temperature=config["temperature"],
                image_data=data,
                image_mime=mime,
            )
            return LLMResult(text=content.strip(), source=provider) if content else LLMResult(text=fallback, source="fallback", error=f"{config['label']} vision returned an empty response.")
        client = OpenAI(base_url=config["base_url"], api_key=api_key)
        messages = openai_style_messages(system, text, provider=provider, image_data=data, image_mime=mime)
        response = client.chat.completions.create(
            model=config["vision_model"],
            messages=messages,
            temperature=config["temperature"],
        )
        content = response.choices[0].message.content
        if not content:
            return LLMResult(text=fallback, source="fallback", error=f"{config['label']} vision returned an empty response.")
        return LLMResult(text=content.strip(), source=provider)
    except Exception as exc:
        return LLMResult(text=fallback, source="fallback", error=safe_error(exc))


def image_output_url(image_path: Path) -> str | None:
    try:
        base = image_path.resolve().relative_to(Path.cwd().resolve() / "outputs")
        return f"/outputs/{base}"
    except ValueError:
        return None


def build_graph(outputs_dir: Path):
    graph = StateGraph(AgentState)
    graph.add_node("understand_prompt", understand_prompt)
    graph.add_node("generate_code", generate_code)
    graph.add_node("execute_code", lambda state: execute_code(state, outputs_dir))
    graph.add_node("validate_output", validate_output)
    graph.add_node("repair_code", repair_code)
    graph.add_node("package_response", lambda state: package_response(state, outputs_dir))

    graph.set_entry_point("understand_prompt")
    graph.add_edge("understand_prompt", "generate_code")
    graph.add_edge("generate_code", "execute_code")
    graph.add_edge("execute_code", "validate_output")
    graph.add_conditional_edges(
        "validate_output",
        route_after_validation,
        {"repair": "repair_code", "package": "package_response"},
    )
    graph.add_edge("repair_code", "execute_code")
    graph.add_edge("package_response", END)
    return graph.compile()


def understand_prompt(state: AgentState) -> AgentState:
    prompt = state["prompt"]
    if state.get("previous_code"):
        system = (
            "You are a senior CAD product designer editing an existing CadQuery model. "
            "Convert the user's requested change into a compact edit plan. Mention what "
            "should be preserved, what should change, dimensions in millimeters when useful, "
            "and validation notes. Return plain text only."
        )
        user = f"Edit request:\n{prompt}\n\nExisting CadQuery code:\n{state['previous_code']}"
        fallback = local_edit_plan_fallback(prompt)
    else:
        system = (
            "You are a senior CAD product designer. Convert a user product prompt into a compact "
            "geometry plan for CadQuery. Include primitives, dimensions in millimeters, features, "
            "and validation notes. Return plain text only."
        )
        user = prompt
        fallback = local_plan_fallback(prompt)
    result = call_provider(system, user, state=state, fallback=fallback)
    next_state = update_llm_status(state, "understand_prompt", result)
    return _append_log(
        next_state,
        "understand_prompt",
        "completed",
        {"plan": result.text, "llm_source": result.source, "llm_error": result.error},
    ) | {"plan": result.text}


def generate_code(state: AgentState) -> AgentState:
    if state.get("previous_code"):
        system = (
            "You edit runnable CadQuery 2.x Python. Return only the complete revised Python code, "
            "no markdown. Preserve the existing model intent unless the user asks to change it. "
            "The code must import cadquery as cq and define build_model() returning a cq.Workplane. "
            "Use millimeters. Keep geometry robust, avoid fragile selectors, avoid external assets, "
            "avoid show_object, avoid writing files, and combine solids before returning when possible."
        )
        user = (
            f"Edit request:\n{state['prompt']}\n\nEdit plan:\n{state['plan']}\n\n"
            f"Existing CadQuery code:\n{state['previous_code']}\n\n"
            "Return the complete edited CadQuery code now."
        )
        fallback = local_edit_fallback(state["prompt"], state["previous_code"] or "")
    else:
        system = (
            "You generate runnable CadQuery 2.x Python. Return only Python code, no markdown. "
            "The code must import cadquery as cq and define build_model() returning a cq.Workplane. "
            "Use millimeters. Keep geometry robust: avoid fragile selectors, avoid external assets, "
            "avoid show_object, avoid writing files, and combine solids before returning when possible."
        )
        user = (
            f"User prompt:\n{state['prompt']}\n\nGeometry plan:\n{state['plan']}\n\n"
            "Generate complete CadQuery code now."
        )
        fallback = local_cadquery_fallback(state["prompt"])
    result = call_provider(system, user, state=state, fallback=fallback)
    code = strip_markdown_fences(result.text)
    next_state = update_llm_status(state, "generate_code", result)
    return _append_log(
        next_state,
        "generate_code",
        "completed",
        {"attempt": state["attempt"], "llm_source": result.source, "llm_error": result.error},
    ) | {"code": code}


def execute_code(state: AgentState, outputs_dir: Path) -> AgentState:
    result = execute_cadquery(
        state["code"],
        outputs_dir,
        run_id=state["run_id"],
        attempt=state["attempt"],
    )
    payload = execution_to_dict(result, outputs_dir)
    status = "success" if result.ok else "failed"
    return _append_log(state, "execute_code", status, payload) | {"execution": payload}


def validate_output(state: AgentState) -> AgentState:
    execution = state["execution"]
    step_path = Path(execution["absolute_step_path"])
    stl_path = Path(execution["absolute_stl_path"])
    ok = (
        execution["ok"]
        and step_path.exists()
        and step_path.stat().st_size > 0
        and stl_path.exists()
        and stl_path.stat().st_size > 0
        and (execution.get("metadata", {}).get("volume") or 0) > 1e-6
    )
    validation = {
        "ok": ok,
        "step_exists": step_path.exists(),
        "stl_exists": stl_path.exists(),
        "non_empty_geometry": (execution.get("metadata", {}).get("volume") or 0) > 1e-6,
        "message": "Geometry validated and exported." if ok else execution.get("error", "Validation failed."),
    }
    retry = not ok and state["attempt"] < state["max_attempts"]
    return _append_log(state, "validate_output", "success" if ok else "failed", validation) | {
        "validation": validation,
        "should_retry": retry,
    }


def repair_code(state: AgentState) -> AgentState:
    next_attempt = state["attempt"] + 1
    system = (
        "You repair CadQuery 2.x Python. Return only complete Python code, no markdown. "
        "It must import cadquery as cq and define build_model() returning a non-empty cq.Workplane. "
        "Do not write files or call show_object."
    )
    user = (
        f"Original user prompt:\n{state['prompt']}\n\nGeometry plan:\n{state['plan']}\n\n"
        f"Previous code:\n{state['code']}\n\n"
        f"Execution error:\n{state['execution'].get('error')}\n\n"
        f"Execution metadata/stdout:\n{json.dumps(state['execution'], indent=2)}\n\n"
        "Repair the code so it executes and exports successfully."
    )
    result = call_provider(system, user, state=state, fallback=local_cadquery_fallback(state["prompt"]))
    repaired = strip_markdown_fences(result.text)
    next_state = update_llm_status(state, "repair_code", result)
    return _append_log(
        next_state,
        "repair_code",
        "completed",
        {"attempt": next_attempt, "llm_source": result.source, "llm_error": result.error},
    ) | {
        "code": repaired,
        "attempt": next_attempt,
    }


def package_response(state: AgentState, outputs_dir: Path) -> AgentState:
    execution = state.get("execution", {})
    validation = state.get("validation", {})
    ok = bool(validation.get("ok"))
    final = {
        "status": "success" if ok else "failed",
        "run_id": state["run_id"],
        "attempt_count": state["attempt"],
        "prompt": state["prompt"],
        "mode": state.get("mode", "generate"),
        "previous_run_id": state.get("previous_run_id"),
        "plan": state.get("plan", ""),
        "code": state.get("code", ""),
        "validation": validation,
        "logs": state.get("logs", []),
        "files": {
            "step": execution.get("step_url") if ok else None,
            "stl": execution.get("stl_url") if ok else None,
            "code": execution.get("code_url"),
        },
        "paths": {
            "step": execution.get("absolute_step_path"),
            "stl": execution.get("absolute_stl_path"),
            "code": execution.get("absolute_code_path"),
        },
        "metadata": execution.get("metadata", {}),
        "llm": state.get("llm", {"used_openrouter": False, "fallback_used": True}),
        "message": (
            "Your CAD model is ready. Preview the mesh or download the STEP file."
            if ok and state.get("mode") != "edit"
            else "Your edited CAD model is ready. Preview the mesh or download the STEP file."
            if ok
            else validation.get("message", "CAD generation failed.")
        ),
    }
    return state | {"final": final}


def route_after_validation(state: AgentState) -> str:
    return "repair" if state.get("should_retry") else "package"


def call_provider(system: str, user: str, *, state: AgentState, fallback: str) -> LLMResult:
    provider = normalize_provider(state.get("llm_provider", "openrouter"))
    config = provider_config(provider)
    api_key = select_api_key(provider, state.get("llm_api_key") or state.get("openrouter_api_key"))
    if not api_key:
        return LLMResult(text=fallback, source="fallback", error=f"No {config['label']} API key was provided.")

    try:
        if provider == "claude":
            content = call_claude(
                system,
                user,
                api_key=api_key,
                model=config["model"],
                temperature=config["temperature"],
            )
        else:
            client = OpenAI(base_url=config["base_url"], api_key=api_key)
            response = client.chat.completions.create(
                model=config["model"],
                messages=openai_style_messages(system, user, provider=provider),
                temperature=config["temperature"],
            )
            content = response.choices[0].message.content
        if not content:
            return LLMResult(text=fallback, source="fallback", error=f"{config['label']} returned an empty response.")
        return LLMResult(text=content.strip(), source=provider)
    except Exception as exc:
        return LLMResult(text=fallback, source="fallback", error=safe_error(exc))


def openai_style_messages(
    system: str,
    user: str,
    *,
    provider: str,
    image_data: str | None = None,
    image_mime: str | None = None,
) -> list[dict[str, Any]]:
    if image_data:
        content = [
            {"type": "text", "text": f"{system}\n\n{user}" if provider == "gemini" else user},
            {"type": "image_url", "image_url": {"url": f"data:{image_mime or 'image/png'};base64,{image_data}"}},
        ]
        if provider == "gemini":
            return [{"role": "user", "content": content}]
        return [{"role": "system", "content": system}, {"role": "user", "content": content}]
    if provider == "gemini":
        return [{"role": "user", "content": f"{system}\n\n{user}"}]
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def call_claude(
    system: str,
    user: str,
    *,
    api_key: str,
    model: str,
    temperature: float,
    image_data: str | None = None,
    image_mime: str | None = None,
) -> str:
    config = provider_config("claude")
    content: list[dict[str, Any]] = []
    if image_data:
        content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": image_mime or "image/png",
                    "data": image_data,
                },
            }
        )
    content.append({"type": "text", "text": f"{system}\n\n{user}"})
    payload = json.dumps(
        {
            "model": model,
            "max_tokens": 4096,
            "temperature": temperature,
            "messages": [{"role": "user", "content": content}],
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        config["endpoint"],
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Claude HTTP {exc.code}: {body[:400]}") from exc
    parts = data.get("content", [])
    return "\n".join(part.get("text", "") for part in parts if part.get("type") == "text").strip()


def safe_error(exc: Exception) -> str:
    message = str(exc).strip() or exc.__class__.__name__
    redacted = re.sub(r"sk-[A-Za-z0-9_-]+", "sk-REDACTED", message)
    redacted = re.sub(r"Bearer\s+[A-Za-z0-9._-]+", "Bearer REDACTED", redacted)
    return redacted[:500]


def normalize_api_key(api_key: str | None) -> str | None:
    if not api_key:
        return None
    cleaned = api_key.strip().strip('"').strip("'")
    return cleaned or None


def normalize_provider(provider: str | None) -> str:
    cleaned = (provider or "openrouter").strip().lower()
    return cleaned if cleaned in PROVIDER_CONFIGS else "openrouter"


def provider_config(provider: str) -> dict[str, Any]:
    return PROVIDER_CONFIGS[normalize_provider(provider)]


def select_api_key(provider: str, api_key: str | None) -> str | None:
    direct = normalize_api_key(api_key)
    if direct:
        return direct
    config = provider_config(provider)
    return normalize_api_key(
        os.getenv(config["env_key"])
        or os.getenv(GENERIC_PROVIDER_ENV)
        or (os.getenv("OPENROUTER_API_KEY") if provider == "openrouter" else None)
    )


def key_fingerprint(api_key: str | None) -> str | None:
    cleaned = normalize_api_key(api_key)
    if not cleaned:
        return None
    return f"{cleaned[:8]}...{cleaned[-6:]}"


def update_llm_status(state: AgentState, node: str, result: LLMResult) -> AgentState:
    existing = state.get("llm", {})
    calls = list(existing.get("calls", []))
    provider = normalize_provider(existing.get("provider") or state.get("llm_provider", "openrouter"))
    config = provider_config(provider)
    calls.append({"node": node, "source": result.source, "error": result.error, "model": config["model"], "provider": provider})
    used_openrouter = any(call["source"] == "openrouter" for call in calls)
    used_provider = any(call["source"] == provider for call in calls)
    fallback_used = any(call["source"] == "fallback" for call in calls)
    return state | {
        "llm": {
            "key_received": bool(existing.get("key_received")),
            "key_fingerprint": existing.get("key_fingerprint"),
            "provider": provider,
            "provider_label": config["label"],
            "used_provider": used_provider,
            "used_openrouter": used_openrouter,
            "fallback_used": fallback_used,
            "model": config["model"],
            "calls": calls,
            "last_error": next((call["error"] for call in reversed(calls) if call.get("error")), None),
        }
    }


def local_cadquery_fallback(prompt: str) -> str:
    normalized = re.sub(r"\s+", " ", prompt.lower())
    if "jar" in normalized or "cosmetic" in normalized or "cylind" in normalized:
        return SIMPLE_COSMETIC_JAR_CODE
    if "box" in normalized or "cube" in normalized or "rectang" in normalized:
        return r'''
import cadquery as cq


def build_model():
    body = cq.Workplane("XY").box(70, 45, 30)
    raised_panel = cq.Workplane("XY").workplane(offset=15).box(46, 24, 4)
    return body.union(raised_panel)
'''
    if "bracket" in normalized or "mount" in normalized:
        return r'''
import cadquery as cq


def build_model():
    base = cq.Workplane("XY").box(80, 35, 8)
    upright = cq.Workplane("XY").workplane(offset=4).transformed(offset=(0, -13.5, 26)).box(80, 8, 44)
    hole_1 = cq.Workplane("XY").workplane(offset=5).center(-24, 0).circle(5).extrude(10)
    hole_2 = cq.Workplane("XY").workplane(offset=5).center(24, 0).circle(5).extrude(10)
    return base.union(upright).cut(hole_1).cut(hole_2)
'''
    return r'''
import cadquery as cq


def build_model():
    base = cq.Workplane("XY").box(60, 40, 18)
    cylinder = cq.Workplane("XY").workplane(offset=9).circle(13).extrude(24)
    top = cq.Workplane("XY").workplane(offset=33).circle(8).extrude(6)
    return base.union(cylinder).union(top)
'''


def local_edit_fallback(prompt: str, previous_code: str) -> str:
    normalized = re.sub(r"\s+", " ", prompt.lower())
    if "box" in normalized:
        return local_cadquery_fallback("box")
    if "jar" in normalized or "cosmetic" in normalized:
        return SIMPLE_COSMETIC_JAR_CODE
    if previous_code.strip():
        return previous_code
    return local_cadquery_fallback(prompt)


def local_edit_plan_fallback(prompt: str) -> str:
    return (
        f"Edit the existing CadQuery model according to this request: {prompt}. "
        "Preserve the existing build_model() structure, keep the result exportable, "
        "and validate that the edited model remains a non-empty solid."
    )


def local_plan_fallback(prompt: str) -> str:
    normalized = re.sub(r"\s+", " ", prompt.lower())
    if "jar" in normalized or "cosmetic" in normalized or "cylind" in normalized:
        return (
            "Create a cylindrical cosmetic jar in millimeters: hollow cylindrical body, "
            "visible lid above the body, raised lid detail, neck ring suggesting threads, "
            "and all solids combined for export."
        )
    if "box" in normalized or "cube" in normalized or "rectang" in normalized:
        return (
            "Create a rectangular box in millimeters with a main cuboid body and a raised "
            "top panel. Keep the shape as a single combined solid for export."
        )
    if "bracket" in normalized or "mount" in normalized:
        return (
            "Create a mounting bracket in millimeters with a rectangular base, vertical "
            "upright plate, and two through holes in the base. Keep geometry simple and robust."
        )
    return (
        "Create a simple manufacturable solid in millimeters using robust CadQuery primitives. "
        "Use a base body with one raised feature and combine solids for export."
    )


def execution_to_dict(result: ExecutionResult, outputs_dir: Path) -> dict[str, Any]:
    outputs_dir = outputs_dir.resolve()
    base = result.workdir.relative_to(outputs_dir)
    return {
        "ok": result.ok,
        "run_id": result.run_id,
        "attempt": result.attempt,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "error": result.error,
        "metadata": result.metadata,
        "step_url": f"/outputs/{base}/model.step",
        "stl_url": f"/outputs/{base}/model.stl",
        "code_url": f"/outputs/{base}/generated_model.py",
        "absolute_step_path": str(result.step_path),
        "absolute_stl_path": str(result.stl_path),
        "absolute_code_path": str(result.code_path),
    }


def _append_log(state: AgentState, node: str, status: str, detail: dict[str, Any]) -> AgentState:
    logs = list(state.get("logs", []))
    logs.append({"node": node, "status": status, "detail": detail})
    return state | {"logs": logs}
