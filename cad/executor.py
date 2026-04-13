from __future__ import annotations

import json
import subprocess
import sys
import textwrap
import uuid
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExecutionResult:
    ok: bool
    run_id: str
    attempt: int
    workdir: Path
    code_path: Path
    step_path: Path
    stl_path: Path
    stdout: str
    stderr: str
    error: str | None
    metadata: dict


HARNESS = r'''
import contextlib
import io
import json
import runpy
import sys
import traceback
from pathlib import Path

import cadquery as cq
from cadquery import exporters

code_path = Path(sys.argv[1])
step_path = Path(sys.argv[2])
stl_path = Path(sys.argv[3])

captured_stdout = io.StringIO()

def as_shape(obj):
    if isinstance(obj, cq.Workplane):
        return obj
    if isinstance(obj, cq.Shape):
        return cq.Workplane("XY").add(obj)
    if isinstance(obj, (list, tuple)) and obj:
        wp = cq.Workplane("XY")
        for item in obj:
            wp = wp.add(item)
        return wp
    return obj

try:
    with contextlib.redirect_stdout(captured_stdout):
        namespace = runpy.run_path(str(code_path))

    model = None
    if callable(namespace.get("build_model")):
        model = namespace["build_model"]()
    elif "model" in namespace:
        model = namespace["model"]
    elif "result" in namespace:
        model = namespace["result"]
    elif "shape" in namespace:
        model = namespace["shape"]

    model = as_shape(model)
    if model is None:
        raise ValueError("CadQuery code must define build_model(), model, result, or shape.")
    if not isinstance(model, cq.Workplane):
        raise TypeError(f"Generated object must be a cadquery.Workplane or Shape, got {type(model)!r}.")

    vals = model.vals()
    solids = [v for v in vals if hasattr(v, "Volume") and v.Volume() > 1e-6]
    if not vals or not solids:
        raise ValueError("Generated geometry is empty or contains no solid with positive volume.")

    combined = model.combine()
    solid = combined.val()
    volume = float(solid.Volume())
    bbox = solid.BoundingBox()
    if volume <= 1e-6:
        raise ValueError("Generated geometry volume is zero.")

    step_path.parent.mkdir(parents=True, exist_ok=True)
    exporters.export(combined, str(step_path))
    exporters.export(combined, str(stl_path))

    if not step_path.exists() or step_path.stat().st_size == 0:
        raise ValueError("STEP export failed or produced an empty file.")
    if not stl_path.exists() or stl_path.stat().st_size == 0:
        raise ValueError("STL preview export failed or produced an empty file.")

    print(json.dumps({
        "ok": True,
        "stdout": captured_stdout.getvalue(),
        "metadata": {
            "volume": volume,
            "bbox": {
                "x": float(bbox.xlen),
                "y": float(bbox.ylen),
                "z": float(bbox.zlen)
            },
            "solids": len(solids),
            "step_size": step_path.stat().st_size,
            "stl_size": stl_path.stat().st_size
        }
    }))
except Exception as exc:
    print(json.dumps({
        "ok": False,
        "stdout": captured_stdout.getvalue(),
        "error": str(exc),
        "traceback": traceback.format_exc()
    }))
    sys.exit(1)
'''


def execute_cadquery(
    code: str,
    outputs_dir: Path,
    *,
    run_id: str | None = None,
    attempt: int = 1,
    timeout_seconds: int = 35,
) -> ExecutionResult:
    """Execute generated CadQuery in a subprocess and export STEP/STL files."""
    outputs_dir = outputs_dir.resolve()
    run_id = run_id or uuid.uuid4().hex
    workdir = outputs_dir / run_id / f"attempt-{attempt}"
    workdir.mkdir(parents=True, exist_ok=True)

    code_path = workdir / "generated_model.py"
    harness_path = workdir / "_runner.py"
    step_path = workdir / "model.step"
    stl_path = workdir / "model.stl"

    code_path.write_text(code, encoding="utf-8")
    harness_path.write_text(HARNESS, encoding="utf-8")

    try:
        proc = subprocess.run(
            [sys.executable, str(harness_path), str(code_path), str(step_path), str(stl_path)],
            cwd=str(workdir),
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return ExecutionResult(
            ok=False,
            run_id=run_id,
            attempt=attempt,
            workdir=workdir,
            code_path=code_path,
            step_path=step_path,
            stl_path=stl_path,
            stdout=exc.stdout or "",
            stderr=exc.stderr or "",
            error=f"CadQuery execution timed out after {timeout_seconds} seconds.",
            metadata={},
        )

    payload = _parse_runner_payload(proc.stdout)
    ok = proc.returncode == 0 and bool(payload.get("ok"))
    return ExecutionResult(
        ok=ok,
        run_id=run_id,
        attempt=attempt,
        workdir=workdir,
        code_path=code_path,
        step_path=step_path,
        stl_path=stl_path,
        stdout=payload.get("stdout") or proc.stdout,
        stderr=proc.stderr,
        error=None if ok else payload.get("error") or proc.stderr or "CadQuery execution failed.",
        metadata=payload.get("metadata") or {"traceback": payload.get("traceback")},
    )


def _parse_runner_payload(stdout: str) -> dict:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return {"ok": False, "error": "CadQuery runner did not return JSON output.", "stdout": stdout}


def strip_markdown_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines)
    return textwrap.dedent(cleaned).strip()
