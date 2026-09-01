"""Post-quant validation axis 2: vLLM runtime load / smoke generate."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Literal

RuntimeMode = Literal["import", "load", "generate"]


def validate_runtime(
    model_path: Path,
    *,
    mode: RuntimeMode = "import",
    prompt: str = "Hello",
    max_tokens: int = 8,
    timeout_s: int = 600,
) -> dict[str, Any]:
    """量化产物运行时校验（vLLM）。

    | mode | 行为 |
    |------|------|
    | ``import`` | 仅确认 ``vllm`` 可 import（CI / 无卡） |
    | ``load`` | 子进程 ``LLM(model=...)`` 加载 |
    | ``generate`` | 加载后跑一句短生成（smoke） |

    返回 ``{"ok": bool, ...}``；失败时含 ``error`` / ``stderr``。
    """
    if not model_path.is_dir():
        return {"ok": False, "error": f"not a directory: {model_path}", "mode": mode}

    manifest_path = model_path / "manifest.json"
    manifest: dict[str, Any] = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    profile = manifest.get("profile_id")

    if mode == "import":
        try:
            import vllm  # noqa: F401
        except ImportError:
            return {
                "ok": False,
                "error": "vllm not installed; pip install luban-sculpt[vllm]",
                "mode": mode,
                "manifest_profile": profile,
            }
        return {"ok": True, "mode": mode, "manifest_profile": profile}

    if mode == "load":
        code = (
            "import sys; "
            "from vllm import LLM; "
            "LLM(model=sys.argv[1], trust_remote_code=True, max_model_len=512); "
            "print('ok')"
        )
    elif mode == "generate":
        code = (
            "import sys, json; "
            "from vllm import LLM, SamplingParams; "
            "m=LLM(model=sys.argv[1], trust_remote_code=True, max_model_len=512); "
            f"p=SamplingParams(max_tokens={int(max_tokens)}, temperature=0.0); "
            f"out=m.generate([{prompt!r}], p); "
            "print(json.dumps({'text': out[0].outputs[0].text}))"
        )
    else:
        return {"ok": False, "error": f"unknown runtime mode: {mode!r}", "mode": mode}

    try:
        proc = subprocess.run(
            [sys.executable, "-c", code, str(model_path)],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        ok = proc.returncode == 0
        result: dict[str, Any] = {
            "ok": ok,
            "mode": mode,
            "stdout": proc.stdout[-2000:],
            "stderr": proc.stderr[-2000:],
            "manifest_profile": profile,
        }
        if not ok:
            result["error"] = proc.stderr[-500:] or f"runtime {mode} failed"
        return result
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"runtime {mode} timeout", "mode": mode}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc), "mode": mode}
