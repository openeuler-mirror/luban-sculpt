"""Run llm-compressor ``oneshot`` with luban Modifier 拦截链。

- 按 scheme 选 GPTQ/AWQ/Quant recipe（见 scheme_map）
- ``save_compressed`` + 可选 ``quantization_format``
- 海光推理 sidecar 由 ``luban_sculpt.backends.hygon`` 负责
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from luban_sculpt.backends.llm_compressor.probe import is_llmcompressor_available
from luban_sculpt.modifiers.recipe import build_recipe_for_plan
from luban_sculpt.backends.llm_compressor.scheme_map import resolve_compress_spec
from luban_sculpt.calib import CalibRunner
from luban_sculpt.contracts import BackendPlan
from luban_sculpt.log import get_logger

logger = get_logger(__name__)


def _lc_opts(plan: BackendPlan) -> dict[str, Any]:
    opts = plan.intent.backend_options or {}
    return opts.get("llm_compressor") or opts


def _compress_spec(plan: BackendPlan):
    return resolve_compress_spec(
        plan.intent.abstract_scheme,
        lc_override=_lc_opts(plan),
    )


def _oneshot_kwargs(
    plan: BackendPlan,
    calib: CalibRunner | None = None,
    data: Any | None = None,
) -> dict[str, Any]:
    """合并 recipe calib（经 CalibRunner）与 llm_compressor.oneshot 配置。

    支持 ``llm_compressor.pipeline`` / ``llm_compressor.oneshot.pipeline``
    （如 ``sequential``：顺序加载、逐层量化）。
    """
    lc = _lc_opts(plan)
    oneshot_cfg = dict(lc.get("oneshot") or {})
    # 顶层 pipeline 与 oneshot.pipeline 等价；oneshot 内显式值优先
    if "pipeline" not in oneshot_cfg and lc.get("pipeline") is not None:
        oneshot_cfg["pipeline"] = lc["pipeline"]
    runner = calib or CalibRunner(plan)
    for key, value in runner.oneshot_kwargs(data).items():
        oneshot_cfg.setdefault(key, value)
    return oneshot_cfg


def _save_kwargs(spec: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"save_compressed": True}
    if spec.quantization_format:
        kwargs["quantization_format"] = spec.quantization_format
    return kwargs


def _log_cuda_mem(prefix: str) -> None:
    try:
        import torch

        if not torch.cuda.is_available():
            return
        free, total = torch.cuda.mem_get_info()
        alloc = torch.cuda.memory_allocated()
        logger.info(
            "%s cuda mem free=%.2fGiB total=%.2fGiB allocated=%.2fGiB",
            prefix,
            free / 1024**3,
            total / 1024**3,
            alloc / 1024**3,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("cuda mem log skip: %s", exc)


def _oneshot_supports_pipeline(oneshot_fn: Any) -> bool:
    import inspect

    try:
        return "pipeline" in inspect.signature(oneshot_fn).parameters
    except Exception:  # noqa: BLE001
        return False


def _maybe_sequential_pipeline(
    oneshot_fn: Any,
    oneshot_kwargs: dict[str, Any],
    lc_opts: dict[str, Any],
    is_gptq: bool,
) -> dict[str, Any]:
    """注入 oneshot(pipeline=...)。

    优先级：
    1. 已有 ``oneshot_kwargs["pipeline"]``（来自 YAML oneshot / 顶层 pipeline）
    2. GPTQ 默认 ``sequential``（降 Propagating 峰值）
    3. 显式 ``pipeline: false`` / ``null`` 关闭
    """
    out = dict(oneshot_kwargs)
    if "pipeline" in out:
        want = out["pipeline"]
    else:
        want = lc_opts.get("pipeline")
        if want is None and is_gptq:
            want = "sequential"

    if want in (None, False, ""):
        out.pop("pipeline", None)
        return out

    if not _oneshot_supports_pipeline(oneshot_fn):
        logger.info("oneshot 无 pipeline 参数，跳过 pipeline=%s", want)
        out.pop("pipeline", None)
        return out

    out["pipeline"] = want
    logger.info("oneshot pipeline=%s", want)
    return out


def run_llm_compressor_oneshot(
    plan: BackendPlan,
    output_dir: Path,
    *,
    dry_run: bool | None = None,
) -> dict[str, Any]:
    """构建 recipe + 拦截链，调用 llmcompressor.oneshot 或 dry-run 写脚本 stub。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    recipe, recipe_meta = build_recipe_for_plan(plan)
    interceptor_log = recipe_meta.get("interceptor_log", [])
    spec = _compress_spec(plan)

    meta: dict[str, Any] = {
        "backend": "llm_compressor",
        "model_id": plan.intent.model_id,
        "abstract_scheme": plan.intent.abstract_scheme,
        "compress_algorithm": spec.algorithm,
        "compress_scheme": spec.scheme,
        "quantization_format": spec.quantization_format,
        "interceptor_log": interceptor_log,
        "modifier_types": recipe_meta.get("modifier_types"),
        "recipe_repr": repr(recipe),
    }

    if dry_run is None:
        dry_run = os.environ.get("LUBAN_LLM_COMPRESSOR_DRY_RUN", "").lower() in (
            "1",
            "true",
            "yes",
        )

    available = is_llmcompressor_available()
    if dry_run or not available:
        if not dry_run and not available:
            logger.warning(
                "llmcompressor oneshot unavailable (need torch>=2.10; "
                "Intel Mac has no such wheel). dry-run. "
                "Set LUBAN_LLM_COMPRESSOR_DRY_RUN=1 to silence this; "
                "on GPU Linux: bash tools/install_llm_compressor.sh"
            )
        logger.info(
            "llm_compressor dry-run model_id=%s scheme=%s algo=%s output=%s",
            plan.intent.model_id,
            spec.scheme,
            spec.algorithm,
            output_dir,
        )
        script = _render_oneshot_script(plan, recipe, output_dir, spec)
        (output_dir / "llm_compressor_oneshot.py").write_text(script, encoding="utf-8")
        (output_dir / "recipe_stub.json").write_text(
            json.dumps(_recipe_to_jsonable(recipe), indent=2),
            encoding="utf-8",
        )
        lc_opts = _lc_opts(plan)
        pipe = lc_opts.get("pipeline")
        if pipe is None and spec.algorithm == "gptq":
            pipe = "sequential"
        meta["oneshot_pipeline"] = pipe if pipe not in (False, "") else None
        meta["status"] = "dry_run"
        return meta

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from llmcompressor import oneshot

    model_id = plan.intent.model_id
    lc_opts = _lc_opts(plan)
    trust = lc_opts.get("trust_remote_code", True)
    if "trust_remote_code" in (plan.intent.backend_options or {}):
        trust = plan.intent.backend_options["trust_remote_code"]

    # GPTQ Propagating 阶段极易把 CPU RAM 打满导致整机假死：
    # 禁止 auto 把层 offload 到 CPU；整模放单卡 + fp16。
    is_gptq = spec.algorithm == "gptq"
    torch_dtype = lc_opts.get("torch_dtype", "float16" if is_gptq else "auto")
    device_map = lc_opts.get(
        "device_map",
        "cuda:0" if is_gptq else "auto",
    )
    load_kw: dict[str, Any] = {"trust_remote_code": trust, "low_cpu_mem_usage": True}
    if torch_dtype is not None:
        load_kw["torch_dtype"] = torch_dtype
    if device_map is not None:
        load_kw["device_map"] = device_map

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    logger.info(
        "[1/4] load model+tokenizer model_id=%s dtype=%s device_map=%s gptq=%s",
        model_id,
        torch_dtype,
        device_map,
        is_gptq,
    )
    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kw)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info(
        "[1/4] model loaded class=%s pad_token=%r",
        type(model).__name__,
        tokenizer.pad_token,
    )

    calib_runner = CalibRunner(plan)
    # GPTQ：硬上限，防止 recipe 仍写 512×2048 把 Propagating 打挂
    if is_gptq:
        hard_samples = int(lc_opts.get("gptq_max_samples_cap", 64))
        hard_seq = int(lc_opts.get("gptq_max_seq_cap", 512))
        if calib_runner.max_samples > hard_samples:
            logger.warning(
                "GPTQ clamp max_samples %s → %s (Propagating OOM 防护)",
                calib_runner.max_samples,
                hard_samples,
            )
            plan.intent.calib["max_samples"] = hard_samples
        if calib_runner.max_seq_length > hard_seq:
            logger.warning(
                "GPTQ clamp max_seq_length %s → %s (Propagating OOM 防护)",
                calib_runner.max_seq_length,
                hard_seq,
            )
            plan.intent.calib["max_seq_length"] = hard_seq
        # 重建 runner 以吃到 clamp 后的 calib
        calib_runner = CalibRunner(plan)

    logger.info(
        "[2/4] load calib source=%s path=%s max_samples=%s max_seq_length=%s "
        "max_chars=%s shuffle=%s seed=%s",
        calib_runner.source,
        calib_runner.path,
        calib_runner.max_samples,
        calib_runner.max_seq_length,
        calib_runner.max_chars,
        calib_runner.shuffle,
        calib_runner.seed,
    )
    calib_data = calib_runner.load()
    meta["calib_report"] = calib_data.report
    _log_calib_preview(calib_data, preview_n=3)

    oneshot_kwargs = _oneshot_kwargs(plan, calib_runner, calib_data)
    # 释放重复的大字符串列表（Dataset 已持有 text）
    calib_data.samples = []
    calib_data.texts = []

    # oneshot(pipeline=sequential)：顺序加载、逐层量化，降峰值内存
    oneshot_kwargs = _maybe_sequential_pipeline(oneshot, oneshot_kwargs, lc_opts, is_gptq)
    meta["oneshot_pipeline"] = oneshot_kwargs.get("pipeline")

    ds = oneshot_kwargs.get("dataset")
    _log_cuda_mem("[3/4] before oneshot")
    logger.info(
        "[3/4] oneshot start keys=%s pipeline=%s num_calibration_samples=%s "
        "max_seq_length=%s dataset_type=%s recipe=%s",
        list(oneshot_kwargs.keys()),
        oneshot_kwargs.get("pipeline"),
        oneshot_kwargs.get("num_calibration_samples"),
        oneshot_kwargs.get("max_seq_length"),
        type(ds).__name__ if ds is not None else None,
        type(recipe).__name__,
    )
    try:
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        oneshot(model=model, recipe=recipe, **oneshot_kwargs)
    except Exception:
        logger.exception(
            "[3/4] oneshot failed — GPTQ Propagating OOM 时请把 "
            "calib.max_samples≤64、max_seq_length≤512，并确认未用 device_map=auto 卸到 CPU"
        )
        raise
    logger.info("[3/4] oneshot done")
    _log_cuda_mem("[3/4] after oneshot")

    save_kw = _save_kwargs(spec)
    logger.info("[4/4] save_pretrained dir=%s kwargs=%s", output_dir, save_kw)
    model.save_pretrained(str(output_dir), **save_kw)
    tokenizer.save_pretrained(str(output_dir))
    logger.info("[4/4] saved ok → %s", output_dir)

    meta["status"] = "ok"
    meta["output_dir"] = str(output_dir)
    return meta


def _log_calib_preview(calib_data: Any, *, preview_n: int = 3, max_chars: int = 240) -> None:
    """打印前几条校准文本，确认 conversations→text 是否正常。"""
    report = getattr(calib_data, "report", {}) or {}
    n = int(getattr(calib_data, "num_samples", 0) or 0)
    logger.info(
        "[2/4] calib loaded samples=%s resolved_path=%s report=%s",
        n,
        getattr(calib_data, "resolved_path", None),
        report,
    )
    if n <= 0:
        logger.warning("[2/4] calib empty — oneshot may fall back / fail")
        return

    texts = list(getattr(calib_data, "texts", None) or [])
    samples = list(getattr(calib_data, "samples", None) or [])
    saw_stub = False
    for i in range(min(preview_n, n)):
        raw = texts[i] if i < len(texts) else str((samples[i] or {}).get("text", ""))
        preview = raw.replace("\n", "\\n")
        if len(preview) > max_chars:
            preview = preview[:max_chars] + "..."
        is_stub = raw.startswith("calib stub sample")
        saw_stub = saw_stub or is_stub
        logger.info(
            "[2/4] calib preview[%d/%d] chars=%d stub=%s text=%r",
            i,
            n,
            len(raw),
            is_stub,
            preview,
        )
    if saw_stub:
        logger.warning(
            "[2/4] calib looks like stub — check `pip install datasets` / "
            "calib.path / open-perfectblend exists"
        )


def _recipe_to_jsonable(recipe: Any) -> Any:
    if hasattr(recipe, "model_dump"):
        return recipe.model_dump()
    if isinstance(recipe, dict):
        return recipe
    return {"repr": repr(recipe)}


def _render_oneshot_script(
    plan: BackendPlan,
    recipe: Any,
    output_dir: Path,
    spec: Any,
) -> str:
    model_id = plan.intent.model_id
    ignore = plan.intent.ignore or ["lm_head"]
    save_kw = _save_kwargs(spec)
    save_kw_repr = ", ".join(f"{k}={v!r}" for k, v in save_kw.items())
    lc = _lc_opts(plan)
    pipeline = lc.get("pipeline")
    if pipeline is None and spec.algorithm == "gptq":
        pipeline = "sequential"
    if pipeline in (None, False, ""):
        oneshot_call = "oneshot(model=model, recipe=recipe)"
    else:
        oneshot_call = f"oneshot(model=model, recipe=recipe, pipeline={pipeline!r})"

    if spec.algorithm == "gptq":
        block = f", block_size={spec.block_size}" if spec.block_size else ""
        recipe_src = f"""from llmcompressor.modifiers.gptq import GPTQModifier
recipe = GPTQModifier(
    targets="Linear",
    scheme={spec.scheme!r},
    ignore={ignore!r}{block},
)"""
    elif spec.algorithm == "awq":
        recipe_src = f"""from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform.awq import AWQModifier
recipe = [
    AWQModifier(duo_scaling=True),
    QuantizationModifier(
        targets="Linear",
        scheme={spec.scheme!r},
        ignore={ignore!r},
    ),
]"""
    else:
        recipe_src = f"""from llmcompressor.modifiers.quantization import QuantizationModifier
recipe = QuantizationModifier(
    targets="Linear",
    scheme={spec.scheme!r},
    ignore={ignore!r},
)"""

    return f'''"""Generated by luban-sculpt — run after: pip install llmcompressor
abstract_scheme={plan.intent.abstract_scheme!r} algorithm={spec.algorithm!r}
"""
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot

MODEL_ID = {model_id!r}
SAVE_DIR = {str(output_dir)!r}

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

{recipe_src}

{oneshot_call}
model.save_pretrained(SAVE_DIR, {save_kw_repr})
tokenizer.save_pretrained(SAVE_DIR)
'''
