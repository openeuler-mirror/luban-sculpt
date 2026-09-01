"""Inject producer / version into manifest and HF config sidecars."""

from __future__ import annotations

from luban_sculpt import __version__
from luban_sculpt.contracts import ArtifactManifest, BackendPlan, ExportFormat


def build_manifest(plan: BackendPlan, repack_log: list[str]) -> ArtifactManifest:
    """由 BackendPlan 与 HAL repack 日志生成 vLLM 启动提示字段。"""
    scheme_cfg = _scheme_from_profile_hint(plan)
    vllm_launch = {
        "quant_method": _vllm_quant_method(plan.export_format),
        "deploy_target": plan.intent.deploy_target,
        "profile_id": plan.hw.profile_id,
    }
    if plan.export_format == ExportFormat.VLLM_ASCEND or plan.intent.backend == "msmodelslim":
        vllm_launch["quant_method"] = "ascend"
        vllm_launch.setdefault("extra_args", ["--quantization=ascend"])
    if plan.intent.backend == "gptq":
        vllm_launch["quant_method"] = "gptq"
        vllm_launch.setdefault("extra_args", ["--quantization=gptq"])
    if plan.intent.backend == "awq":
        vllm_launch["quant_method"] = "awq"
        vllm_launch.setdefault("extra_args", ["--quantization=awq"])
    if scheme_cfg.get("infer", {}).get("extra_args"):
        vllm_launch["extra_args"] = scheme_cfg["infer"]["extra_args"]

    return ArtifactManifest(
        profile_id=plan.hw.profile_id,
        hw_decision=plan.hw,
        backend=plan.intent.backend,
        abstract_scheme=plan.intent.abstract_scheme,
        export_format=plan.export_format,
        model_arch=plan.intent.arch_snapshot.arch.value,
        model_arch_snapshot=plan.intent.arch_snapshot.model_dump(mode="json"),
        tool_version=__version__,
        producer={
            "name": "luban-sculpt",
            "version": __version__,
            "backend": plan.intent.backend,
            "model_arch": plan.intent.arch_snapshot.arch.value,
            **(
                {"msmodelslim": True}
                if plan.intent.backend == "msmodelslim"
                else {"gptqmodel": True}
                if plan.intent.backend == "gptq"
                else {}
            ),
        },
        repack_applied=list(repack_log),
        vllm_launch=vllm_launch,
    )


def inject_hf_config_producer(config: dict, manifest: ArtifactManifest) -> dict:
    """Merge luban producer block into model config.json (optional)."""
    config.setdefault("quantization_config", {})
    qc = config["quantization_config"]
    if isinstance(qc, dict):
        qc["luban_producer"] = manifest.producer
        qc["luban_profile_id"] = manifest.profile_id
    return config


def _vllm_quant_method(export_format: ExportFormat) -> str:
    mapping = {
        ExportFormat.COMPRESSED_TENSORS: "compressed-tensors",
        ExportFormat.FP8_HF: "fp8",
        ExportFormat.AWQ_HF: "awq",
        ExportFormat.GPTQ_HF: "gptq",
        ExportFormat.VLLM_ASCEND: "ascend",
        ExportFormat.INFERA_V1: "infera-v1",
    }
    return mapping.get(export_format, export_format.value)


def _scheme_from_profile_hint(plan: BackendPlan) -> dict:
    return {}
