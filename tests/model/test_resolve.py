"""Model arch classification tests."""

from __future__ import annotations

from pathlib import Path

from luban_sculpt.compiler.plan import compile_plan
from luban_sculpt.model import MOE_ARCHES, merge_ignore_list, resolve_model_arch
from luban_sculpt.contracts import ModelArch
from tests.paths import RECIPES


def test_classify_from_model_id_heuristic() -> None:
    assert resolve_model_arch("Qwen/Qwen2.5-7B-Instruct").arch == ModelArch.QWEN
    assert resolve_model_arch("meta-llama/Llama-3.1-8B-Instruct").arch == ModelArch.LLAMA
    mc = resolve_model_arch("deepseek-ai/DeepSeek-V3")
    assert mc.arch == ModelArch.DEEPSEEK_MOE
    assert mc.is_moe is True
    assert mc.arch in MOE_ARCHES


def test_classify_recipe_override() -> None:
    mc = resolve_model_arch(
        "some/unknown-checkpoint",
        recipe={"model_arch": "qwen_moe"},
    )
    assert mc.arch == ModelArch.QWEN_MOE
    assert mc.source == "recipe"
    assert mc.is_moe is True


def test_classify_from_hf_config() -> None:
    mc = resolve_model_arch(
        "/tmp/does-not-exist",
        hf_config={"model_type": "qwen2_moe", "architectures": ["Qwen2MoeForCausalLM"]},
    )
    assert mc.arch == ModelArch.QWEN_MOE
    assert mc.source == "hf_config"
    assert mc.hf_model_type == "qwen2_moe"


def test_dense_generic_from_baichuan_id() -> None:
    snap = resolve_model_arch("baichuan-inc/Baichuan2-7B-Chat")
    assert snap.arch == ModelArch.DENSE_GENERIC


def test_multimodal_qwen_vl_hf_config() -> None:
    snap = resolve_model_arch(
        "/tmp/stub",
        hf_config={
            "model_type": "qwen2_vl",
            "architectures": ["Qwen2VLForConditionalGeneration"],
        },
    )
    assert snap.arch == ModelArch.QWEN
    assert snap.is_multimodal is True
    ignore = merge_ignore_list(["lm_head"], snap.arch, is_multimodal=snap.is_multimodal)
    assert "visual.*" in ignore


def test_qwen_ignore_merges_gate() -> None:
    ignore = merge_ignore_list(["lm_head"], ModelArch.QWEN)
    assert "lm_head" in ignore
    assert "re:.*mlp.gate$" in ignore


def test_qwen_moe_ignore_matches_hf_router_paths() -> None:
    ignore = merge_ignore_list([], ModelArch.QWEN_MOE)
    assert "re:.*mlp\\.gate$" in ignore
    assert "re:.*shared_expert_gate$" in ignore


def test_deepseek_moe_ignore_matches_hf_router_and_shared() -> None:
    ignore = merge_ignore_list([], ModelArch.DEEPSEEK_MOE)
    assert "re:.*mlp\\.gate$" in ignore
    assert "re:.*mlp\\.shared_experts" in ignore
    assert "re:.*kv_a_proj_with_mqa$" in ignore
    assert "re:.*eh_proj$" in ignore


def test_mixtral_moe_ignore_gate() -> None:
    ignore = merge_ignore_list([], ModelArch.MIXTRAL)
    assert "re:.*block_sparse_moe\\.gate$" in ignore


def test_compile_injects_arch_snapshot() -> None:
    plan = compile_plan(RECIPES / "hygon_qwen_w4a16_awq.yaml", "hygon_dcu")
    assert plan.intent.arch_snapshot.arch == ModelArch.QWEN
    assert "re:.*mlp.gate$" in plan.intent.ignore
    assert plan.intent.backend_options.get("model_arch") == "qwen"


def test_moe_scales_calib_samples() -> None:
    plan = compile_plan(RECIPES / "moe_int4.yaml", "generic_cpu")
    assert plan.intent.arch_snapshot.arch == ModelArch.QWEN_MOE
    assert plan.intent.arch_snapshot.is_moe is True
    assert plan.intent.calib.get("max_samples") == 512
    assert plan.intent.calib.get("expert_aware") is True


def test_dense_and_moe_presets_split() -> None:
    from luban_sculpt.model.dense import DENSE_PRESETS
    from luban_sculpt.model.moe import MOE_PRESETS

    assert ModelArch.QWEN in DENSE_PRESETS
    assert ModelArch.QWEN_MOE in MOE_PRESETS
    assert ModelArch.QWEN_MOE not in DENSE_PRESETS
    assert all(arch in MOE_ARCHES for arch in MOE_PRESETS)
    assert all(arch not in MOE_ARCHES for arch in DENSE_PRESETS)
