"""MoE arch quant policies (router / expert aware).

HF / vLLM module paths (representative):
- Qwen2/3 MoE: ``...mlp.gate``, ``...mlp.shared_expert_gate``, ``...mlp.shared_expert.*``
- DeepSeek V2/V3 MoE: ``...mlp.gate``, ``...mlp.shared_experts.*``; MLA attn ``kv_a_proj_with_mqa``等
- Mixtral: ``...block_sparse_moe.gate``
"""

from __future__ import annotations

from luban_sculpt.model.types import ArchQuantPolicy, ModelArch

MOE_PRESETS: dict[ModelArch, ArchQuantPolicy] = {
    ModelArch.QWEN_MOE: ArchQuantPolicy(
        default_ignore=(
            "lm_head",
            "re:.*mlp\\.gate$",  # Qwen2MoeSparseMoeBlock.gate (not gate_proj)
            "re:.*shared_expert_gate$",
        ),
        quant_hints={
            "expert_aware": True,
            "calib_samples_factor": 2.0,
            "router_fp16": True,
        },
        notes="Qwen2/3 MoE: mlp.gate + shared_expert_gate (see HF ...mlp.gate weight)",
    ),
    ModelArch.DEEPSEEK_MOE: ArchQuantPolicy(
        default_ignore=(
            "lm_head",
            "re:.*mlp\\.gate$",  # DeepseekV2MoE.gate
            "re:.*mlp\\.shared_experts",  # shared_experts.* (plural in HF/vLLM)
            "re:.*kv_a_proj_with_mqa$",  # MLA low-rank KV path
            "re:.*eh_proj$",  # MTP / 部分变体上的 embedding-head proj
        ),
        quant_hints={
            "expert_aware": True,
            "mla_aware": True,
            "calib_samples_factor": 2.0,
            "router_fp16": True,
        },
        notes="DeepSeek MoE: router + shared_experts; MLA kv_a; optional eh_proj (MTP)",
    ),
    ModelArch.MIXTRAL: ArchQuantPolicy(
        default_ignore=(
            "lm_head",
            "re:.*block_sparse_moe\\.gate$",
        ),
        quant_hints={"expert_aware": True, "router_fp16": True},
        notes="Mixtral: block_sparse_moe.gate",
    ),
    ModelArch.MOE_GENERIC: ArchQuantPolicy(
        default_ignore=(
            "lm_head",
            "re:.*block_sparse_moe\\.gate$",
            "re:.*mlp\\.gate$",
            "re:.*shared_expert_gate$",
        ),
        quant_hints={"expert_aware": True, "router_fp16": True},
        notes="Fallback: common MoE router / shared-expert-gate paths",
    ),
}
