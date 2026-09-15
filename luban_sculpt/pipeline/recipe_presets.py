"""Recipe 校准 / compress 预设（CLI --calib-preset 选用）。"""

from __future__ import annotations

from typing import Any

# preset 名 → 与 recipe 模板合并的 calib + stage.compress 片段
CALIB_PRESETS: dict[str, dict[str, Any]] = {
    "standard": {
        "calib": {
            "source": "open-perfectblend",
            "path": "./open-perfectblend",
            "max_samples": 128,
            "max_seq_length": 512,
            "max_chars": 2048,
            "batch_size": 1,
            "shuffle": True,
            "seed": 42,
            "distributed": False,
            "world_size": 1,
        },
        "compress": {},
    },
    "fast": {
        "calib": {
            "source": "open-perfectblend",
            "path": "./open-perfectblend",
            "max_samples": 32,
            "max_seq_length": 256,
            "max_chars": 1024,
            "batch_size": 4,
            "shuffle": True,
            "seed": 42,
            "distributed": False,
            "world_size": 1,
        },
        "compress": {"pipeline": False},
    },
    "stub": {
        "calib": {
            "source": "stub",
            "max_samples": 128,
            "batch_size": 1,
            "distributed": False,
            "world_size": 1,
        },
        "compress": {},
    },
}

# --precision 触发的单阶段 compress / calib 补丁（如 w4a16 GPTQ）
PRECISION_OVERLAYS: dict[str, dict[str, Any]] = {
    "w4a16": {
        "calib": {
            "source": "open-perfectblend",
            "path": "./open-perfectblend",
            "max_samples": 64,
            "max_seq_length": 512,
            "max_chars": 2048,
            "batch_size": 1,
            "shuffle": True,
            "seed": 42,
            "distributed": False,
            "world_size": 1,
        },
        "compress": {
            "algorithm": "gptq",
            "scheme": "W4A16",
            "block_size": 128,
            "dampening_frac": 0.01,
            "quantization_format": "pack-quantized",
            "pipeline": "sequential",
            "gptq_max_samples_cap": 64,
            "gptq_max_seq_cap": 512,
            "oneshot": {},
        },
    },
}

# --pipeline-preset 替换整条 pipeline.stages（多阶段编配）
PIPELINE_PRESETS: dict[str, dict[str, Any]] = {
    "fp8_then_gptq": {
        "calib": {
            "source": "stub",
            "max_samples": 64,
        },
        "stages": [
            {
                "name": "fp8_prep",
                "backend": "auto",
                "precision": "fp8_dynamic",
                "ignore": ["lm_head"],
                "output_subdir": "stage1_fp8",
            },
            {
                "name": "gptq_w4",
                "backend": "gptq",
                "algorithm": "gptq",
                "abstract_scheme": "w4_gptq",
                "input_from": "previous",
                "output_subdir": "stage2_gptq",
                "backend_options": {"bits": 4, "group_size": 128},
            },
        ],
    },
}
