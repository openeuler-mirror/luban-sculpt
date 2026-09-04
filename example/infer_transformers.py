#!/usr/bin/env python3
"""用 Transformers 原生接口加载 luban-sculpt / llm-compressor 量化产物并推理。

依赖::
    pip install transformers accelerate torch
    # W4A16 compressed-tensors 还需：
    pip install compressed-tensors

用法::
    python example/infer_transformers.py --model ./out-h20-llama3-w4a16
    python example/infer_transformers.py --model /data/out/h20-llama3-w4a16 \\
        --prompt "Hello, my name is" --max-new-tokens 64
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def resolve_model_dir(path: Path) -> Path:
    """兼容 pipeline 产物：优先含 config.json 的目录，否则找 stage_* 子目录。"""
    path = path.expanduser().resolve()
    if (path / "config.json").is_file():
        return path
    for child in sorted(path.glob("stage_*")):
        if (child / "config.json").is_file():
            return child
    raise FileNotFoundError(
        f"未找到 config.json：{path}（或 {path}/stage_*/config.json）"
    )


def load_manifest_hint(model_dir: Path) -> dict:
    for cand in (model_dir / "manifest.json", model_dir.parent / "manifest.json"):
        if cand.is_file():
            return json.loads(cand.read_text(encoding="utf-8"))
    return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Transformers 推理量化模型")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("./out-h20-llama3-w4a16"),
        help="量化产物目录（luban-sculpt compress 的 --output）",
    )
    parser.add_argument("--prompt", default="Hello, my name is")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "bfloat16", "auto"],
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help='如 "auto" / "cuda:0"',
    )
    args = parser.parse_args()

    model_dir = resolve_model_dir(args.model)
    manifest = load_manifest_hint(model_dir)
    print(f"model_dir={model_dir}")
    if manifest:
        print(
            f"manifest scheme={manifest.get('abstract_scheme')} "
            f"export={manifest.get('export_format')} "
            f"profile={manifest.get('profile_id')}"
        )

    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "auto": "auto",
    }[args.dtype]

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=dtype,
        device_map=args.device_map,
        trust_remote_code=True,
    )
    model.eval()

    inputs = tokenizer(args.prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    text = tokenizer.decode(out[0], skip_special_tokens=True)
    print("========== GENERATION ==========")
    print(text)
    print("================================")


if __name__ == "__main__":
    main()
