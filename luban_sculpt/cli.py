"""luban-sculpt CLI: compress / validate / report / probe."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from luban_sculpt.log import LOG_LEVEL_NAMES, configure_logging, get_logger
from luban_sculpt.backends.base import BackendRegistry
from luban_sculpt.hae.engine import HardwareAwareEngine
from luban_sculpt.pipeline import QuantPipeline
from luban_sculpt.validate.hf_config import validate_hf_config
from luban_sculpt.validate.runtime import RuntimeMode, validate_runtime

logger = get_logger(__name__)


def _cmd_probe(args: argparse.Namespace) -> int:
    """HAE 探测并输出 JSON（profile、probe、hw_decision）。"""
    logger.info("cmd probe profile=%s", args.profile)
    try:
        hae = HardwareAwareEngine(args.profile)
        hw, probe, profile = hae.run(None if args.profile == "auto" else args.profile)
    except Exception:
        logger.error("probe failed profile=%s", args.profile, exc_info=True)
        return 1
    out = {
        "profile_id": hw.profile_id,
        "vendor": hw.vendor,
        "probe_ok": probe.ok,
        "missing_ops": probe.missing_ops,
        "hw_decision": hw.model_dump(mode="json"),
        "profile_keys": list(profile.get("schemes", {}).keys()),
    }
    if not probe.ok:
        logger.error("probe not ok missing_ops=%s", probe.missing_ops)
    else:
        logger.info("probe ok profile_id=%s", hw.profile_id)
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0 if probe.ok else 1


def _cmd_compress(args: argparse.Namespace) -> int:
    """主路径：QuantPipeline.run。"""
    logger.info(
        "cmd compress profile=%s recipe=%s output=%s",
        args.profile,
        args.recipe,
        args.output,
    )
    try:
        pipe = QuantPipeline(
            profile_name=args.profile,
            validate_quantized_model=bool(
                getattr(args, "validate_quantized_model", False)
            ),
            validate_runtime=bool(getattr(args, "validate_runtime", False)),
            runtime_mode=getattr(args, "runtime_mode", None) or "import",
        )
        artifact = pipe.run(Path(args.recipe), Path(args.output))
    except Exception:
        logger.error(
            "compress failed profile=%s recipe=%s output=%s",
            args.profile,
            args.recipe,
            args.output,
            exc_info=True,
        )
        return 1
    logger.info("compress ok output=%s", artifact.output_dir)
    print(
        json.dumps(
            {
                "output": str(artifact.output_dir),
                "manifest": artifact.manifest.model_dump(mode="json"),
            },
            indent=2,
        )
    )
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    """量化后校验：HF 配置 + 可选运行时。"""
    model_path = Path(args.model)
    check_runtime = bool(args.runtime)
    runtime_mode: RuntimeMode = args.runtime_mode or "import"

    logger.info(
        "cmd validate model=%s hf_config=True runtime=%s mode=%s",
        args.model,
        check_runtime,
        runtime_mode,
    )
    manifest_path = model_path / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        print(
            f"manifest profile_id={manifest.get('profile_id')} "
            f"export={manifest.get('export_format')}"
        )

    errors = list(validate_hf_config(model_path))
    if check_runtime:
        result = validate_runtime(model_path, mode=runtime_mode)
        if not result.get("ok"):
            errors.append(
                result.get("error") or result.get("stderr", "runtime check failed")
            )
        else:
            print("runtime:", result)

    if errors:
        for e in errors:
            logger.error("%s", e)
            print("ERROR:", e, file=sys.stderr)
        return 1
    logger.info("validate ok")
    print("validate ok")
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    path = Path(args.model) / "manifest.json"
    if not path.is_file():
        logger.error("no manifest.json under %s", args.model)
        print("no manifest.json", file=sys.stderr)
        return 1
    print(path.read_text(encoding="utf-8"))
    return 0


def _cmd_backends(_args: argparse.Namespace) -> int:
    reg = BackendRegistry()
    print(json.dumps(reg.registered(), indent=2))
    return 0


def _cmd_ascend_chips(_args: argparse.Namespace) -> int:
    from luban_sculpt.backends.msmodelslim.chips import ASCEND_CHIPS

    rows = []
    for spec in ASCEND_CHIPS.values():
        rows.append(
            {
                "profile_name": spec.profile_name,
                "soc_key": spec.soc_key,
                "soc": spec.soc,
                "peak_tflops_fp16": spec.peak_tflops_fp16,
                "fp8_native": spec.fp8_native,
                "allowed_quant_types": list(spec.allowed_quant_types),
                "default_quant_type": spec.default_quant_type,
                "default_scheme": spec.default_abstract_scheme,
                "notes": spec.notes,
            }
        )
    print(json.dumps(rows, indent=2, ensure_ascii=False))
    return 0


def _cmd_model_arches(_args: argparse.Namespace) -> int:
    from luban_sculpt.model import list_arches

    print(json.dumps(list_arches(), indent=2, ensure_ascii=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="luban-sculpt")
    parser.add_argument(
        "--log-level",
        default="info",
        choices=LOG_LEVEL_NAMES,
        help="Log verbosity: debug, info, warn, error",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    probe_parser = sub.add_parser("probe", help="HAE detect + profile probe")
    probe_parser.add_argument("--profile", default="auto")
    probe_parser.set_defaults(func=_cmd_probe)

    compress_parser = sub.add_parser("compress", help="Run quant pipeline (stub HAL + manifest)")
    compress_parser.add_argument("--recipe", required=True)
    compress_parser.add_argument("--output", required=True)
    compress_parser.add_argument("--profile", default="auto")
    compress_parser.add_argument(
        "--validate-quantized-model",
        action="store_true",
        help="After compress: HF config checks (QuantizedModelValidateStage)",
    )
    compress_parser.add_argument(
        "--validate-runtime",
        action="store_true",
        help="With --validate-quantized-model, also run vLLM runtime check",
    )
    compress_parser.add_argument(
        "--runtime-mode",
        choices=["import", "load", "generate"],
        default="import",
        help="Runtime check mode when --validate-runtime is set",
    )
    compress_parser.set_defaults(func=_cmd_compress)

    validate_parser = sub.add_parser(
        "validate",
        help="Post-quant validate: HF config.json + optional vLLM runtime",
    )
    validate_parser.add_argument("--model", required=True)
    validate_parser.add_argument(
        "--runtime",
        action="store_true",
        help="Also run vLLM runtime check (import/load/generate)",
    )
    validate_parser.add_argument(
        "--runtime-mode",
        choices=["import", "load", "generate"],
        default="import",
        help="Runtime mode (default: import)",
    )
    validate_parser.set_defaults(func=_cmd_validate)

    report_parser = sub.add_parser("report", help="Print manifest.json")
    report_parser.add_argument("--model", required=True)
    report_parser.set_defaults(func=_cmd_report)

    backends_parser = sub.add_parser("backends", help="List registered backends")
    backends_parser.set_defaults(func=_cmd_backends)

    ascend_chips_parser = sub.add_parser("ascend-chips", help="List Ascend SoC quant policy")
    ascend_chips_parser.set_defaults(func=_cmd_ascend_chips)

    model_arches_parser = sub.add_parser("model-arches", help="List model arch quant policies")
    model_arches_parser.set_defaults(func=_cmd_model_arches)

    args = parser.parse_args(argv)
    configure_logging(level=args.log_level, force=True)
    logger.info("luban-sculpt command=%s log_level=%s", args.command, args.log_level)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
