"""Hardware-Aware Engine (HAE): detect → probe → synthesize HwDecision."""

from __future__ import annotations

import os
from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml

from luban_sculpt.contracts import (
    CalibFusion,
    Fp8Encoding,
    HardwareCapability,
    HwDecision,
    ParallelTopology,
    ProbeResult,
    WeightLayout,
)
from luban_sculpt.hae.capability import build_hardware_capability
from luban_sculpt.hae.detect_hw import (
    detect_hardware,
    detect_topology_cann,
    detect_topology_nvml,
)
from luban_sculpt.hae.probe_ops import probe_one_op
from luban_sculpt.log import get_logger

logger = get_logger(__name__)


def _profiles_dir() -> Path:
    return Path(str(files("luban_sculpt").joinpath("profiles")))


def load_profile_template(name: str) -> dict[str, Any]:
    """加载 profiles/{name}.yaml 芯片能力模板。"""
    path = _profiles_dir() / f"{name}.yaml"
    if not path.is_file():
        logger.error("Profile not found: %s (%s)", name, path)
        raise FileNotFoundError(f"Profile not found: {name} ({path})")
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _version_lt(current: str, minimum: str) -> bool:
    """简易版本比较：按非数字切分后逐段比；无法解析则视为不小于。"""
    if not current or current == "unknown" or not minimum:
        return False

    def _parts(v: str) -> list[int | str]:
        out: list[int | str] = []
        buf = ""
        for ch in v:
            if ch.isdigit():
                buf += ch
            else:
                if buf:
                    out.append(int(buf))
                    buf = ""
                if ch.isalnum():
                    out.append(ch.lower())
        if buf:
            out.append(int(buf))
        return out

    a, b = _parts(current), _parts(minimum)
    for i in range(max(len(a), len(b))):
        x = a[i] if i < len(a) else 0
        y = b[i] if i < len(b) else 0
        if type(x) is not type(y):
            x, y = str(x), str(y)
        if x < y:
            return True
        if x > y:
            return False
    return False


class HardwareAwareEngine:
    """硬件感知引擎：探测环境 → 选 Profile → Probe → 合成 HwDecision 供 HAL/Gate 使用。"""

    def __init__(
        self,
        profile_name: str | None = None,
        *,
        enable_real_probe: bool = True,
        probe_timeout_sec: float = 30.0,
        fallback_on_probe_failure: bool = True,
        latency_thresholds: dict[str, float] | None = None,
        mse_threshold: float = 0.01,
    ) -> None:
        # "auto" 时在 run() 里按环境变量/设备再解析
        self.profile_name = profile_name or "generic_cpu"
        self.enable_real_probe = enable_real_probe
        self.probe_timeout_sec = probe_timeout_sec
        self.fallback_on_probe_failure = fallback_on_probe_failure
        self.latency_thresholds = dict(latency_thresholds or {})
        self.mse_threshold = mse_threshold

        self._identity: dict[str, Any] = {}
        self._topology: ParallelTopology = ParallelTopology.UNKNOWN
        self._capability: HardwareCapability | None = None

    @property
    def capability(self) -> HardwareCapability | None:
        return self._capability

    def detect(self) -> dict[str, Any]:
        """识别 vendor / 设备 / 软件栈：API → smi → 环境变量。"""
        raw = detect_hardware()
        self._identity = {
            "device_name": raw.get("device_name", "unknown"),
            "vendor": raw.get("vendor", "unknown"),
            "stack": dict(raw.get("stack") or {}),
            "source": raw.get("source", "env"),
        }
        for key in (
            "compute_capability",
            "memory_bytes",
            "sm_count",
            "driver_version",
            "fp8_native",
            "memory_bandwidth_gbps",
            "peak_tflops_fp16",
        ):
            if key in raw and raw[key] is not None:
                self._identity[key] = raw[key]
        self._capability = self._query_hardware_capability()
        self._identity["device_name"] = self._capability.device_name
        self._identity["vendor"] = self._capability.vendor
        self._identity["stack"] = dict(self._capability.stack)
        logger.info(
            "detect vendor=%s device=%s source=%s fp8_native=%s",
            self._capability.vendor,
            self._capability.device_name,
            self._capability.source,
            self._capability.fp8_native,
        )
        return self._identity

    def _query_hardware_capability(self) -> HardwareCapability:
        """根据 detect() 结果构建 HardwareCapability（查表补齐 FP8/带宽/算力）。"""
        extras = {
            "device_name": self._identity.get("device_name"),
            "vendor": self._identity.get("vendor"),
            "stack": self._identity.get("stack"),
            "source": self._identity.get("source"),
        }
        for key in (
            "compute_capability",
            "memory_bytes",
            "sm_count",
            "driver_version",
            "fp8_native",
            "memory_bandwidth_gbps",
            "peak_tflops_fp16",
        ):
            if key in self._identity:
                extras[key] = self._identity[key]
        return build_hardware_capability(self._identity, extras=extras)

    def detect_topology(self) -> ParallelTopology:
        """多卡互联拓扑：NVML/CANN API → LUBAN_TOPOLOGY → vendor 启发式。"""
        mapping = {
            "hccs_ring": ParallelTopology.HCCS_RING,
            "hccs_mesh": ParallelTopology.HCCS_MESH,
            "pcie_switch": ParallelTopology.PCIE_SWITCH,
        }

        api_topo: str | None = None
        vendor = (self._identity.get("vendor") or "").lower()
        if vendor == "nvidia":
            api_topo = detect_topology_nvml()
        elif vendor == "ascend":
            api_topo = detect_topology_cann()

        if api_topo and api_topo in mapping:
            self._topology = mapping[api_topo]
            return self._topology

        topo = os.environ.get("LUBAN_TOPOLOGY", "").lower()
        if topo in mapping:
            self._topology = mapping[topo]
            return self._topology

        # vendor heuristic fallback
        if vendor == "ascend":
            self._topology = ParallelTopology.HCCS_RING
        elif vendor == "nvidia":
            self._topology = ParallelTopology.PCIE_SWITCH
        elif vendor == "hygon":
            self._topology = ParallelTopology.PCIE_SWITCH
        else:
            self._topology = ParallelTopology.UNKNOWN
        logger.info("topology=%s", self._topology.value)
        return self._topology

    def resolve_profile_name(self, identity: dict[str, Any] | None = None) -> str:
        """非 auto 时返回用户指定 profile；auto 时按算力/FP8/SoC/vendor 映射。"""
        if self.profile_name and self.profile_name != "auto":
            return self.profile_name

        from luban_sculpt.hae.capability import match_profile_id

        cap = self._capability
        identity = identity or self._identity
        vendor = (identity.get("vendor") or (cap.vendor if cap else "unknown")).lower()
        device = (identity.get("device_name") or (cap.device_name if cap else "")).lower()
        gpu = os.environ.get("LUBAN_NVIDIA_GPU", "").lower()

        # profiles.match_keys 优先（设备名子串 → profile id）
        matched = match_profile_id(device) or match_profile_id(gpu)
        if matched:
            return matched

        # Ascend SoC（环境变量 / 芯片表）
        soc = os.environ.get("LUBAN_ASCEND_SOC") or os.environ.get("ASCEND_SOC_VERSION")
        if soc or vendor == "ascend":
            from luban_sculpt.backends.msmodelslim.chips import normalize_soc_key

            key = normalize_soc_key(soc or device)
            if key:
                from luban_sculpt.backends.msmodelslim.chips import ASCEND_CHIPS

                return ASCEND_CHIPS[key].profile_name

        # NVIDIA：本仓库仅保留 H20 profile
        if vendor == "nvidia":
            return "nvidia_h20"

        if vendor == "ascend":
            return "ascend_910b"
        if vendor == "hygon":
            return "hygon_dcu"
        return "generic_cpu"

    def probe(self, profile: dict[str, Any]) -> ProbeResult:
        """三阶段探测：栈版本 →（可选）真实 Kernel → 汇总；支持失败降级。"""
        messages: list[str] = []
        missing: list[str] = []
        benchmark: dict[str, Any] = {"ops": []}
        stack = self._identity.get("stack", {})
        vendor = self._identity.get("vendor", "unknown")

        # --- stage 1: stack gates ---
        gates = profile.get("stack_gates", {})
        for scheme, gate in gates.items():
            min_cann = gate.get("min_cann")
            cann = stack.get("cann")
            if min_cann and cann and cann != "unknown" and _version_lt(str(cann), str(min_cann)):
                missing.append(f"stack:{scheme}:cann>={min_cann}")
                messages.append(f"{scheme} needs CANN>={min_cann}, got {cann}")

            min_cuda = gate.get("min_cuda")
            cuda = stack.get("cuda")
            if min_cuda and cuda and cuda != "unknown" and _version_lt(str(cuda), str(min_cuda)):
                missing.append(f"stack:{scheme}:cuda>={min_cuda}")
                messages.append(f"{scheme} needs CUDA>={min_cuda}, got {cuda}")

            min_dtk = gate.get("min_dtk")
            dtk = stack.get("dtk")
            if min_dtk and dtk and dtk != "unknown" and _version_lt(str(dtk), str(min_dtk)):
                missing.append(f"stack:{scheme}:dtk>={min_dtk}")
                messages.append(f"{scheme} needs DTK>={min_dtk}, got {dtk}")

        # --- stage 2: kernel probe ---
        ops = list(profile.get("required_ops_probe", []))
        if self.enable_real_probe:
            for op in ops:
                thr = self.latency_thresholds.get(op)
                result = probe_one_op(
                    op,
                    vendor=vendor,
                    timeout_sec=self.probe_timeout_sec,
                    latency_threshold_ms=thr,
                    mse_threshold=self.mse_threshold,
                )
                messages.append(result["message"])
                benchmark["ops"].append(result)
                if not result["ok"]:
                    missing.append(op)
        else:
            fail_ops = {
                x.strip()
                for x in os.environ.get("LUBAN_PROBE_FAIL_OPS", "").split(",")
                if x.strip()
            }
            for op in ops:
                if op in fail_ops:
                    missing.append(op)
                    messages.append(f"probe op {op}: skipped-fail (real probe disabled)")
                else:
                    messages.append(f"probe op {op}: ok (real probe disabled)")

        # --- stage 3: summarize + optional degrade ---
        hard_fail = len(missing) > 0
        degraded = False
        if hard_fail and self.fallback_on_probe_failure:
            degraded = True
            messages.append(
                "probe degraded: fallback_on_probe_failure=True; "
                f"missing={missing}"
            )
            ok = True
        else:
            ok = not hard_fail

        if benchmark["ops"]:
            p50s = [
                o["latency_ms_p50"]
                for o in benchmark["ops"]
                if o.get("latency_ms_p50") is not None
            ]
            benchmark["summary"] = {
                "ops_total": len(benchmark["ops"]),
                "ops_failed": sum(1 for o in benchmark["ops"] if not o.get("ok")),
                "latency_ms_p50_avg": (sum(p50s) / len(p50s)) if p50s else None,
            }

        return ProbeResult(
            ok=ok,
            messages=messages,
            missing_ops=missing,
            degraded=degraded,
            benchmark=benchmark,
        )

    def synthesize_decision(
        self, profile: dict[str, Any], probe: ProbeResult
    ) -> HwDecision:
        """将 profile.hw_defaults 与探测结果合并为 HwDecision（驱动 HAL encoding/layout）。"""
        defaults = profile.get("hw_defaults", {})
        vendor = profile.get("vendor", self._identity.get("vendor", "unknown"))

        def _enum(enum_cls, key, default):
            raw = defaults.get(key, default)
            try:
                return enum_cls(raw)
            except ValueError:
                return enum_cls(default)

        topo_default = profile.get("topology_default", ParallelTopology.UNKNOWN.value)
        if self._topology != ParallelTopology.UNKNOWN:
            parallel_topology = self._topology
        else:
            try:
                parallel_topology = ParallelTopology(topo_default)
            except ValueError:
                parallel_topology = ParallelTopology.UNKNOWN

        from luban_sculpt.hae.profile_fields import profile_soc_key

        ascend_soc = profile_soc_key(profile)

        return HwDecision(
            profile_id=profile.get("id", self.profile_name),
            vendor=vendor,
            ascend_soc=str(ascend_soc) if ascend_soc else None,
            calib_fusion=_enum(
                CalibFusion, "calib_fusion", CalibFusion.GENERIC.value
            ),
            fp8_encoding=_enum(
                Fp8Encoding, "fp8_encoding", Fp8Encoding.IEEE_E4M3_REF.value
            ),
            weight_layout=_enum(
                WeightLayout, "weight_layout", WeightLayout.CT_PACK.value
            ),
            parallel_topology=parallel_topology,
            tp_split_hint=defaults.get("tp_split_hint", "column"),
            comm_backend=defaults.get(
                "comm_backend", "hccl" if vendor == "ascend" else "nccl"
            ),
            stack_snapshot=dict(self._identity.get("stack", {})),
            probe_passed_ops=[m for m in probe.messages if "ok" in m and "fail" not in m],
            probe_failed_ops=probe.missing_ops,
        )

    def run(self, profile_name: str | None = None) -> tuple[HwDecision, ProbeResult, dict]:
        """完整 HAE 流水线：detect → resolve profile YAML → probe → HwDecision。"""
        if profile_name:
            self.profile_name = profile_name
        logger.info("HAE pipeline start requested_profile=%s", self.profile_name)
        self.detect()
        self.detect_topology()
        resolved = self.resolve_profile_name()
        logger.info("HAE resolved profile=%s", resolved)
        try:
            profile = load_profile_template(resolved)
        except Exception:
            logger.error("failed to load profile template %s", resolved, exc_info=True)
            raise
        probe = self.probe(profile)
        logger.info(
            "HAE probe ok=%s degraded=%s missing_ops=%s",
            probe.ok,
            probe.degraded,
            probe.missing_ops,
        )
        if not probe.ok:
            logger.error("HAE probe failed missing_ops=%s", probe.missing_ops)
        decision = self.synthesize_decision(profile, probe)
        logger.info(
            "HAE decision profile_id=%s vendor=%s layout=%s encoding=%s",
            decision.profile_id,
            decision.vendor,
            decision.weight_layout.value,
            decision.fp8_encoding.value,
        )
        return decision, probe, profile
