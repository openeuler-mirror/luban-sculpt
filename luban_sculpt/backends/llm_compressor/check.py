"""llm-compressor availability probe — quiet, cached (Intel Mac / old torch may fail)."""

from __future__ import annotations

import contextlib
import io
import warnings
from typing import Any

_STATE: dict[str, Any] | None = None


def probe_llm_compressor() -> dict[str, Any]:
    """Import llmcompressor once; swallow stdout/stderr/warnings from broken stacks."""
    global _STATE
    if _STATE is not None:
        return _STATE

    buf = io.StringIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with contextlib.redirect_stderr(buf), contextlib.redirect_stdout(buf):
            try:
                from llmcompressor import oneshot
                from llmcompressor.core.events import Event
                from llmcompressor.core.state import State
                from llmcompressor.modifiers import Modifier
                from llmcompressor.modifiers.quantization import QuantizationModifier
                import luban_sculpt.observers  # noqa: F401 — register custom observers

                _STATE = {
                    "available": True,
                    "oneshot": oneshot,
                    "Event": Event,
                    "State": State,
                    "Modifier": Modifier,
                    "QuantizationModifier": QuantizationModifier,
                }
            except Exception as exc:  # noqa: BLE001 — probe must never raise
                _STATE = {"available": False, "error": repr(exc)}
    return _STATE


def is_llm_compressor_available() -> bool:
    return bool(probe_llm_compressor().get("available"))


def ensure_llm_compressor_modifier_classes() -> dict[str, Any]:
    """Lazily define Luban* Modifier subclasses when llmcompressor is importable."""
    probe_state = probe_llm_compressor()
    if not probe_state.get("available"):
        return probe_state
    if "LubanHALCalibModifier" in probe_state:
        return probe_state

    import logging

    modifier_base = probe_state["Modifier"]
    lc_state_cls = probe_state["State"]
    lc_event_cls = probe_state["Event"]
    log = logging.getLogger("luban_sculpt.modifiers.builtins")

    class LubanHALCalibModifier(modifier_base):
        """校准开始/结束时写入 HAL 选中的 kernel（挂到 state.metadata）。"""

        calib_kernel: str = "generic_fake_quant"
        profile_name: str = "generic_cpu"

        def on_initialize(self, state: lc_state_cls, **kwargs) -> bool:
            meta = getattr(state, "metadata", None)
            if meta is not None:
                meta["luban_hal_calib_kernel"] = self.calib_kernel
                meta["luban_profile_id"] = self.profile_name
            log.info(
                "LubanHALCalibModifier init kernel=%s profile=%s",
                self.calib_kernel,
                self.profile_name,
            )
            return True

        def on_calibration_start(
            self, state: lc_state_cls, event: lc_event_cls, **kwargs
        ) -> None:
            log.debug("calibration_start hal_kernel=%s", self.calib_kernel)

    class LubanProducerMetadataModifier(modifier_base):
        """Finalize 前写入 luban-sculpt producer 信息，便于 export 对账。"""

        producer: dict[str, Any] | None = None

        def on_initialize(self, state: lc_state_cls, **kwargs) -> bool:
            log.debug(
                "LubanProducerMetadataModifier init producer=%s",
                self.producer,
            )
            return True

        def on_finalize(self, state: lc_state_cls, **kwargs) -> bool:
            meta = getattr(state, "metadata", None)
            if meta is not None and self.producer:
                meta["luban_producer"] = dict(self.producer)
            return True

    probe_state["LubanHALCalibModifier"] = LubanHALCalibModifier
    probe_state["LubanProducerMetadataModifier"] = LubanProducerMetadataModifier
    return probe_state
