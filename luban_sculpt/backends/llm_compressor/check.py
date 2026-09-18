"""llm-compressor 导入检测（静默、缓存；旧 torch / 无 GPU 栈可能 import 失败）。"""

from __future__ import annotations

import contextlib
import io
import warnings
from typing import Any

from luban_sculpt.log import get_logger

logger = get_logger(__name__)

_LC_STATE: dict[str, Any] | None = None


def get_llm_compressor_state() -> dict[str, Any]:
    """获取 llmcompressor 导入结果（懒加载、模块内缓存）；失败不抛错。

    常见键：``available``、``QuantizationModifier``、``error``；
    ``register_luban_llm_compressor_modifiers()`` 还会在成功时写入 Luban* 类。
    """
    global _LC_STATE
    if _LC_STATE is not None:
        return _LC_STATE

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

                _LC_STATE = {
                    "available": True,
                    "oneshot": oneshot,
                    "Event": Event,
                    "State": State,
                    "Modifier": Modifier,
                    "QuantizationModifier": QuantizationModifier,
                }
            except Exception as exc:  # noqa: BLE001 — import 检测不得 raise
                _LC_STATE = {"available": False, "error": repr(exc)}
    return _LC_STATE


def is_llm_compressor_available() -> bool:
    return bool(get_llm_compressor_state().get("available"))


def get_llm_compressor_quantization_modifier_class() -> type[Any] | None:
    """LC 可用时返回 ``QuantizationModifier`` 类，否则 ``None``。"""
    state = get_llm_compressor_state()
    if not state.get("available"):
        return None
    return state["QuantizationModifier"]


def register_luban_llm_compressor_modifiers() -> dict[str, Any]:
    """LC 可用时注册 Luban* Modifier 子类到 state；返回 ``get_llm_compressor_state()``。"""
    state = get_llm_compressor_state()
    if not state.get("available"):
        return state
    if "LubanHALCalibModifier" in state:
        return state

    modifier_base = state["Modifier"]
    lc_state_cls = state["State"]
    lc_event_cls = state["Event"]

    class LubanHALCalibModifier(modifier_base):
        """校准开始/结束时写入 HAL 选中的 kernel（挂到 state.metadata）。"""

        calib_kernel: str = "generic_fake_quant"
        profile_name: str = "generic_cpu"

        def on_initialize(self, state: lc_state_cls, **kwargs) -> bool:
            meta = getattr(state, "metadata", None)
            if meta is not None:
                meta["luban_hal_calib_kernel"] = self.calib_kernel
                meta["luban_profile_id"] = self.profile_name
            logger.info(
                "LubanHALCalibModifier init kernel=%s profile=%s",
                self.calib_kernel,
                self.profile_name,
            )
            return True

        def on_calibration_start(
            self, state: lc_state_cls, event: lc_event_cls, **kwargs
        ) -> None:
            logger.debug("calibration_start hal_kernel=%s", self.calib_kernel)

    class LubanProducerMetadataModifier(modifier_base):
        """Finalize 前写入 luban-sculpt producer 信息，便于 export 对账。"""

        producer: dict[str, Any] | None = None

        def on_initialize(self, state: lc_state_cls, **kwargs) -> bool:
            logger.debug(
                "LubanProducerMetadataModifier init producer=%s",
                self.producer,
            )
            return True

        def on_finalize(self, state: lc_state_cls, **kwargs) -> bool:
            meta = getattr(state, "metadata", None)
            if meta is not None and self.producer:
                meta["luban_producer"] = dict(self.producer)
            return True

    state["LubanHALCalibModifier"] = LubanHALCalibModifier
    state["LubanProducerMetadataModifier"] = LubanProducerMetadataModifier
    return state
