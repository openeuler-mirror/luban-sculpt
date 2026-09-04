"""ModifierManager / ChainModifier 单元测试。"""

from __future__ import annotations

from luban_sculpt.contracts import BackendPlan, ExportFormat, HwDecision, QuantIntent
from luban_sculpt.modifiers.base import ChainModifier
from luban_sculpt.modifiers.manager import ModifierManager
from luban_sculpt.modifiers.recipe import build_recipe_for_plan
from luban_sculpt.modifiers.registry import list_modifier_names, resolve_modifier_class


def _plan(*, modifiers=None) -> BackendPlan:
    opts = {}
    if modifiers is not None:
        opts["modifiers"] = modifiers
    return BackendPlan(
        intent=QuantIntent(
            model_id="m",
            backend="llm_compressor",
            abstract_scheme="fp8_dynamic",
            deploy_target="vllm_cuda",
            backend_options=opts,
        ),
        hw=HwDecision(profile_id="generic_cpu"),
        export_format=ExportFormat.COMPRESSED_TENSORS,
    )


def test_list_builtin_modifier_names() -> None:
    names = list_modifier_names()
    assert "HALCalibHook" in names
    assert "DomesticFakeQuant" in names
    assert resolve_modifier_class("HALCalibHook") is not None


def test_manager_apply_append_stub() -> None:
    class _Append(ChainModifier):
        name = "AppendStub"

        def build_modifier(self, plan, spec):
            return {"stub": spec.get("tag", "x")}

    plan = _plan()
    mgr = ModifierManager(plan, base_builder=lambda _p: [{"QuantizationModifier": {}}])
    # 直接注入已解析实例路径：走 apply 的 specs；此处手工调用 intercept
    chain = [{"QuantizationModifier": {}}]
    chain = _Append().intercept(chain, plan, {"mode": "append", "tag": "a"})
    assert len(chain) == 2
    assert chain[1] == {"stub": "a"}


def test_build_recipe_for_plan_dry_run() -> None:
    plan = _plan(
        modifiers=[
            {"name": "HALCalibHook", "mode": "prepend"},
            {"name": "DomesticFakeQuant", "mode": "append"},
        ]
    )
    recipe, meta = build_recipe_for_plan(plan)
    assert meta["modifier_count"] >= 1
    assert any("HALCalibHook" in x for x in meta["interceptor_log"])
    assert any("DomesticFakeQuant" in x for x in meta["interceptor_log"])
    assert recipe is not None


def test_lc_manager_defaults() -> None:
    from luban_sculpt.modifiers.recipe import (
        LLMCompressorModifierManager,
        ModifierInterceptor,
    )

    assert ModifierInterceptor is LLMCompressorModifierManager
    plan = _plan()
    mgr = LLMCompressorModifierManager(plan)
    assert mgr.base_builder is not None
