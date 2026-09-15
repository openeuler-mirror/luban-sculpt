"""Recipe extends: templates/quant.yaml merge."""

from __future__ import annotations

from luban_sculpt.compiler.recipe_compiler import load_recipe_yaml
from luban_sculpt.compiler.recipe_template import merge_recipe_documents
from tests.paths import LLAMA3_EXAMPLE, QWEN25_EXAMPLE, TEMPLATES


def test_quant_template_has_defaults() -> None:
    import yaml

    doc = yaml.safe_load((TEMPLATES / "quant.yaml").read_text(encoding="utf-8"))
    assert doc["pipeline"]["stages"][0]["precision"] == "fp8_dynamic"
    assert doc["pipeline"]["stages"][0]["backend"] == "auto"


def test_llama3_example_extends_quant() -> None:
    doc = load_recipe_yaml(LLAMA3_EXAMPLE, validate_model_layout=False)
    assert doc["model"]["arch"] == "llama"
    assert doc["pipeline"]["stages"][0]["precision"] == "fp8_dynamic"
    assert doc["calib"]["source"] == "open-perfectblend"
    assert doc["pipeline"]["stages"][0]["compress"]["modifiers"]


def test_qwen_example_merges_compress_observer() -> None:
    doc = load_recipe_yaml(QWEN25_EXAMPLE, validate_model_layout=False)
    obs = doc["pipeline"]["stages"][0]["compress"]["observer"]
    assert obs["weights"] == "luban_ema_absmax"
    assert doc["model"]["arch"] == "qwen"


def test_merge_pipeline_stages_by_index() -> None:
    base = {
        "pipeline": {
            "stages": [
                {"name": "a", "precision": "fp8_dynamic", "compress": {"modifiers": [1]}},
            ],
        },
    }
    over = {
        "pipeline": {
            "stages": [{"compress": {"observer": {"weights": "x"}}}],
        },
    }
    merged = merge_recipe_documents(base, over)
    stage = merged["pipeline"]["stages"][0]
    assert stage["precision"] == "fp8_dynamic"
    assert stage["compress"]["observer"]["weights"] == "x"
    assert stage["compress"]["modifiers"] == [1]
