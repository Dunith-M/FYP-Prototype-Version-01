from pathlib import Path

import yaml


def test_synthetic_options_are_explicit_demo_parameters_only():
    path = Path(__file__).parents[1] / "config" / "synthetic" / "options.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert config["mode"] == "RESEARCH_SYNTHETIC"
    assert config["parameter_source"] == "SYNTHETIC_DEMO"
    assert config["delta"] == 0
    assert config["default_quantity_step"] == 1
    assert config["max_target_per_product"] == 5
    assert config["constraints"] == {
        "max_skus_per_branch": 150,
        "max_introductions_per_branch": 5,
        "category_minimums": {},
        "budget_limit": None,
        "space_limit": None,
    }
    assert not {"profit_margin", "cost_limit", "actual_budget", "shelf_space", "supplier_limit"} & set(config)
