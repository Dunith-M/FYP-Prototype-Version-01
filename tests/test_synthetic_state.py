import pandas as pd
import inspect

from planning.services.synthetic_state import build_synthetic_l0_q0_fixture, validate_synthetic_l0_q0_fixture


def test_synthetic_fixture_has_deterministic_l0_q0_semantics():
    features = pd.DataFrame([{"branch_code":"BIL01", "canonical_product_id":f"CP_L{i}", "year_month":"2026-03", "lag_1":1} for i in range(15)])
    candidates = pd.DataFrame([{"branch_code":"BIL01", "canonical_product_id":f"CP_C{i}", "candidate_rank":i+1} for i in range(15)])
    options = {"parameter_source":"SYNTHETIC_DEMO", "max_target_per_product":5}
    fixture = build_synthetic_l0_q0_fixture(features, candidates, options)
    validate_synthetic_l0_q0_fixture(fixture, options, branch_codes={"BIL01"}, product_ids=set(fixture.canonical_product_id), candidate_ids=set(candidates.canonical_product_id))
    assert len(fixture) == 30
    assert fixture.loc[fixture.L0.eq(0), "q0"].isna().all()
    assert fixture.loc[fixture.L0.eq(1), "q0"].tolist() == [1, 2, 3, 4, 5] * 3
    assert "inventory_qty" not in inspect.getsource(build_synthetic_l0_q0_fixture)
