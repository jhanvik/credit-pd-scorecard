"""Tests for the one-time final evaluation machinery. The real test split is never loaded here."""
import numpy as np
import pandas as pd
import pytest

from credit_pd_scorecard import config, final_test, inference

SPEC = {"cuts": [10.0, 20.0], "woe": [0.5, 0.0, -0.5], "nan_woe": -0.1}


def test_vectorised_woe_matches_the_single_customer_lookup():
    values = np.array([5.0, 10.0, 15.0, 20.0, 99.0, np.nan])
    assert final_test.woe_vector(values, SPEC).tolist() == [inference.woe_value(v, SPEC) for v in values]


def test_must_choose_exactly_one_mode(capsys):
    assert final_test.main([]) == 2
    assert final_test.main(["--rehearsal", "--confirm-final-run"]) == 2


def test_final_run_refuses_when_the_marker_exists_and_never_touches_the_data(tmp_path, monkeypatch):
    marker = tmp_path / "TEST_SET_USED.json"
    marker.write_text("{}")
    monkeypatch.setattr(final_test, "MARKER", marker)

    def boom(*args, **kwargs):
        raise AssertionError("the data must not be loaded")

    monkeypatch.setattr(final_test, "load_split", boom)
    monkeypatch.setattr(final_test, "load_artifacts", boom)
    assert final_test.main(["--confirm-final-run"]) == 1


needs_data = pytest.mark.skipif(
    not ((config.ARTIFACTS / "scorecard_model.json").exists() and (config.DATA_PROCESSED / "valid_predictions.parquet").exists()),
    reason="needs local artifacts and data",
)


@needs_data
def test_frozen_predictions_reproduce_the_saved_validation_predictions():
    from credit_pd_scorecard.split import load_split

    art = inference.load_artifacts(config.ARTIFACTS)
    valid = load_split("valid")
    preds = pd.read_parquet(config.DATA_PROCESSED / "valid_predictions.parquet")
    p_card, p_lgb = final_test.frozen_predictions(valid, art)
    assert p_card == pytest.approx(preds["pd_scorecard"].to_numpy(), abs=1e-9)
    assert p_lgb == pytest.approx(preds["pd_lgbm"].to_numpy(), abs=1e-9)


@needs_data
def test_report_renders_without_error_using_validation_as_a_stand_in():
    from credit_pd_scorecard.split import load_split

    art = inference.load_artifacts(config.ARTIFACTS)
    result = final_test.evaluate_frozen(load_split("valid"), art)
    text = final_test.render(result, result, art)
    assert "Final test-set report" in text and "TEST" in text
