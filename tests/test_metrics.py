import math

from turbofan_rul.metrics import mae, nasa_score, rmse


def test_rmse_and_mae_on_known_errors():
    y_true = [10, 20, 30]
    y_pred = [13, 16, 30]  # errors +3, -4, 0

    assert mae(y_true, y_pred) == 7 / 3
    assert math.isclose(rmse(y_true, y_pred), math.sqrt(25 / 3))


def test_nasa_score_is_zero_for_perfect_predictions():
    assert nasa_score([50, 80], [50, 80]) == 0


def test_nasa_score_penalises_late_more_than_early():
    late = nasa_score([50], [70])   # d = +20
    early = nasa_score([50], [30])  # d = -20

    assert math.isclose(late, math.exp(20 / 10) - 1)
    assert math.isclose(early, math.exp(20 / 13) - 1)
    assert late > early
