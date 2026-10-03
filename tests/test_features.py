import math

import pandas as pd

from turbofan_rul.features import add_window_features


def make_two_engines() -> pd.DataFrame:
    # Engine 1: x = 2 * cycle + 1 (slope 2). Engine 2 starts at a very different level.
    return pd.DataFrame(
        {
            "unit_id": [1, 1, 1, 1, 2, 2],
            "cycle": [1, 2, 3, 4, 1, 2],
            "x": [3.0, 5.0, 7.0, 9.0, 100.0, 90.0],
        }
    )


def test_window_never_mixes_engines():
    result, _ = add_window_features(make_two_engines(), ["x"], window=3)

    first_row_engine_2 = result[(result["unit_id"] == 2) & (result["cycle"] == 1)]
    assert first_row_engine_2["x_mean3"].item() == 100.0


def test_window_only_looks_back_and_shrinks_at_the_start():
    result, _ = add_window_features(make_two_engines(), ["x"], window=3)
    engine_1 = result[result["unit_id"] == 1]

    # cycle 1 -> [3], cycle 2 -> [3, 5], cycle 3 -> [3, 5, 7], cycle 4 -> [5, 7, 9]
    assert engine_1["x_mean3"].tolist() == [3.0, 4.0, 5.0, 7.0]


def test_slope_recovers_linear_trend_and_is_zero_for_a_single_row():
    result, _ = add_window_features(make_two_engines(), ["x"], window=3)
    engine_1 = result[result["unit_id"] == 1]

    assert engine_1["x_slope3"].iloc[0] == 0.0
    assert all(math.isclose(s, 2.0) for s in engine_1["x_slope3"].iloc[1:])
    assert engine_1["x_std3"].iloc[0] == 0.0


def test_feature_list_contains_raw_and_window_columns():
    _, columns = add_window_features(make_two_engines(), ["x"], window=3)

    assert columns == ["x", "x_mean3", "x_std3", "x_slope3"]
