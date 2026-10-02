import pandas as pd

from turbofan_rul.targets import add_linear_rul


def test_linear_rul_counts_down_to_zero_per_engine():
    frame = pd.DataFrame({"unit_id": [1, 1, 1, 2, 2], "cycle": [1, 2, 3, 1, 2]})

    result = add_linear_rul(frame)

    assert result["rul"].tolist() == [2, 1, 0, 1, 0]


def test_linear_rul_does_not_depend_on_row_order():
    frame = pd.DataFrame({"unit_id": [2, 1, 2, 1], "cycle": [2, 1, 1, 2]})

    result = add_linear_rul(frame)

    assert result["rul"].tolist() == [0, 1, 1, 0]


def test_linear_rul_leaves_input_unchanged():
    frame = pd.DataFrame({"unit_id": [1, 1], "cycle": [1, 2]})

    add_linear_rul(frame)

    assert "rul" not in frame.columns
