import numpy as np
import pandas as pd

from turbofan_rul.sequences import make_sequences


def make_two_engines() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "unit_id": [1, 1, 1, 1, 2, 2],
            "cycle": [1, 2, 3, 4, 1, 2],
            "x": [1.0, 2.0, 3.0, 4.0, 10.0, 20.0],
        }
    )


def test_one_window_per_cycle_with_expected_shape():
    x, ends = make_sequences(make_two_engines(), ["x"], seq_len=3)

    assert x.shape == (6, 3, 1)
    assert x.dtype == np.float32
    assert ends[["unit_id", "cycle"]].values.tolist() == [[1, 1], [1, 2], [1, 3], [1, 4], [2, 1], [2, 2]]


def test_windows_end_at_their_cycle_and_pad_with_first_value():
    x, _ = make_sequences(make_two_engines(), ["x"], seq_len=3)

    assert x[0, :, 0].tolist() == [1.0, 1.0, 1.0]   # engine 1, cycle 1: padded
    assert x[1, :, 0].tolist() == [1.0, 1.0, 2.0]   # engine 1, cycle 2: partly padded
    assert x[3, :, 0].tolist() == [2.0, 3.0, 4.0]   # engine 1, cycle 4: last 3 cycles


def test_windows_never_mix_engines():
    x, _ = make_sequences(make_two_engines(), ["x"], seq_len=3)

    assert x[4, :, 0].tolist() == [10.0, 10.0, 10.0]  # engine 2 does not see engine 1


def test_last_only_keeps_each_engines_final_window():
    x, ends = make_sequences(make_two_engines(), ["x"], seq_len=3, last_only=True)

    assert x.shape == (2, 3, 1)
    assert ends["cycle"].tolist() == [4, 2]
    assert x[1, :, 0].tolist() == [10.0, 10.0, 20.0]
