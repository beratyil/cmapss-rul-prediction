import pandas as pd

from turbofan_rul.preprocessing import last_cycle_rows, split_engines


def make_fleet(n_engines: int = 10, cycles: int = 3) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "unit_id": [unit for unit in range(1, n_engines + 1) for _ in range(cycles)],
            "cycle": list(range(1, cycles + 1)) * n_engines,
        }
    )


def test_split_engines_keeps_each_engine_on_one_side():
    fleet = make_fleet()

    train, val = split_engines(fleet, val_fraction=0.2, seed=0)

    train_ids, val_ids = set(train["unit_id"]), set(val["unit_id"])
    assert train_ids.isdisjoint(val_ids)
    assert train_ids | val_ids == set(fleet["unit_id"])
    assert len(val_ids) == 2
    assert len(train) + len(val) == len(fleet)


def test_split_engines_is_reproducible_with_seed():
    fleet = make_fleet()

    _, first = split_engines(fleet, seed=7)
    _, second = split_engines(fleet, seed=7)

    assert set(first["unit_id"]) == set(second["unit_id"])


def test_last_cycle_rows_returns_one_final_row_per_engine():
    fleet = make_fleet(n_engines=3, cycles=4)

    result = last_cycle_rows(fleet)

    assert result["unit_id"].tolist() == [1, 2, 3]
    assert result["cycle"].tolist() == [4, 4, 4]
