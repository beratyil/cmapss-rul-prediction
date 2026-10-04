import torch

from turbofan_rul.models import GRURegressor, LSTMRegressor


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def test_models_map_a_batch_of_windows_to_one_value_each():
    x = torch.randn(8, 50, 14)  # (batch, seq_len, n_features)

    for model in (LSTMRegressor(14, hidden_size=64), GRURegressor(14, hidden_size=64)):
        assert model(x).shape == (8,)


def test_parameter_counts_match_the_gate_formulas():
    # Per gate block: input weights H*F, recurrent weights H*H, two bias vectors 2H.
    # LSTM has 4 gate blocks, GRU has 3; the linear head adds H + 1.
    f, h = 14, 64
    gate_block = h * f + h * h + 2 * h

    assert count_parameters(LSTMRegressor(f, hidden_size=h)) == 4 * gate_block + h + 1 == 20545
    assert count_parameters(GRURegressor(f, hidden_size=h)) == 3 * gate_block + h + 1 == 15425
