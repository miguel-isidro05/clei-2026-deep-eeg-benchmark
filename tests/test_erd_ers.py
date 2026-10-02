import numpy as np

from deepbench.erd_ers import peterson_mi_rest_change, souza_lateralization


def _sine(amplitude: float, frequency: float = 10.0) -> np.ndarray:
    time = np.arange(256) / 128.0
    return (amplitude * np.sin(2 * np.pi * frequency * time)).astype(np.float32)


def test_peterson_synthetic_mu_desynchronization_is_negative() -> None:
    x = np.zeros((4, 3, 256), dtype=np.float32)
    x[:2] = _sine(2.0)
    x[2:] = _sine(1.0)
    result = peterson_mi_rest_change(x, np.array([0, 0, 1, 1]), ("C3", "Cz", "C4"), 128.0)
    assert all(result["mu"][channel] < 0 for channel in ("C3", "Cz", "C4"))


def test_souza_synthetic_lateralization_has_expected_direction() -> None:
    x = np.zeros((4, 2, 256), dtype=np.float32)
    x[:2, 0] = _sine(1.0)
    x[:2, 1] = _sine(2.0)
    x[2:, 0] = _sine(2.0)
    x[2:, 1] = _sine(1.0)
    result = souza_lateralization(x, np.array([0, 0, 1, 1]), ("C3", "C4"), 128.0)
    assert result["mu"]["left_minus_right"] > 0
