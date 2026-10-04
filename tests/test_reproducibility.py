from __future__ import annotations

import torch

from deepbench.reproducibility import configure_determinism


def test_determinism_is_strict_and_disables_nondeterministic_cuda_attention() -> None:
    configure_determinism()

    assert torch.are_deterministic_algorithms_enabled()
    assert not torch.is_deterministic_algorithms_warn_only_enabled()
    if torch.cuda.is_available():
        assert not torch.backends.cuda.flash_sdp_enabled()
        assert not torch.backends.cuda.mem_efficient_sdp_enabled()
        assert torch.backends.cuda.math_sdp_enabled()
