from __future__ import annotations

import numpy as np
import pytest
import torch

from deepbench.config import MODEL_NAMES
from deepbench.models import make_module
from deepbench.transfer import (
    make_target_module,
    optimizer_parameter_groups,
    trainable_parameter_names,
)


@pytest.mark.parametrize("model_name", MODEL_NAMES)
@pytest.mark.parametrize("arm", ("linear_probe", "full_finetune"))
def test_transfer_loads_backbone_reinitializes_head_and_applies_freezing(
    model_name: str,
    arm: str,
) -> None:
    torch.manual_seed(7)
    source = make_module(
        model_name,
        n_chans=15,
        n_outputs=2,
        n_times=256,
        sfreq=128.0,
    )
    with torch.no_grad():
        for parameter in source.parameters():
            parameter.fill_(0.125)
    source_state = source.state_dict()

    torch.manual_seed(19)
    fresh_target = make_module(
        model_name,
        n_chans=15,
        n_outputs=2,
        n_times=256,
        sfreq=128.0,
    )
    natively_trainable = [
        name for name, parameter in fresh_target.named_parameters() if parameter.requires_grad
    ]

    target = make_target_module(
        model_name,
        arm=arm,
        seed=19,
        source_state=source_state,
        n_chans=15,
        n_times=256,
        sfreq=128.0,
    )

    target_state = target.state_dict()
    backbone_keys = [key for key in source_state if not key.startswith("final_layer.")]
    head_keys = [key for key in source_state if key.startswith("final_layer.")]
    assert backbone_keys
    assert head_keys
    for key in backbone_keys:
        assert torch.equal(target_state[key], source_state[key])
    assert any(not torch.equal(target_state[key], source_state[key]) for key in head_keys)
    trainable = trainable_parameter_names(target)
    if arm == "linear_probe":
        assert trainable
        assert all(name.startswith("final_layer.") for name in trainable)
    else:
        assert trainable == natively_trainable


def test_scratch_target_rejects_source_state() -> None:
    module = make_module("EEGNet", n_chans=15, n_outputs=2, n_times=256, sfreq=128.0)
    with pytest.raises(ValueError, match="scratch"):
        make_target_module(
            "EEGNet",
            arm="scratch",
            seed=0,
            source_state=module.state_dict(),
            n_chans=15,
            n_times=256,
            sfreq=128.0,
        )


def test_full_finetune_optimizer_uses_lower_backbone_learning_rate() -> None:
    module = make_target_module(
        "EEGNet",
        arm="full_finetune",
        seed=0,
        source_state=make_module(
            "EEGNet", n_chans=15, n_outputs=2, n_times=256, sfreq=128.0
        ).state_dict(),
        n_chans=15,
        n_times=256,
        sfreq=128.0,
    )

    groups = optimizer_parameter_groups(module, arm="full_finetune", base_lr=1e-3)

    assert len(groups) == 2
    assert sorted(group["lr"] for group in groups) == pytest.approx([1e-4, 1e-3])
    grouped = [parameter for group in groups for parameter in group["params"]]
    assert len(grouped) == len(list(module.parameters()))
    assert len({id(parameter) for parameter in grouped}) == len(grouped)


def test_probability_threshold_remains_binary() -> None:
    probabilities = np.array([0.49, 0.5, 0.9])
    assert (probabilities >= 0.5).astype(int).tolist() == [0, 1, 1]
