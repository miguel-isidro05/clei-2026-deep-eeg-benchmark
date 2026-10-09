from __future__ import annotations

import pytest
import torch

from experiments.peterson_diffusion_v11.diffusion import DiffusionSchedule
from experiments.peterson_diffusion_v11.energy import class_energies
from experiments.peterson_diffusion_v11.models import make_model, parameter_report
from experiments.peterson_diffusion_v11.objectives import diffusion_objective


@pytest.mark.parametrize("architecture", ["res", "filterbank", "smooth_basis", "conformer"])
@pytest.mark.parametrize("n_times", [256, 512])
def test_denoisers_preserve_shape_and_have_finite_gradients(
    architecture: str, n_times: int
) -> None:
    torch.manual_seed(4)
    model = make_model(architecture, "diffusion", n_chans=15, hidden=16)
    x = torch.randn(2, 15, n_times, requires_grad=True)
    timestep = torch.tensor([3, 7])
    label = torch.tensor([0, 1])
    mask = torch.ones(2, 15)
    output = model(x, timestep=timestep, label=label, channel_mask=mask)
    assert output.shape == x.shape
    output.square().mean().backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


@pytest.mark.parametrize("architecture", ["res", "filterbank", "smooth_basis", "conformer"])
def test_diffusion_and_discriminative_variants_share_backbone_capacity(
    architecture: str,
) -> None:
    report = parameter_report(architecture, n_chans=15, hidden=16)
    assert report["diffusion_backbone"] == report["discriminative_backbone"]
    assert report["backbone_relative_difference"] == 0.0


def test_class_energies_use_common_random_numbers() -> None:
    class RecordingModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.inputs: list[torch.Tensor] = []

        def forward(self, x, *, timestep, label, channel_mask):
            self.inputs.append(x.detach().clone())
            return torch.zeros_like(x)

    model = RecordingModel()
    schedule = DiffusionSchedule(steps=10)
    generator = torch.Generator().manual_seed(9)
    energies = class_energies(
        model,
        torch.randn(3, 4, 12),
        torch.ones(3, 4),
        schedule,
        k=1,
        generator=generator,
    )
    assert energies.shape == (3, 2)
    assert torch.equal(model.inputs[0], model.inputs[1])


def test_masked_elements_do_not_contribute_to_energy() -> None:
    class ZeroModel(torch.nn.Module):
        def forward(self, x, *, timestep, label, channel_mask):
            return torch.zeros_like(x)

    x = torch.zeros(1, 2, 8)
    mask = torch.tensor([[1.0, 0.0]])
    schedule = DiffusionSchedule(steps=5)
    first = class_energies(
        ZeroModel(), x, mask, schedule, k=1, generator=torch.Generator().manual_seed(1)
    )
    changed = x.clone()
    changed[:, 1] = 1000.0
    second = class_energies(
        ZeroModel(), changed, mask, schedule, k=1, generator=torch.Generator().manual_seed(1)
    )
    assert torch.allclose(first, second)


def test_objective_terms_are_explicit_ablations() -> None:
    model = make_model("res", "diffusion", n_chans=3, hidden=8)
    x = torch.randn(4, 3, 32)
    y = torch.tensor([0, 1, 0, 1])
    mask = torch.ones(4, 3)
    schedule = DiffusionSchedule(steps=8)
    losses = diffusion_objective(
        model,
        x,
        y,
        mask,
        schedule,
        objective="noise_only",
        generator=torch.Generator().manual_seed(2),
    )
    assert losses.rank.item() == 0.0
    assert losses.consistency.item() == 0.0
    assert torch.isfinite(losses.total)
