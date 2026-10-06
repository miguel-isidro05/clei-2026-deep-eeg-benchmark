"""Braindecode model registry and fixed training recipes."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass

os.environ.setdefault("MNE_DONTWRITE_HOME", "true")

import numpy as np
import torch
from braindecode import EEGClassifier
from braindecode.models import EEGConformer, EEGInceptionMI, EEGNet, FBCNet, ShallowFBCSPNet
from skorch.callbacks import LRScheduler

from .classical import CSP_RECIPE
from .config import MODEL_NAMES
from .reproducibility import set_seeds


@dataclass(frozen=True)
class TrainingRecipe:
    lr: float
    optimizer: str
    provenance_id: str
    weight_decay: float = 0.0
    batch_size: int = 64
    epochs: int = 300
    betas: tuple[float, float] = (0.9, 0.999)
    selection_policy: str = "predeclared_no_test_tuning"


TRAINING_RECIPES: dict[str, TrainingRecipe] = {
    "EEGNet": TrainingRecipe(
        lr=6.25e-4, optimizer="adamw", provenance_id="miccai_fixed_convergence_v1"
    ),
    "FBCNet": TrainingRecipe(
        lr=6.25e-4, optimizer="adamw", provenance_id="shared_fixed_convergence_v1"
    ),
    "ShallowConvNet": TrainingRecipe(
        lr=6.25e-4, optimizer="adamw", provenance_id="shared_fixed_convergence_v1"
    ),
    "EEGConformer": TrainingRecipe(
        lr=5e-5, optimizer="adam", provenance_id="project_predeclared_conformer_v1"
    ),
    "EEGInceptionMI": TrainingRecipe(
        lr=1e-3, optimizer="adam", provenance_id="project_predeclared_inception_v1"
    ),
}


def make_module(
    name: str,
    *,
    n_chans: int,
    n_outputs: int,
    n_times: int,
    sfreq: float,
) -> torch.nn.Module:
    """Instantiate one official Braindecode architecture."""
    if name == "EEGNet":
        return EEGNet(n_chans=n_chans, n_outputs=n_outputs, n_times=n_times, sfreq=sfreq)
    if name == "FBCNet":
        return FBCNet(
            n_chans=n_chans,
            n_outputs=n_outputs,
            n_times=n_times,
            sfreq=sfreq,
            n_bands=[(8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 30)],
        )
    if name == "ShallowConvNet":
        return ShallowFBCSPNet(
            n_chans=n_chans,
            n_outputs=n_outputs,
            n_times=n_times,
            final_conv_length="auto",
        )
    if name == "EEGConformer":
        return EEGConformer(
            n_chans=n_chans,
            n_outputs=n_outputs,
            n_times=n_times,
            drop_prob=0.7,
        )
    if name == "EEGInceptionMI":
        return EEGInceptionMI(
            n_chans=n_chans,
            n_outputs=n_outputs,
            n_times=n_times,
            sfreq=sfreq,
        )
    raise ValueError(f"Unknown model {name}; expected one of {MODEL_NAMES}")


def make_classifier(
    name: str,
    *,
    n_chans: int,
    n_times: int,
    sfreq: float,
    device: str,
    seed: int,
    epochs: int | None = None,
) -> EEGClassifier:
    """Build a fixed-epoch classifier with no test-driven stopping or tuning."""
    set_seeds(seed)
    base = TRAINING_RECIPES[name]
    effective_epochs = int(epochs if epochs is not None else base.epochs)
    module = make_module(
        name,
        n_chans=n_chans,
        n_outputs=2,
        n_times=n_times,
        sfreq=sfreq,
    )
    optimizer = torch.optim.AdamW if base.optimizer == "adamw" else torch.optim.Adam
    return EEGClassifier(
        module=module,
        criterion=torch.nn.CrossEntropyLoss,
        optimizer=optimizer,
        optimizer__lr=base.lr,
        optimizer__weight_decay=base.weight_decay,
        optimizer__betas=base.betas,
        batch_size=base.batch_size,
        max_epochs=effective_epochs,
        train_split=None,
        iterator_train__shuffle=True,
        iterator_train__drop_last=False,
        callbacks=[
            (
                "lr_scheduler",
                LRScheduler("CosineAnnealingLR", T_max=max(1, effective_epochs - 1)),
            )
        ],
        device=device,
        classes=[0, 1],
        verbose=0,
    )


def recipe_dict(name: str, epochs: int | None = None) -> dict[str, object]:
    if name == "CSP+LDA":
        return dict(CSP_RECIPE)
    values = asdict(TRAINING_RECIPES[name])
    if epochs is not None:
        values["epochs"] = int(epochs)
    return values


def predict_scores(classifier: EEGClassifier, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    contiguous = np.ascontiguousarray(x, dtype=np.float32)
    predicted = np.asarray(classifier.predict(contiguous), dtype=np.int64)
    probabilities = np.asarray(classifier.predict_proba(contiguous), dtype=float)[:, 1]
    return predicted, probabilities


def parameter_count(name: str, n_chans: int, n_times: int, sfreq: float) -> int:
    module = make_module(name, n_chans=n_chans, n_outputs=2, n_times=n_times, sfreq=sfreq)
    return int(
        sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
    )
