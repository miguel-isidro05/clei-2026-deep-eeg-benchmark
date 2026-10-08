from __future__ import annotations

import numpy as np
import pytest
import torch

from deepbench.config import MODEL_NAMES
from deepbench.models import make_classifier, make_module, recipe_dict


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_every_model_accepts_common_full_trial(model_name: str) -> None:
    module = make_module(model_name, n_chans=15, n_outputs=2, n_times=512, sfreq=128.0).eval()
    with torch.no_grad():
        output = module(torch.zeros(2, 15, 512))
    assert output.shape == (2, 2)


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_every_model_accepts_common_augmentation_crop(model_name: str) -> None:
    module = make_module(model_name, n_chans=15, n_outputs=2, n_times=256, sfreq=128.0).eval()
    with torch.no_grad():
        output = module(torch.zeros(2, 15, 256))
    assert output.shape == (2, 2)


def test_training_uses_partial_batch_when_dataset_is_smaller_than_batch_size() -> None:
    classifier = make_classifier(
        "EEGNet",
        n_chans=3,
        n_times=256,
        sfreq=128.0,
        device="cpu",
        seed=0,
        epochs=1,
    )
    x = np.zeros((20, 3, 256), dtype=np.float32)
    y = np.tile([0, 1], 10)
    classifier.fit(x, y)
    assert classifier.history[-1, "train_batch_count"] == 1


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_every_recipe_records_predeclared_provenance(model_name: str) -> None:
    recipe = recipe_dict(model_name)
    assert recipe["provenance_id"]
    assert recipe["selection_policy"] == "predeclared_no_test_tuning"
