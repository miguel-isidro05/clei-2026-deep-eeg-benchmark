from __future__ import annotations

from functools import partial

import numpy as np

from experiments.peterson_search_v12.exp03_quality_fbcnet.models import fit_quality_profile
from experiments.peterson_search_v12.exp04_qcr_ssnet.catalog import QCRCandidate
from experiments.peterson_search_v12.exp04_qcr_ssnet.models import QCRSSNet
from experiments.peterson_search_v12.exp04_qcr_ssnet.training import train_qcr


def test_qcr_training_retains_clean_primary_best_and_last_checkpoints():
    rng = np.random.default_rng(44)
    x = rng.normal(size=(16, 5, 64)).astype(np.float32)
    y = np.asarray([0, 1] * 8, dtype=np.int64)
    profile = fit_quality_profile(x[:12])
    candidate = QCRCandidate(
        config_id="test",
        role="qcr_full",
        architecture="qcr",
        clean_n_filters_spat=2,
        robust_n_filters_spat=2,
        clean_stride_factor=2,
        robust_stride_factor=2,
        channel_dropout=0.5,
        router_mode="learned",
        use_quality_gate=True,
        lambda_corrupt=1.0,
        lambda_consistency=0.1,
        lambda_router=0.2,
        lambda_specialization=0.5,
        sha256="a" * 64,
    )
    factory = partial(
        QCRSSNet,
        candidate,
        profile=profile,
        n_chans=5,
        n_times=64,
        sfreq=128.0,
        gate_hidden=4,
        router_hidden=4,
    )
    outcome = train_qcr(
        factory,
        candidate,
        x[:12],
        y[:12],
        x[12:],
        y[12:],
        device="cpu",
        seed=3,
        epochs=2,
        batch_size=4,
        learning_rate=1e-3,
        weight_decay=1e-4,
        corruption_probability=0.5,
        max_missing_channels=2,
    )
    assert outcome.updates == 6
    assert len(outcome.history) == 2
    assert 1 <= outcome.best.epoch <= outcome.last.epoch == 2
    assert outcome.best.trajectory_sha256 == outcome.last.trajectory_sha256
    assert set(outcome.best.state_dict) == set(outcome.last.state_dict)
