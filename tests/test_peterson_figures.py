from __future__ import annotations

import pandas as pd

from deepbench.peterson_figures import MODEL_COLORS, _forest_labels, _set_publication_style


def test_forest_labels_include_protocol_for_augmentation_rows() -> None:
    table = pd.DataFrame(
        [
            {
                "protocol": "within_session",
                "model": "EEGNet",
                "comparison": "overlap-center_x6",
            }
        ]
    )

    assert _forest_labels(table) == [
        "EEGNet: overlap − center ×6 (Within-session)"
    ]


def test_publication_style_defines_a_stable_color_for_every_model() -> None:
    _set_publication_style()

    assert set(MODEL_COLORS) == {
        "CSP+LDA",
        "EEGNet",
        "FBCNet",
        "ShallowConvNet",
        "EEGConformer",
    }
    assert len(set(MODEL_COLORS.values())) == len(MODEL_COLORS)
