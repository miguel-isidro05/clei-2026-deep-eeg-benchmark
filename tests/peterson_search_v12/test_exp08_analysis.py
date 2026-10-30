from __future__ import annotations

from experiments.peterson_search_v12.exp08_fixed_convergence.analysis import (
    collapse_counts,
)


def test_collapse_counts_do_not_double_count_cell_and_fold_flags() -> None:
    cells = [
        {
            "quality_flags": ["validation_class_collapse"],
            "fold_reports": [
                {"quality_flags": ["validation_class_collapse"]},
                {"quality_flags": []},
            ],
        },
        {
            "quality_flags": ["validation_class_collapse"],
            "fold_reports": [
                {"quality_flags": ["validation_class_collapse"]},
                {"quality_flags": ["validation_class_collapse"]},
            ],
        },
        {"quality_flags": [], "fold_reports": [{"quality_flags": []}]},
    ]
    assert collapse_counts(cells) == {
        "collapsed_fold_count": 3,
        "affected_cell_count": 2,
    }

