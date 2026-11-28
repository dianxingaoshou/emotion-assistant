import numpy as np

from scripts.chsims_teacher_precision_review import accept_any


def test_direction_gate_checks_margin_and_support_for_each_proposal():
    positive = [.33, .31, .36]
    neutral = [.34, .48, .18]
    strong_positive = [.23, .17, .60]
    initial = np.asarray([positive, neutral, neutral, positive, strong_positive])
    negative = [.60, .24, .16]
    support = np.asarray([[negative, positive], [negative, positive], [positive, positive], [negative, positive],
                          [negative, positive]])
    record = {"initial": initial, "support": support, "active": np.asarray([True, True, True, False, True])}
    proposals = np.asarray([negative] * len(initial))
    from_positive = {"direction": (2, 0), "max_initial_gap": .04, "min_proposed_gap": .12, "corroborate": True}
    from_neutral = {"direction": (1, 0), "max_initial_gap": .20, "min_proposed_gap": .12, "corroborate": True}
    np.testing.assert_array_equal(accept_any(record, proposals, [from_positive, from_neutral]),
                                  [True, True, False, False, False])
