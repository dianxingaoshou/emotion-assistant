import numpy as np

from scripts.chsims_teacher_temporal_neutral_review import infer


def test_temporal_review_only_changes_supported_neutral_with_available_window():
    incumbent = np.array([[.25, .43, .32], [.25, .43, .32], [.21, .2, .59], [.17, .56, .27]])
    supported = [[.1, .2, .7], [.35, .4, .25]]
    batch = {
        "retained": np.asarray([supported, [[np.nan] * 3, [.2, .21, .59]], supported, supported]),
        "context": incumbent[:, None], "acoustic": incumbent, "voice": incumbent, "prior": incumbent,
        "mini": incumbent, "design": np.zeros((4, 13)), "active": np.ones(4, dtype=bool),
    }
    policy = {"rule": {"channel": 0, "weight": .4, "target": 2, "max_gap": .13, "min_lead": .035}}
    revised = infer(batch, incumbent, policy)
    np.testing.assert_allclose(revised[0], .6 * incumbent[0] + .4 * batch["retained"][0, 0])
    np.testing.assert_allclose(revised[1:], incumbent[1:])
