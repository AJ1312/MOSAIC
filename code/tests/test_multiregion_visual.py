import numpy as np

from mosaic.l3_visual import (
    MULTIREGION_VISUAL_FEATURE_NAMES,
    VISUAL_FEATURE_NAMES,
    extract_multiregion_visual_features,
    extract_visual_features,
)


def test_multiregion_preserves_baseline_and_has_stable_schema():
    frames = np.random.default_rng(7).integers(0, 256, size=(12, 64, 64, 3), dtype=np.uint8)
    baseline = extract_visual_features(frames)
    localized = extract_multiregion_visual_features(frames)

    assert baseline.names == VISUAL_FEATURE_NAMES
    assert localized.names == MULTIREGION_VISUAL_FEATURE_NAMES
    assert localized.vector.shape == (5 * len(VISUAL_FEATURE_NAMES),)
    np.testing.assert_allclose(localized.vector[: len(VISUAL_FEATURE_NAMES)], baseline.vector)
    assert np.isfinite(localized.vector).all()


def test_multiregion_rejects_invalid_frames():
    with np.testing.assert_raises(ValueError):
        extract_multiregion_visual_features(np.zeros((4, 1, 8), dtype=np.uint8))
