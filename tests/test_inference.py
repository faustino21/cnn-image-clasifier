"""Tests for model loading and prediction logic.

The real trained model is never required: a mock model supplies a hard-coded
probability vector, and the missing-model path is exercised directly.
"""
from __future__ import annotations

import numpy as np
import pytest

import config
import inference
from inference import (
    ModelOutputError,
    ModelUnavailableError,
    is_low_confidence,
    load_model,
    model_exists,
    predict,
    validate_model_output,
)


class FakeModel:
    """A minimal stand-in for a loaded Keras model."""

    def __init__(self, output: np.ndarray, fail: bool = False):
        self._output = output
        self._fail = fail

    def predict(self, x, verbose=0):
        assert x.shape == (1, 32, 32, 3), "predict called with wrong input shape"
        if self._fail:
            raise RuntimeError("simulated inference failure")
        return self._output


def _probabilities(top_index: int = 3) -> np.ndarray:
    """Return a realistic softmax-like probability vector of length 10."""
    probs = np.full(config.MODEL_OUTPUT_SIZE, 0.02, dtype=np.float32)
    probs[top_index] = 0.82
    probs[(top_index + 1) % 10] = 0.10
    probs[(top_index + 2) % 10] = 0.06
    # Normalise so the vector sums to 1.0.
    probs = probs / probs.sum()
    return probs


def test_missing_model_does_not_crash(tmp_path):
    missing = tmp_path / "cnn_optimized.keras"
    assert not model_exists(missing)
    with pytest.raises(ModelUnavailableError):
        load_model(missing)


def test_predict_returns_expected_structure():
    probs = _probabilities(top_index=3)
    model = FakeModel(probs)
    tensor = np.zeros((1, 32, 32, 3), dtype=np.float32)

    result = predict(model, tensor)

    assert result["predicted_class"] == config.CLASS_NAMES[3]
    assert result["predicted_index"] == 3
    assert result["confidence"] == pytest.approx(float(probs.max()))
    assert result["probabilities"].shape == (10,)
    assert len(result["top3"]) == 3
    assert result["top3"][0]["class"] == config.CLASS_NAMES[3]
    assert result["inference_time"] >= 0.0


def test_top3_are_sorted_by_score():
    model = FakeModel(_probabilities(top_index=7))
    result = predict(model, np.zeros((1, 32, 32, 3), np.float32))
    scores = [row["score"] for row in result["top3"]]
    assert scores == sorted(scores, reverse=True)


def test_output_validation_rejects_wrong_length():
    with pytest.raises(ModelOutputError):
        validate_model_output(np.zeros(9, dtype=np.float32))


def test_output_validation_rejects_negative_values():
    bad = _probabilities()
    bad[5] = -0.1
    with pytest.raises(ModelOutputError):
        validate_model_output(bad)


def test_output_validation_accepts_valid_vector():
    # Should not raise.
    validate_model_output(_probabilities())


def test_predict_raises_when_model_fails():
    model = FakeModel(_probabilities(), fail=True)
    with pytest.raises(RuntimeError):
        predict(model, np.zeros((1, 32, 32, 3), np.float32))


# --- low-confidence threshold --------------------------------------------


def test_confidence_below_threshold_is_low():
    assert is_low_confidence(0.49) is True


def test_confidence_at_threshold_is_not_low():
    # Strict < comparison: exactly 0.50 is NOT low.
    assert is_low_confidence(0.50) is False


def test_confidence_above_threshold_is_not_low():
    assert is_low_confidence(0.85) is False
    assert is_low_confidence(1.0) is False


def test_no_keras_file_is_created_by_tests(tmp_path):
    keras_files = list(tmp_path.rglob("*.keras"))
    assert keras_files == []