"""Model loading and inference.

This module owns everything related to the trained Keras model: checking for
its presence, loading it, running predictions and validating the raw model
output. It performs **no** training and never modifies the model.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

import config
from preprocessing import ImageValidationError

logger = logging.getLogger(__name__)


class ModelUnavailableError(RuntimeError):
    """Raised when the trained model file is missing or cannot be loaded."""


class ModelOutputError(ValueError):
    """Raised when the model returns an output with an unexpected shape."""


def model_exists(model_path: Path | str = config.MODEL_PATH) -> bool:
    """Return True when a model file exists at the given path."""
    return Path(model_path).is_file()


def is_low_confidence(confidence: float) -> bool:
    """Return True when ``confidence`` is below the configured threshold.

    ``confidence`` must be a probability in [0.0, 1.0]. The comparison is strict
    (`<`), so exactly the threshold value (0.50) is NOT treated as low.
    """
    return confidence < config.CONFIDENCE_THRESHOLD


def load_model(model_path: Path | str = config.MODEL_PATH) -> Any:
    """Load the trained Keras model from ``model_path``.

    Raises
    ------
    ModelUnavailableError
        If the file is missing or Keras cannot load it.
    """
    path = Path(model_path)
    if not path.is_file():
        raise ModelUnavailableError(
            f"Trained model not found at '{path}'. Add the trained .keras file "
            "to continue."
        )

    # Imported lazily so that the model-missing path works even inside tests or
    # bare environments where TensorFlow is not installed.
    from tensorflow import keras

    try:
        logger.info("Loading model from %s ...", path)
        return keras.models.load_model(str(path))
    except Exception as exc:  # noqa: BLE001 - surface any load failure to caller
        logger.exception("Failed to load model from %s", path)
        raise ModelUnavailableError(
            f"The model at '{path}' exists but could not be loaded. "
            "It may be corrupt, an incompatible format, or built with a "
            "different TensorFlow/Keras version. Please check the logs."
        ) from exc


def validate_model_output(probabilities: np.ndarray) -> None:
    """Validate that ``probabilities`` is a length-10 vector of probabilities.

    Raises
    ------
    ModelOutputError
        If the shape is wrong, or values are not valid probabilities.
    """
    probs = np.asarray(probabilities, dtype=np.float32).flatten()

    if probs.shape[0] != config.MODEL_OUTPUT_SIZE:
        raise ModelOutputError(
            f"Model returned {probs.shape[0]} output values, expected "
            f"{config.MODEL_OUTPUT_SIZE} (one per CIFAR-10 class)."
        )

    if not np.all(np.isfinite(probs)):
        raise ModelOutputError("Model returned non-finite (NaN/Inf) values.")

    if probs.min() < 0.0:
        raise ModelOutputError("Model returned negative probability values.")

    return None


def predict(model: Any, image_tensor: np.ndarray) -> dict[str, Any]:
    """Run inference on a preprocessed image and return structured results.

    Parameters
    ----------
    model:
        A loaded Keras model.
    image_tensor:
        A preprocessed image of shape ``(1, 32, 32, 3)`` in [0.0, 1.0].

    Returns
    -------
    dict
        With keys ``predicted_index``, ``predicted_class``, ``confidence``,
        ``probabilities``, ``top3`` and ``inference_time``.

    Raises
    ------
    ModelOutputError
        If the model output shape is unexpected.
    RuntimeError
        If ``model.predict`` itself fails.
    """
    start = time.perf_counter()
    try:
        raw = model.predict(image_tensor, verbose=0)
    except Exception as exc:  # noqa: BLE001 - report runtime failures
        logger.exception("Prediction failed.")
        raise RuntimeError(
            "Prediction failed. Please try again or check the model and logs."
        ) from exc

    probabilities = np.asarray(raw, dtype=np.float32).flatten()
    logger.debug("Raw model output shape=%s", np.asarray(raw).shape)

    try:
        validate_model_output(probabilities)
    except ModelOutputError:
        raise  # re-raise the domain error unchanged

    confidence = float(probabilities.max())
    predicted_index = int(np.argmax(probabilities))

    # Order indices by descending probability.
    top_indices = np.argsort(probabilities)[::-1][:3]
    top3 = [
        {
            "index": int(i),
            "class": config.CLASS_NAMES[i],
            "score": float(probabilities[i]),
        }
        for i in top_indices
    ]

    inference_time = time.perf_counter() - start
    logger.info(
        "Prediction: %s (%.2f%%) in %.3fs",
        config.CLASS_NAMES[predicted_index],
        confidence * 100.0,
        inference_time,
    )

    return {
        "predicted_index": predicted_index,
        "predicted_class": config.CLASS_NAMES[predicted_index],
        "confidence": confidence,
        "probabilities": probabilities,
        "top3": top3,
        "inference_time": inference_time,
    }