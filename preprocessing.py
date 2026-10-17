"""Image loading, validation and preprocessing.

Kept independent from the Streamlit interface so the pipeline can be tested
(and reused) on its own. The pure input/output contract used here is:

    * accepted input ..... a Pillow-openable file object plus a filename
    * output ............. a float32 array of shape ``(1, 32, 32, 3)`` whose
                           pixel values lie in the range [0.0, 1.0]
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import IO

import numpy as np
from PIL import Image, UnidentifiedImageError

import config

logger = logging.getLogger(__name__)


class ImageValidationError(ValueError):
    """Raised when an uploaded file is not a usable image for this app."""


def _extension_of(filename: str) -> str:
    """Return the lowercase file extension (including the dot) of a filename."""
    return Path(filename).suffix.lower()


def supported_extension(filename: str) -> bool:
    """Return True when ``filename`` carries a supported image extension."""
    return _extension_of(filename) in config.SUPPORTED_EXTENSIONS


def validate_upload(file: IO[bytes], filename: str) -> None:
    """Validate an uploaded file before any decoding attempt.

    Raises :class:`ImageValidationError` for unsupported extensions or files
    that Pillow cannot parse as an image host file. Detailed causes are logged.

    Parameters
    ----------
    file:
        The uploaded file-like object (must support ``seek``).
    filename:
        Original name of the uploaded file.
    """
    if not supported_extension(filename):
        logger.info(
            "Rejected upload with unsupported extension %r (file=%r)",
            _extension_of(filename),
            filename,
        )
        raise ImageValidationError(
            "Unsupported file type. Please upload a JPG, JPEG or PNG image."
        )

    # Rewind in case the stream was already partially consumed.
    file.seek(0)
    try:
        with Image.open(file) as img:
            img.load()  # force a full decode; catches corrupt/truncated images
    except UnidentifiedImageError as exc:
        logger.warning("Rejected upload: not a decodable image (file=%r).", filename)
        raise ImageValidationError(
            "The uploaded file is not a valid image. Please upload a JPG, JPEG or PNG."
        ) from exc
    except (OSError, SyntaxError) as exc:
        logger.warning("Rejected upload: image decode failed (file=%r): %s", filename, exc)
        raise ImageValidationError(
            "The image appears to be corrupted and could not be read."
        ) from exc


def preprocess_image(file: IO[bytes]) -> np.ndarray:
    """Open, validate, resize and normalise an image for model inference.

    Returns a ``float32`` array of shape ``(1, 32, 32, 3)`` matching the
    CIFAR-10 training pipeline: RGB, resized to 32x32, values in [0.0, 1.0].

    Parameters
    ----------
    file:
        An open, Pillow-readable image file.

    Raises
    ------
    ImageValidationError
        If the file cannot be decoded as an image.
    """
    file.seek(0)
    try:
        with Image.open(file) as img:
            img = img.convert(config.IMAGE_MODE)
            img = img.resize((config.IMAGE_WIDTH, config.IMAGE_HEIGHT))
            array = np.asarray(img, dtype=np.float32)
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        logger.warning("preprocess_image: could not decode image: %s", exc)
        raise ImageValidationError(
            "The image could not be processed. Please upload a valid JPG, JPEG or PNG."
        ) from exc

    # Normalize to [0.0, 1.0] and add the batch dimension.
    array = array / config.RGB_MAX
    array = np.expand_dims(array, axis=0)

    if array.shape != (1, config.IMAGE_WIDTH, config.IMAGE_HEIGHT, config.IMAGE_CHANNELS):
        # Defensive check; in practice the operations above guarantee the shape.
        logger.error("Unexpected preprocessed shape %s", array.shape)
        raise ImageValidationError("Image preprocessing produced an unexpected shape.")

    logger.debug("Preprocessed image -> shape=%s dtype=%s", array.shape, array.dtype)
    return array