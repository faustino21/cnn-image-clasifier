"""Tests for the image preprocessing pipeline."""
from __future__ import annotations

import numpy as np
import pytest

import config
from preprocessing import (
    ImageValidationError,
    preprocess_image,
    supported_extension,
    validate_upload,
)


def test_preprocess_produces_rgb_batched_normalized_tensor(png_upload):
    tensor = preprocess_image(png_upload)

    assert isinstance(tensor, np.ndarray)
    assert tensor.dtype == np.float32
    assert tensor.shape == (1, config.IMAGE_WIDTH, config.IMAGE_HEIGHT,
                            config.IMAGE_CHANNELS)
    # All pixel values must lie in [0.0, 1.0].
    assert tensor.min() >= 0.0
    assert tensor.max() <= 1.0


def test_preprocessing_matches_config_dimensions(png_upload):
    tensor = preprocess_image(png_upload)
    assert tensor.shape[1] == config.IMAGE_HEIGHT
    assert tensor.shape[2] == config.IMAGE_WIDTH
    assert tensor.shape[3] == config.IMAGE_CHANNELS


def test_supported_extensions():
    assert supported_extension("a.jpg")
    assert supported_extension("a.jpeg")
    assert supported_extension("a.png")
    assert supported_extension("A.PNG")  # case-insensitive
    assert not supported_extension("a.gif")
    assert not supported_extension("a")


def test_invalid_image_raises(corrupt_upload):
    with pytest.raises(ImageValidationError):
        preprocess_image(corrupt_upload)


def test_validate_upload_rejects_unsupported_extension(unsupported_upload):
    with pytest.raises(ImageValidationError):
        validate_upload(unsupported_upload, unsupported_upload.name)


def test_validate_upload_rejects_corrupt_file(corrupt_upload):
    with pytest.raises(ImageValidationError):
        validate_upload(corrupt_upload, corrupt_upload.name)


def test_validate_upload_accepts_valid_image(png_upload):
    # Should not raise.
    validate_upload(png_upload, png_upload.name)