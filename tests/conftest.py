"""Shared test fixtures.

Provides lightweight in-memory image files so tests never rely on a trained
model, a downloaded dataset, or any on-disk asset.
"""
from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image


def _png_bytes(color: tuple[int, int, int] = (120, 60, 200)) -> bytes:
    """Return the bytes of a small, valid RGB PNG."""
    img = Image.new("RGB", (64, 48), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class _StubFile:
    """Minimal file-like object mimicking a Streamlit UploadedFile."""

    def __init__(self, data: bytes, name: str):
        self._data = data
        self._stream = io.BytesIO(data)
        self.name = name

    def seek(self, offset: int, whence: int = 0):
        return self._stream.seek(offset, whence)

    def tell(self) -> int:
        return self._stream.tell()

    def read(self, size: int = -1) -> bytes:
        return self._stream.read(size)


@pytest.fixture
def png_upload() -> _StubFile:
    """A valid RGB PNG uploaded with a .png name."""
    return _StubFile(_png_bytes(), "sample.png")


@pytest.fixture
def corrupt_upload() -> _StubFile:
    """A file claiming to be an image but containing garbage bytes."""
    return _StubFile(b"this is definitely not image data", "broken.png")


@pytest.fixture
def unsupported_upload() -> _StubFile:
    """A valid PNG carrying an unsupported file extension."""
    return _StubFile(_png_bytes(), "notes.txt")