"""Central configuration for the CNN Image Classifier application.

All configurable values live here so they are not scattered across modules.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
# Location of the trained Keras model. Intentionally absent until training is
# performed; the application must handle its absence gracefully.
MODEL_PATH: Path = Path("models") / "cnn_optimized.keras"

# Expected model contract (see README for details).
MODEL_INPUT_SHAPE: tuple = (None, 32, 32, 3)  # (batch, height, width, channels)
MODEL_INPUT_DTYPE: str = "float32"
MODEL_OUTPUT_SIZE: int = 10  # one probability per CIFAR-10 class

# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------
IMAGE_HEIGHT: int = 32
IMAGE_WIDTH: int = 32
IMAGE_CHANNELS: int = 3
IMAGE_MODE: str = "RGB"
RGB_MAX: int = 255  # divisor used to normalize pixel values into [0.0, 1.0]

SUPPORTED_EXTENSIONS: tuple = (".jpg", ".jpeg", ".png")

# Maximum upload size (MB). Guarded when the running Streamlit version supports
# it (streamlit >= 1.44 exposed a max_upload_size_mb argument).
MAX_UPLOAD_SIZE_MB: int = 20

# ---------------------------------------------------------------------------
# CIFAR-10 class labels, in the exact order the model is expected to output.
# ---------------------------------------------------------------------------
CLASS_NAMES: list[str] = [
    "Airplane",
    "Automobile",
    "Bird",
    "Cat",
    "Deer",
    "Dog",
    "Frog",
    "Horse",
    "Ship",
    "Truck",
]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_FORMAT: str = "%(asctime)s %(levelname)s [%(name)s] %(message)s"