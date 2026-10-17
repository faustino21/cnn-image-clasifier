"""Rendering tests for the low-confidence warning.

The app's prediction results panel is driven directly inside a Streamlit
script context (AppTest.from_string), so warnings, metrics and the class
probability chart are captured and asserted without requiring a trained model
or a real file upload. Prediction results are mocked.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from streamlit.testing.v1 import AppTest

import config

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _vector_with_max(top_conf: float, top_index: int = 3) -> np.ndarray:
    """Return a length-10 probability vector whose max equals ``top_conf``."""
    probs = np.full(config.MODEL_OUTPUT_SIZE, (1.0 - top_conf) / 9.0, dtype=np.float32)
    probs[top_index] = top_conf
    return probs


def _png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 48), (120, 60, 200)).save(buf, format="PNG")
    return buf.getvalue()


def _render_results(top_conf: float) -> AppTest:
    """Render the results panel for a mocked confidence and return AppTest."""
    script = (
        "import sys, io, numpy as np\n"
        f"sys.path.insert(0, {str(PROJECT_ROOT)!r})\n"
        "import app, config\n"
        f"probs = {_vector_with_max(top_conf).tolist()!r}\n"
        "vector = np.array(probs, dtype=np.float32)\n"
        "top = np.argsort(vector)[::-1][:3]\n"
        "top3 = [{'index': int(i), 'class': config.CLASS_NAMES[i], "
        "'score': float(vector[i])} for i in top]\n"
        f"results = {{'predicted_index': 3, 'predicted_class': "
        f"config.CLASS_NAMES[3], 'confidence': {top_conf}, "
        f"'probabilities': vector, 'top3': top3, 'inference_time': 0.11}}\n"
        "app._render_results(results, io.BytesIO("
        f"{_png_bytes()!r}))\n"
    )
    return AppTest.from_string(script).run()


# --- threshold behaviour -----------------------------------------------


def test_confidence_below_threshold_shows_warning():
    at = _render_results(top_conf=0.49)
    warnings = [w.value for w in at.warning]
    assert any(config.LOW_CONFIDENCE_MESSAGE in msg for msg in warnings)


def test_confidence_at_threshold_shows_no_warning():
    at = _render_results(top_conf=0.50)
    assert at.warning == []
    assert len(at.metric) >= 2


def test_high_confidence_shows_no_warning():
    at = _render_results(top_conf=0.85)
    assert at.warning == []
    assert len(at.metric) >= 2


# --- prediction & chart remain visible ----------------------------------


def test_low_confidence_still_shows_prediction_and_confidence():
    at = _render_results(top_conf=0.49)
    labels = [m.label for m in at.metric]
    assert "Predicted class" in labels
    assert "Confidence" in labels
    conf_values = [m.value for m in at.metric if m.label == "Confidence"]
    assert conf_values and conf_values[0].endswith("%")


def test_class_probability_chart_still_renders_on_low_confidence():
    at = _render_results(top_conf=0.49)
    # The bar_chart proto can surface under different element names across
    # Streamlit versions; accept any of the known vega-lite chart elements.
    charts = sum(
        len(at.get(name))
        for name in ("bar_chart", "arrow_vega_lite_chart", "vega_lite_chart")
    )
    assert charts >= 1


# --- warning message wording --------------------------------------------


def test_warning_does_not_claim_out_of_category():
    # Must express uncertainty ("mungkin"), never a definitive exclusion.
    msg = config.LOW_CONFIDENCE_MESSAGE.lower()
    assert "mungkin" in msg
    assert "tidak mungkin" not in msg
    assert "bukan" not in msg
    assert "pastikan bukan" not in msg