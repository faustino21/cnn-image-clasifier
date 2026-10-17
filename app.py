"""Streamlit web interface for the CNN Image Classifier.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import html
import logging

import pandas as pd
import streamlit as st

import config
import preprocessing
import inference
from inference import ModelOutputError, ModelUnavailableError

# ---------------------------------------------------------------------------
# Logging: send everything to stderr so UI users never see raw tracebacks.
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format=config.LOG_FORMAT,
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("cnn_classifier.app")

st.set_page_config(
    page_title="CNN Image Classifier",
    page_icon="🖼️",
    layout="centered",
)


@st.cache_resource(show_spinner="Loading CNN model...")
def _load_model_cached() -> object:
    """Load the model once per session; reused across reruns."""
    return inference.load_model()


@st.cache_resource
def _model_status() -> str:
    """Return 'ready', 'missing' or 'error' for the current model file."""
    if not inference.model_exists():
        return "missing"
    try:
        inference.load_model()
        return "ready"
    except ModelUnavailableError:
        return "error"


def render_header(status: str) -> None:
    """Render the application title, description and model-availability badge."""
    st.title("CNN Image Classifier")
    st.markdown(
        "Upload an image and this application uses a convolutional neural "
        "network (CNN) to classify it into one of the 10 CIFAR-10 categories: "
        "Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck."
    )

    if status == "ready":
        st.success("✅ Model available — ready to make predictions.")
    elif status == "missing":
        st.info(
            "ℹ️ **Trained model not added yet.**\n\n"
            "Place your trained `.keras` file at "
            "`models/cnn_optimized.keras` and this application will pick it up. "
            "You can still upload an image to preview it below."
        )
    else:
        st.error(
            "❌ **Model could not be loaded.**\n\n"
            "The file exists at `models/cnn_optimized.keras` but failed to load. "
            "Please check the application logs for technical details."
        )


def main() -> None:
    status = _model_status()
    render_header(status)

    uploaded = st.file_uploader(
        "Choose a JPG, JPEG or PNG image to classify.",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded is not None:
        try:
            _validate(uploaded)
        except preprocessing.ImageValidationError as exc:
            st.error(str(exc))
            return

        st.image(
            uploaded,
            caption="Uploaded image (preview)",
            use_container_width=True,
        )

        model = None
        if status == "ready":
            model = _load_model_cached()
            if st.button("🔍 Predict", type="primary"):
                _run_prediction(model, uploaded)

    # No upload: keep the UI clear.
    if status == "missing" and uploaded is None:
        st.caption(
            "An image preview and prediction controls will appear here once you "
            "upload a supported file. Predictions are disabled until the model "
            "is added."
        )


def _validate(file) -> None:
    """Validate an uploaded file before downstream processing."""
    preprocessing.validate_upload(file, file.name)  # raises on error


def _run_prediction(model, uploaded) -> None:
    """Preprocess the uploaded image and run model inference, showing results."""
    try:
        tensor = preprocessing.preprocess_image(uploaded)
    except preprocessing.ImageValidationError as exc:
        st.error(str(exc))
        return

    try:
        results = inference.predict(model, tensor)
    except ModelOutputError as exc:
        st.error(f"Model output problem: {exc}")
        logger.exception("Model output validation failed.")
        return
    except (ModelUnavailableError, RuntimeError) as exc:
        st.error(str(exc))
        logger.exception("Inference failed.")
        return

    _render_results(results, uploaded)


def _render_results(results: dict, uploaded) -> None:
    """Display confidence, probabilities, top-3 and timing."""
    st.subheader("Prediction results")
    col1, col2 = st.columns(2)
    col1.metric(
        "Predicted class",
        results["predicted_class"].upper(),
    )
    col2.metric(
        "Confidence",
        f"{results['confidence'] * 100:.2f}%",
    )

    st.markdown("**Class probabilities**")
    chart = pd.DataFrame(
        {
            "class": config.CLASS_NAMES,
            "probability": results["probabilities"],
        }
    ).sort_values("probability", ascending=True)
    st.bar_chart(chart.set_index("class")["probability"])

    st.markdown("**Top three predictions**")
    for i, item in enumerate(results["top3"], start=1):
        st.markdown(
            f"{i}. **{html.escape(item['class'])}** — "
            f"{item['score'] * 100:.2f}%"
        )

    st.markdown(f"⏱️ Inference time: **{results['inference_time']:.3f} s**")

    st.image(
        uploaded,
        caption="Classified image",
        use_container_width=True,
    )


if __name__ == "__main__":
    main()