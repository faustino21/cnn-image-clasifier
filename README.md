# CNN Image Classifier

A simple, clean web application built with **Streamlit** that lets you upload an
image and get a classification from a convolutional neural network (CNN) trained
on the **CIFAR-10** dataset (10 classes: Airplane, Automobile, Bird, Cat, Deer,
Dog, Frog, Horse, Ship, Truck).

> **Important:** the trained model is intentionally **not** included. The
> application is fully wired and ready; you add the trained model file later
> (see [Adding the trained model](#adding-the-trained-model)). No placeholder or
> fake model is used — inference stays disabled until a real model is present.

---

## Project structure

```
cnn-image-clasifier/
├── app.py              # Streamlit web interface
├── preprocessing.py    # Image validation + preprocessing (Pillow -> tensor)
├── inference.py        # Model loading, prediction, output validation
├── config.py           # Central configuration (paths, sizes, classes, ...)
├── requirements.txt    # Pinned, compatible dependencies
├── README.md
├── .gitignore          # Excludes venv, caches, and model files
├── models/
│   └── .gitkeep        # <-- put cnn_optimized.keras here later
└── assets/
    └── .gitkeep
```

### Module roles

- **`config.py`** — every tunable value in one place: model path, image size,
  supported extensions, CIFAR-10 class names, upload limit, log format.
- **`preprocessing.py`** — mirrors the training pipeline. Validates the upload
  (extension + decodability), converts to RGB, resizes to 32×32, converts to a
  `float32` array, normalizes to `[0.0, 1.0]`, and adds the batch dimension.
  Independent of Streamlit so it can be unit-tested on its own.
- **`inference.py`** — loads the real `.keras` model (only when present), runs
  `model.predict()`, validates the output has exactly 10 values, and returns the
  predicted class, confidence, probabilities, top‑3 and inference time.
- **`app.py`** — the Streamlit UI: header + model status badge, image upload and
  preview, and the results panel. Model loading is cached with
  `st.cache_resource`.

---

## Prerequisites

- **Python 3.11** (TensorFlow 2.17 supports Python 3.9–3.12; 3.11 is the
  recommended target).
- `pip` and, optionally, `git`.

## Installation

Create and activate a virtual environment, then install the pinned
dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows activation:**

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

> TensorFlow is a large download; installation may take a few minutes.

## Run the application locally

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501` by default. **Even without a model the
app starts successfully** — it shows a status message that the model has not
been added yet, and still lets you upload and preview an image. Prediction is
disabled until a model is present.

## Adding the trained model

When your training completes, place the trained model here:

```
models/cnn_optimized.keras
```

The file is created by saving with `model.save("models/cnn_optimized.keras")` in
your training script (the Keras `.keras` format). Refresh the Streamlit page —
the status badge turns green and the **Predict** button becomes available.

If the file exists but fails to load, the app shows a friendly error and the
technical details are written to the application log (stderr).

### Expected model contract

| Property      | Value                          |
|---------------|--------------------------------|
| Input shape   | `(None, 32, 32, 3)`            |
| Input format  | RGB                            |
| Input range   | `0.0` – `1.0`                  |
| Output        | 10 class probabilities         |
| Output order  | exactly `CLASS_NAMES` in `config.py` |

The output index maps to the classes in `config.py`:

```python
CLASS_NAMES = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck",
]
```

## Running tests

Tests are written with `pytest` and do **not** require the trained model (the
model layer is exercised with mocks). A single dummy fixture for preprocessing
(could be a Pillow-checkable PNG) is created at test time:

```bash
pip install pytest        # already present? if not, install it
pytest -q
```

## Known limitations

- The model is **not** trained or included; classification only works after you
  add `models/cnn_optimized.keras`.
- Input images are downscaled to 32×32 pixels, matching CIFAR‑10; very large
  images lose most detail before classification.
- The application exposes model output directly; for a production deployment you
  would add authentication and a load balancer (out of scope here).
- Supported uploads are limited to JPG, JPEG and PNG (max ~20 MB in the running
  Streamlit version).