"""
LAB 1 – Iris Predictor (Terminal Version)
-----------------------------------------
Run the Iris classifier from the saved scikit-learn model without any HTTP server.

Usage (interactive):
    python main.py

Usage (one-shot):
    python main.py --features 5.1 3.5 1.4 0.2

Usage (batch CSV: file with 4 numeric columns, no header):
    python main.py --csv samples.csv
"""

import argparse
import sys
from pathlib import Path
import joblib
import numpy as np

# --- Paths --------------------------------------------------------------------
# Adjust if your model lives elsewhere. Typical options:
CANDIDATE_PATHS = [
    Path(__file__).parent / "model.joblib",
    Path(__file__).parent / "models" / "iris_model.joblib",
    Path(__file__).parent / "models" / "model.joblib",
]

# --- Species helpers -----------------------------------------------------------
DEFAULT_SPECIES = ["setosa", "versicolor", "virginica"]

def _species_from_model(model):
    """
    Try to recover class names from the trained model/pipeline.
    Falls back to DEFAULT_SPECIES if not available.
    """
    try:
        # If it's a plain estimator with .classes_
        classes = getattr(model, "classes_", None)
        if classes is not None:
            # If already strings, return as-is
            if all(isinstance(c, str) for c in classes):
                return list(classes)
            # Otherwise map common encodings 0/1/2 -> species
            if set(classes) == {0, 1, 2}:
                return DEFAULT_SPECIES

        # If it's a pipeline, final step usually has .classes_
        if hasattr(model, "named_steps"):
            for step in model.named_steps.values():
                classes = getattr(step, "classes_", None)
                if classes is not None:
                    if all(isinstance(c, str) for c in classes):
                        return list(classes)
                    if set(classes) == {0, 1, 2}:
                        return DEFAULT_SPECIES
    except Exception:
        pass
    return DEFAULT_SPECIES

# --- Model I/O ----------------------------------------------------------------
def load_model():
    for p in CANDIDATE_PATHS:
        if p.exists():
            model = joblib.load(p)
            print(f"✅ Loaded model: {p}")
            return model
    raise FileNotFoundError(
        "❌ Could not find a model file. Tried:\n" + "\n".join(f"- {p}" for p in CANDIDATE_PATHS)
    )

# --- Prediction ---------------------------------------------------------------
def predict_one(model, features, species_names):
    """
    features: length-4 iterable [sepal_length, sepal_width, petal_length, petal_width]
    returns: (pred_idx, pred_label, proba_dict or None)
    """
    x = np.asarray(features, dtype=float).reshape(1, -1)
    pred_idx = int(model.predict(x)[0])

    # Map index → label
    if 0 <= pred_idx < len(species_names):
        pred_label = species_names[pred_idx]
    else:
        # In case model already outputs string labels
        pred_label = str(pred_idx)
        if pred_label not in species_names:
            species_names = DEFAULT_SPECIES  # steady fallback

    # Try probabilities if available
    proba = None
    try:
        p = model.predict_proba(x)[0]
        proba = {species_names[i]: float(p[i]) for i in range(len(species_names))}
    except Exception:
        pass

    return pred_idx, pred_label, proba

def predict_csv(model, path, species_names):
    data = np.loadtxt(path, delimiter=",", dtype=float)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] != 4:
        raise ValueError(f"CSV must have exactly 4 columns (got {data.shape[1]}).")
    preds = model.predict(data)
    # Best-effort mapping
    labels = []
    for v in preds:
        if isinstance(v, (int, np.integer)) and 0 <= int(v) < len(species_names):
            labels.append(species_names[int(v)])
        else:
            labels.append(str(v))
    return labels

# --- CLI ----------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Iris predictor (terminal)")
    parser.add_argument("--features", nargs=4, type=float, metavar=("SL", "SW", "PL", "PW"),
                        help="Iris features: sepal_len sepal_wid petal_len petal_wid")
    parser.add_argument("--csv", type=str, help="Path to CSV (no header, 4 cols)")
    parser.add_argument("--example", action="store_true",
                        help="Print an example and exit")
    return parser.parse_args()

def print_example():
    print(
        "Example usage:\n"
        "  python main.py --features 5.1 3.5 1.4 0.2\n"
        "  python main.py --csv samples.csv\n"
        "Interactive mode (no args):\n"
        "  python main.py\n"
    )

def interactive_loop(model, species_names):
    print("\n💬 Interactive Iris prediction (Ctrl+C to exit)")
    print("Enter 4 comma-separated values: sepal_len,sepal_wid,petal_len,petal_wid")
    print("Example: 5.1,3.5,1.4,0.2")

    while True:
        try:
            raw = input("\nFeatures> ").strip()
            if not raw:
                continue
            parts = [float(t) for t in raw.split(",")]
            if len(parts) != 4:
                print("⚠️ Please enter exactly 4 numbers.")
                continue
            _, label, proba = predict_one(model, parts, species_names)
            if proba:
                ranked = sorted(proba.items(), key=lambda kv: kv[1], reverse=True)
                conf = ", ".join([f"{k}={v:.3f}" for k, v in ranked])
                print(f"🔮 Prediction: {label}  (proba → {conf})")
            else:
                print(f"🔮 Prediction: {label}")
        except KeyboardInterrupt:
            print("\n👋 Exiting.")
            break
        except Exception as e:
            print(f"⚠️ Error: {e}")

def main():
    args = parse_args()
    if args.example:
        print_example()
        sys.exit(0)

    model = load_model()
    species_names = _species_from_model(model)

    # One-shot or batch modes
    if args.features:
        _, label, proba = predict_one(model, args.features, species_names)
        if proba:
            ranked = sorted(proba.items(), key=lambda kv: kv[1], reverse=True)
            conf = ", ".join([f"{k}={v:.3f}" for k, v in ranked])
            print(f"🔮 Prediction: {label}  (proba → {conf})")
        else:
            print(f"🔮 Prediction: {label}")
        return

    if args.csv:
        labels = predict_csv(model, args.csv, species_names)
        for i, lbl in enumerate(labels, 1):
            print(f"row {i}: {lbl}")
        return

    # Otherwise interactive
    interactive_loop(model, species_names)

if __name__ == "__main__":
    main()
