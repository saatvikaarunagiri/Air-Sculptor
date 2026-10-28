# Example outputs

These images were produced by running the pipeline on **synthetic** data
(`python ml/make_synthetic_data.py` → `train.py` → `evaluate.py`). Synthetic
gestures are cleanly separable, so the metrics are a perfect 100% — that only
proves the pipeline works end to end.

**Your real numbers, trained on webcam data you collect, will be lower and more
interesting** — the confusion matrix will show which gestures actually get
confused (e.g. `clear` vs `save` if your fist and thumbs-up look similar), which
is exactly the kind of thing to discuss in a writeup.
