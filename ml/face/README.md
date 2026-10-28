# 🙂 Facial-Expression Recognition (second ML model)

A second machine-learning model you train yourself: a facial-expression
classifier (`neutral` / `happy` / `sad` / `surprised`) that, in the app, colours
your pen by your **mood** — happy → gold, sad → blue, surprised → purple,
neutral → lavender.

This is a *separate* model from the gesture recognizer, built the same honest
way: you collect the data, design the network, train, and evaluate with real
metrics. Together with the gesture model, the project trains **two** neural nets.

## The pipeline

```
collect_face_data.py ─▶ ml/face/data/<emotion>/*.npy   (label your expressions)
make_synthetic_data.py (optional dry run) ────────────┘
      │
  dataset.py  ─▶ normalise (nose-centred, eye-scaled) + split + augment
      │
  train.py    ─▶ FaceEmotionNet (MLP over 1404 landmark features) ─▶ emotion_model.pt
      │                                                               history.png
  evaluate.py ─▶ accuracy, precision/recall/F1, confusion_matrix.png (TEST set)
      │
emotion_recognizer.py ─▶ live inference, used by the app
```

## How to run it

From the project root, in your activated env (needs `ml/requirements-ml.txt`):

```bash
# 1. data — synthetic dry run:
python ml/face/make_synthetic_data.py --per-class 250
#    …or real data (recommended): make each face and hold its key
python ml/face/collect_face_data.py           # 1 neutral 2 happy 3 sad 4 surprised

# 2. train (grad-clipping keeps it stable; uses MPS automatically)
python ml/face/train.py --epochs 40 --lr 5e-4

# 3. evaluate on the held-out test set
python ml/face/evaluate.py

# 4. use it live — your mood colours the pen
python app.py --emotion
#    combine with gestures:
python app.py --gestures --emotion
```

If no trained model exists, `--emotion` is simply ignored (fails soft).

## Design notes (for your writeup)

- **Expression is a pose, so one frame = one sample.** The input is a single
  normalised face (nose-centred, scaled by inter-ocular distance), so the model
  is invariant to where your face is and how close you sit.
- **An MLP fits the static-pose task** — small, fast, trains in seconds.
- **Honest metrics.** Stratified held-out test set; augmentation on train only;
  gradient clipping for stable training. Expect realistic confusions (happy vs
  neutral), not a perfect score.

## Limitations

- Four expressions, one face. Add one by extending `features.LABELS`, collecting
  data, and retraining.
- Trained on *your* expressions, so it personalises to you — collect a variety of
  angles/lighting for robustness.
