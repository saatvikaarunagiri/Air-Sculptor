# 🤖 Gesture Recognition (the ML part)

This is the machine-learning component of Air Sculptor: a hand-gesture
classifier **you train yourself** on data **you collect yourself**, then run live
in the app. It replaces the hardcoded "pinch = draw" rule with a learned model
that recognises several gestures:

| id | gesture | suggested pose | what it does in the app |
|----|---------|----------------|--------------------------|
| 1 | `idle`  | relaxed hand   | do nothing |
| 2 | `draw`  | thumb+index pinch | draw |
| 3 | `stop`  | open palm      | stop drawing |
| 4 | `clear` | fist           | clear the stroke |
| 5 | `save`  | thumbs up      | export the STL |

Unlike the rest of the app (which *uses* a pretrained landmark model), here
**you** design the network, build the dataset, train, and measure — a genuine
end-to-end ML project.

## The pipeline

```
collect_data.py ─▶ ml/data/<gesture>/*.npy        (you record labelled windows)
       │
make_synthetic_data.py (optional, for a dry run) ─┘
       │
   dataset.py   ─▶ normalise + stratified train/val/test split + augmentation
       │
   train.py     ─▶ GestureNet (1-D temporal CNN)  ─▶ artifacts/gesture_model.pt
       │                                              artifacts/history.png
   evaluate.py  ─▶ accuracy, precision/recall/F1, confusion_matrix.png (TEST set)
       │
gesture_recognizer.py ─▶ live inference, imported by the app
```

## How to run it

From the project root, in your activated venv:

```bash
pip install -r ml/requirements-ml.txt
```

**1. Get data.** Either collect real data (recommended):
```bash
python ml/collect_data.py
# hold 1-5 to record each gesture; aim for ~150-300 samples each,
# varying speed, angle, and distance from the camera.
```
…or generate synthetic data for a quick dry run of the whole pipeline:
```bash
python ml/make_synthetic_data.py --per-class 150
```

**2. Train.**
```bash
python ml/train.py --epochs 40
```
Uses your M-series GPU (MPS) automatically. Writes the model and loss/accuracy
curves to `ml/artifacts/`.

**3. Evaluate on the held-out test set.**
```bash
python ml/evaluate.py
```
Prints per-gesture precision / recall / F1 and overall accuracy, and saves a
confusion matrix image.

**4. Use it live.** Launch the app and turn on gesture mode:
```bash
python app.py --gestures
```
If no trained model is found, the app silently falls back to the classic pinch,
so it never breaks.

## Design notes (worth putting in your writeup)

- **Features are translation- and scale-invariant.** Each frame is re-centred on
  the wrist and scaled by hand size, so the model generalises across where your
  hand is and how close you sit. (`features.py`)
- **A gesture is a sequence, not a pose.** Each sample is a 16-frame window, so
  the 1-D CNN learns *motion* (a pinch closing, a palm opening), not just a
  snapshot.
- **Honest evaluation.** The test split is stratified and never seen in training;
  augmentation is applied to the training split only. That's what makes the
  reported accuracy trustworthy.
- **Fails soft.** Missing model or missing torch → the app falls back to pinch.

## Limitations

- Synthetic data only proves the plumbing; real webcam data is what makes it work
  in the app.
- Five gestures, one hand. Adding gestures = add a label in `features.LABELS`,
  collect data, retrain.
