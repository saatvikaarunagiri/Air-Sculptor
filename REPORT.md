# Air Sculptor — Project Report

## 1. Summary

Air Sculptor is a real-time computer-vision application that lets a user draw in the air with their fingertip and export a 3D-printable model. On top of the geometry engine, it adds **two machine-learning models trained from scratch**: a hand gesture recognizer for hands-free control and a facial-expression recognizer that drives mood-based colouring. The system runs fully on-device (Apple M-series, MPS) and is privacy-first (no video/audio leaves the machine).

## 2. Problem & motivation

[1–2 sentences: why air-drawing + ML control is interesting, and a real use you
care about — e.g. accessible/hands-free 3D creation, printable wearables.]

## 3. System architecture

```
Webcam ─▶ MediaPipe (hands + face landmarks)
   ├─ 3D geometry engine: polyline → parallel-transport tube → watertight STL
   ├─ ML model 1: gesture recognizer (1-D CNN)  → app control
   └─ ML model 2: emotion recognizer (MLP)      → mood colour
UI: Tkinter (pastel), offline voice (Vosk/pyttsx3), local-only session storage
```

Key engineering: threaded camera capture, fail-soft ML loading, translation/ scale-invariant feature design, reproducible stratified data splits.

## 4. Machine learning

### 4.1 Gesture recognizer
- **Task:** classify [6] gestures (idle/draw/stop/clear/save/undo) from hand-
  landmark **sequences** (16-frame windows).
- **Features:** per-frame 21×3 landmarks, wrist-centred and scaled by hand size
  (translation/scale invariant); window resampled to fixed length.
- **Model:** 1-D temporal CNN, [~50k] parameters (PyTorch).
- **Data:** [N] samples collected (≈[n]/class), stratified 70/15/15 split,
  augmentation (jitter/scale/time-shift) on train only.
- **Results (held-out test):** accuracy **[0.__]**; per-class F1 [list];
  confusion matrix in `ml/examples/`. [1 sentence on main confusion.]

### 4.2 Facial-expression recognizer
- **Task:** classify [4] expressions (neutral/happy/sad/surprised) from a single
  face-landmark frame.
- **Features:** 468×3 FaceMesh landmarks, nose-centred and scaled by inter-ocular
  distance → 1404-d vector.
- **Model:** MLP, [~400k] parameters; gradient clipping for stable training.
- **Data:** [N] samples ([n]/class), stratified split, augmentation on train only.
- **Results (held-out test):** accuracy **[0.__]**; confusion matrix in
  `ml/face/examples/`. [1 sentence — e.g. happy/neutral most confused.]

## 5. 3D geometry (software engineering)

The stroke (a 3D polyline over time) is resampled by arc length, smoothed, and
swept into a watertight tube mesh using **parallel-transport frames** (avoids
cross-section twist), with capped ends so slicers accept it. Optional *keychain
mode* fuses a torus loop for a printable, wearable result. Output: STL.

## 6. Privacy & UX

Consent gate (Always/Once/Deny) before any capture; all processing on-device; no
video/audio recorded; four end-of-session data choices with real deletion;
offline voice; offline FAQ assistant with a content filter.

## 7. Limitations & future work

[Monocular depth is approximate (flat Z); models personalise to the collector;
small datasets. Future: stereo depth, more gestures/expressions, boolean-union
loop, Ollama-backed conversational assistant.]

## 8. How to reproduce

See `README.md`, `ml/README.md`, and `ml/face/README.md`. Collect data → train →
evaluate → `python app.py --gestures --emotion`.
