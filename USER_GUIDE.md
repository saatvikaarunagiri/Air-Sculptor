# Air Sculptor — User Guide & Manual

Everything you need, step by step, for **macOS, Windows, and Linux**. Stuck on
something not covered here? → **saatvika22@outlook.com**

---

## Contents
1. [What you need](#1-what-you-need)
2. [Install](#2-install)
3. [Run the app](#3-run-the-app)
4. [Controls](#4-controls)
5. [Keychain mode (printing & wearing)](#5-keychain-mode)
6. [Voice & the Buddy assistant](#6-voice--buddy)
7. [The ML models — train your own](#7-the-ml-models)
8. [Privacy & ending a session](#8-privacy)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. What you need
- A computer with a **webcam**.
- **Python 3.10–3.12** (NOT 3.13 — mediapipe has no 3.13 build). Check: `python --version`.
- The easiest way to get 3.12 is a **conda environment** (below).

## 2. Install

**Recommended — one conda environment (Python 3.12):**
```bash
conda create -n sculptor python=3.12 -y
conda activate sculptor

cd "/path/to/air-sculptor"
pip install -r requirements.txt            # core app
pip install -r ml/requirements-ml.txt      # ML (PyTorch etc.)
```

Verify it worked:
```bash
python -c "import cv2, mediapipe as mp; print('cv2 ok | solutions:', hasattr(mp,'solutions'))"
# want:  cv2 ok | solutions: True
```

**Linux only** also needs Tk for the windows: `sudo apt install python3-tk`
(Fedora: `python3-tkinter` · Arch: `tk`).

**Voice (optional):** `pip install -r requirements-voice.txt` then
`python download_model.py` (downloads a ~40 MB offline speech model once).

> ⚠️ **One environment, always.** Pick `sculptor` and `conda activate sculptor`
> every time. Mixing a `venv` and conda is the #1 cause of "ModuleNotFoundError".

## 3. Run the app
```bash
conda activate sculptor
python app.py
```
A **permission window** appears first — choose **Always / Once / Deny** (Deny
exits). Mic is optional.

| Command | Mode |
|---|---|
| `python app.py` | Pinch to draw (classic) |
| `python app.py --gestures` | Trained gesture control (needs a trained model) |
| `python app.py --emotion` | Mood colors from your expression (needs a trained model) |
| `python app.py --gestures --emotion` | Both ML models live |

> **First launch (macOS):** your Mac shows its *own* camera/mic prompt — click OK.
> Missed it? System Settings → Privacy & Security → Camera → enable Terminal,
> then quit & reopen Terminal.

## 4. Controls
1. Click **Calibrate**, show your hand, pinch slowly ~6 s so it learns your hand.
2. **Pinch** thumb + index and move = draw. Separate = stop.
3. **Save** (or `s`) exports an STL into your session folder.
4. **Clear** wipes the current stroke.
5. Drag the window edges to resize; the video fills the space. Press **f** in the
   data-collector windows for fullscreen.

## 5. Keychain mode
Tick **🔑 Keychain mode** before saving. It fuses a small loop onto your
sculpture so the printed piece can be threaded onto a keyring or necklace. Opens
in any slicer (Bambu Studio, Cura, PrusaSlicer) or macOS Quick Look.

## 6. Voice & Buddy
- With the mic on, say what you're making — *"the leaves should be green"* — to set
  the pen color and caption.
- **Buddy** is an offline help bot: type in the box (left of **Send**, or press
  Enter) or speak a question. It only answers about the app and politely declines
  anything off-topic — by design, so it's safe.

## 7. The ML models
This project trains **two neural networks** on data **you collect**. Both use
`ml/requirements-ml.txt` and run on the Apple-Silicon GPU automatically.

### A) Gesture recognizer (`ml/`) — 6 gestures, hands-free control
```bash
# 1) collect YOUR hand data (hold a number key to record each gesture)
python ml/collect_data.py          # 1 idle 2 draw 3 stop 4 clear 5 save 6 undo
#    aim for ~150-300 samples each; vary angle/speed/distance
# 2) train            3) evaluate
python ml/train.py --epochs 40
python ml/evaluate.py              # prints accuracy + saves a confusion matrix
# 4) use it live
python app.py --gestures
```
*(Just testing the pipeline? `python ml/make_synthetic_data.py --per-class 150`
makes fake data — but a model trained on it won't know YOUR hand. Collect real
data for real use.)*

### B) Expression recognizer (`ml/face/`) — mood colors
```bash
python ml/face/collect_face_data.py            # 1 neutral 2 happy 3 sad 4 surprised
python ml/face/train.py --epochs 40 --lr 5e-4
python ml/face/evaluate.py
python app.py --emotion
```
Details & design notes: [ml/README.md](ml/README.md) and
[ml/face/README.md](ml/face/README.md). A fill-in report template is in
[REPORT.md](REPORT.md).

## 8. Privacy
Nothing is uploaded; no video/audio is recorded. When you close the app you choose
what to keep, all on your machine: **this session only · identity + session ·
identity only · delete everything** (delete really deletes).

## 9. Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'cv2'` | Wrong environment. `conda activate sculptor`, then `pip install -r requirements.txt`. Prompt must show `(sculptor)`. |
| `mediapipe ... has no attribute 'solutions'` | You're on Python 3.13. Use the 3.12 conda env in [Install](#2-install). |
| No camera window / "Could not open webcam" | Allow Camera for Terminal in OS privacy settings, quit & reopen Terminal. Close Zoom/FaceTime/Photo Booth. |
| numpy `_ARRAY_API` error | `pip install "numpy<2"` |
| `No module named 'tkinter'` (Linux) | `sudo apt install python3-tk` |
| `--gestures`/`--emotion` do nothing | You haven't trained that model yet — see [section 7](#7-the-ml-models). |
| A camera "busy"/frozen, or a stuck background job | Close other camera apps; `kill <pid>` any leftover `python` process, or reopen Terminal. |

Still stuck, or found a bug? → **saatvika22@outlook.com**
