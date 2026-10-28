# Air Sculptor ✏️🖐️

**Draw in the air with your fingertip and export a 3D-printable model — privately,
on your own computer.**

A soft, pastel desktop app. Pinch your fingers in the air to draw a 3D curve, say
what you're making to colour it, and save a watertight `.stl` you can print.
Nothing is ever uploaded: the camera, the mic, and everything you make stay on
your machine, and *you* decide at the end what (if anything) is kept.

> ⚠️ A quick honesty note: the in-app permission popup is an **informed-consent and
> data-handling layer**, not a replacement for your operating system's own camera/
> mic permission. Your OS still asks separately the first time — see the
> [User Guide](USER_GUIDE.md).

---

## What it does

- 🤍 **Consent first.** On launch it asks permission (Always / Once / Deny) and
  explains exactly what's used. Deny = it won't start. Mic is optional.
- ✋ **Hand scan / calibration.** A quick scan learns *your* hand so pinch-to-draw
  feels natural and tracks better.
- 🎨 **Voice colour.** Turn on the mic and say *"the leaves should be green"* —
  the pen turns green and your words show on screen.
- 💬 **Offline help assistant.** A little bot ("Buddy") answers questions about the
  app by voice or text. It only knows this app and politely refuses anything off-
  topic or unsafe — it can't be talked into it, because it has no generative model.
- 🧊 **3D export.** Your stroke becomes a smooth, watertight tube mesh saved as
  `.stl`, ready for any 3D-print slicer.
- 🔒 **Your choice at the end.** Save this session only · save identity + session ·
  save only identity · delete everything. "Delete" really deletes, locally.

## 💡 What it's for (real-world purpose)

Air Sculptor is a playful idea with genuinely useful directions:

- **Wearables you design in the air** — draw a pendant or charm, flip on
  *Keychain mode* to add a printable loop, and 3D-print something giftable.
- **Hands-free / accessible creation** — gesture- and voice-controlled 3D making
  for people who can't comfortably use a mouse, tablet, or stylus.
- **A STEM toy for kids** — "draw it, then hold it": sketch a shape in the air and
  watch it become a real, printable 3D object — a tangible intro to 3D modelling.
- **AR-style signatures & logos** — turn a hand-drawn flourish into a 3D model for
  branding, badges, or toppers.

## 🤖 The ML part — TWO trained models

This project trains **two neural networks from scratch** (PyTorch), on data you
collect yourself, each evaluated with real metrics (accuracy, precision/recall/F1,
confusion matrix). They train in minutes on the M-series GPU (MPS).

**1. Hand-gesture recognizer** — a 1-D temporal CNN that recognises
`draw` / `stop` / `clear` / `save` / `undo` / `idle` and drives the app hands-free,
replacing the pinch rule. Guide: **[ml/README.md](ml/README.md)**.

**2. Facial-expression recognizer** — an MLP that classifies
`neutral` / `happy` / `sad` / `surprised` and colours your pen by your **mood**.
Guide: **[ml/face/README.md](ml/face/README.md)**.

```bash
pip install -r ml/requirements-ml.txt

# gesture model
python ml/make_synthetic_data.py --per-class 150   # or ml/collect_data.py for real data
python ml/train.py --epochs 40 && python ml/evaluate.py

# emotion model
python ml/face/make_synthetic_data.py --per-class 250  # or ml/face/collect_face_data.py
python ml/face/train.py --epochs 40 --lr 5e-4 && python ml/face/evaluate.py

# use them live (either or both)
python app.py --gestures --emotion
```

Everything else below is the classical 3D-geometry side of the app.

## How it works

```
webcam ─▶ MediaPipe Hands ─▶ fingertip (x, y, z)
   │              └─ calibration personalises the pinch threshold
   │        pinch gesture gates recording
   ▼
3D polyline ─▶ resample ─▶ smooth ─▶ parallel-transport tube ─▶ watertight .stl
   ▲
mic ─▶ Vosk (offline speech-to-text) ─▶ colour words + captions / questions to Buddy
```

## Quickstart

```bash
git clone https://github.com/<your-username>/air-sculptor.git
cd air-sculptor
python3 -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Voice is optional: `pip install -r requirements-voice.txt` then
`python download_model.py` once.

**Full step-by-step for macOS / Windows / Linux, including camera permissions and
fixes for every common error, is in [USER_GUIDE.md](USER_GUIDE.md).**

## Privacy

- Camera and microphone are used **only while the app is open**.
- **No video or audio is ever recorded or uploaded.** Voice is transcribed on-
  device and the audio is discarded immediately.
- Saved data (your hand profile, your `.stl`) lives in a folder in your home
  directory and is only written if you choose to keep it.
- The content filter on the assistant is a reasonable keyword guard, not a perfect
  one — it can occasionally be over- or under-cautious.

## Known limitations

- Single-camera depth is approximate, so sculptures are a bit flat in Z (raise the
  depth gain or add a second camera).
- Voice accuracy depends on your mic and the small offline model.

## Contact

Hit a wall you can't solve, or just want to say hi? **saatvika22@outlook.com**

## License

MIT — see [LICENSE](LICENSE). Built with MediaPipe, OpenCV, trimesh, Vosk & pyttsx3.
