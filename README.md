# Air Sculptor

**Draw in the air with your finger -> export a 3D-printable model.**

A **3D/4D computer vision + ML** app: it tracks your hand in 3D, reconstructs your air-strokes into watertight 3D meshes, and ships **two neural nets I trained from scratch** — a gesture recogniser (hands-free control) and a facial-expression classifier (mood-based color).

```bash
conda activate sculptor
pip install -r requirements.txt
python app.py  
```

| | |
|---|---|
| **3D** | monocular fingertip -> parallel-transport tube -> printable STL |
| **4D ML** | 1-D CNN over 3D hand landmarks *across time* -> 6 gestures |
| **ML #2** | MLP on FaceMesh -> mood colors (happy/sad/surprised/neutral) |
| **Voice** | say a color, ask "Buddy" for help - all offline! |
| **Private** | nothing leaves your machine; you choose what to keep |

**Run with ML:** `python app.py --gestures --emotion` · **Train the models + full setup:** see **[USER_GUIDE.md](USER_GUIDE.md)**

<sub>Python | PyTorch | MediaPipe | OpenCV | trimesh | MIT License | questions -s aatvika22@outlook.com</sub>
