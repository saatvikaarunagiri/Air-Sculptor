"""
Data collection harness for facial-expression recognition.

Make an expression and hold its key to record samples:

    1 neutral     2 happy     3 sad     4 surprised

Keys:  1-4 hold to record   ·   d = delete last   ·   f = fullscreen   ·   q = quit

Aim for a few hundred of each, varying your pose and distance slightly.

Privacy: saves only numeric face-landmark coordinates, never video or images.
Everything stays in ml/face/data/ on your computer.
"""

import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import LABELS, NUM_LANDMARKS                   # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
KEY_TO_LABEL = {str(i + 1): name for i, name in enumerate(LABELS)}


def _face():
    import mediapipe as mp
    if not hasattr(mp, "solutions"):
        raise RuntimeError("Install mediapipe==0.10.14 (this build lacks "
                           "the 'solutions' API).")
    return mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1, refine_landmarks=False,
        min_detection_confidence=0.5, min_tracking_confidence=0.5)


def counts():
    return {n: (len(os.listdir(os.path.join(DATA_DIR, n)))
                if os.path.isdir(os.path.join(DATA_DIR, n)) else 0)
            for n in LABELS}


def main():
    for name in LABELS:
        os.makedirs(os.path.join(DATA_DIR, name), exist_ok=True)

    face = _face()
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (check camera permission).")

    WIN = "Expression data collection"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1280, 960)
    fullscreen = False
    last_saved = None
    print("Hold 1-4 to record an expression, d=delete, f=fullscreen, q=quit")
    print("Labels:", KEY_TO_LABEL)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        res = face.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        have = bool(res.multi_face_landmarks)
        pts = None
        if have:
            lm = res.multi_face_landmarks[0].landmark
            pts = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
            for p in lm[::6]:                             # sparse dots
                cv2.circle(frame, (int(p.x * w), int(p.y * h)), 1,
                           (169, 197, 160), -1)

        key = cv2.waitKey(1) & 0xFF
        c = chr(key) if key != 255 else ""

        if c in KEY_TO_LABEL and have and pts is not None \
                and pts.shape[0] == NUM_LANDMARKS:
            label = KEY_TO_LABEL[c]
            fn = os.path.join(DATA_DIR, label, f"{int(time.time()*1000)}.npy")
            np.save(fn, pts)
            last_saved = fn
        elif c == "d" and last_saved and os.path.exists(last_saved):
            os.remove(last_saved); last_saved = None
        elif c == "f":
            fullscreen = not fullscreen
            cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
        elif c == "q":
            break

        cnts = counts()
        y = 26
        for i, name in enumerate(LABELS):
            cv2.putText(frame, f"{i+1} {name}: {cnts[name]}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (90, 80, 86), 2)
            y += 26
        cv2.putText(frame, "hold 1-4 record  d delete  f fullscreen  q quit",
                    (10, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (154, 143, 148), 1)
        cv2.imshow(WIN, frame)

    cap.release(); cv2.destroyAllWindows(); face.close()
    print("Final counts:", counts())


if __name__ == "__main__":
    main()
