"""
Data collection harness for gesture recognition.

Run it, then perform each gesture while holding its number key. Every WINDOW
frames of tracked hand become one labelled sample, saved to ml/data/<label>/.
Collect a few hundred samples per gesture, varying speed / angle / distance.

Gestures to record (suggested):
    1 idle   – relaxed hand, no gesture
    2 draw   – thumb + index pinched together
    3 stop   – open palm, fingers spread
    4 clear  – closed fist
    5 save   – thumbs up

Keys:  1-5 hold to record that gesture   ·   d = delete last sample   ·   q = quit

Privacy: this records only numeric hand coordinates, never video or images.
Everything stays in ml/data/ on your computer.
"""

import os
import sys
import time
from collections import deque

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import WINDOW, LABELS, NUM_LANDMARKS, NUM_COORDS  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
KEY_TO_LABEL = {str(i + 1): name for i, name in enumerate(LABELS)}


def _hands():
    import mediapipe as mp
    if not hasattr(mp, "solutions"):
        raise RuntimeError("Install mediapipe==0.10.14 (this build lacks "
                           "the 'solutions' API).")
    return mp.solutions.hands.Hands(max_num_hands=1,
                                    min_detection_confidence=0.6,
                                    min_tracking_confidence=0.6)


def counts():
    out = {}
    for name in LABELS:
        d = os.path.join(DATA_DIR, name)
        out[name] = len(os.listdir(d)) if os.path.isdir(d) else 0
    return out


def main():
    for name in LABELS:
        os.makedirs(os.path.join(DATA_DIR, name), exist_ok=True)

    hands = _hands()
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (check camera permission).")

    buf = deque(maxlen=WINDOW)
    last_saved = None

    # big, resizable preview window with a fullscreen toggle (press f)
    WIN = "Gesture data collection"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1280, 960)
    fullscreen = False

    print("Hold 1-5 to record a gesture, d=delete last, f=fullscreen, q=quit")
    print("Labels:", {str(i + 1): n for i, n in enumerate(LABELS)})

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        have_hand = bool(res.multi_hand_landmarks)
        if have_hand:
            lm = res.multi_hand_landmarks[0].landmark
            pts = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
            buf.append(pts)
            for p in lm:
                cv2.circle(frame, (int(p.x * w), int(p.y * h)), 2,
                           (169, 197, 160), -1)

        key = cv2.waitKey(1) & 0xFF
        c = chr(key) if key != 255 else ""

        if c in KEY_TO_LABEL and have_hand and len(buf) == WINDOW:
            label = KEY_TO_LABEL[c]
            arr = np.stack(buf)                      # (WINDOW, 21, 3)
            fname = os.path.join(DATA_DIR, label, f"{int(time.time()*1000)}.npy")
            np.save(fname, arr)
            last_saved = fname
        elif c == "d" and last_saved and os.path.exists(last_saved):
            os.remove(last_saved)
            last_saved = None
        elif c == "f":
            fullscreen = not fullscreen
            cv2.setWindowProperty(
                WIN, cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
        elif c == "q":
            break

        # HUD
        cnts = counts()
        y = 26
        for i, name in enumerate(LABELS):
            cv2.putText(frame, f"{i+1} {name}: {cnts[name]}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (90, 80, 86), 2)
            y += 26
        cv2.putText(frame, "hold 1-5 record  d delete  f fullscreen  q quit",
                    (10, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (154, 143, 148), 1)
        cv2.imshow(WIN, frame)

    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print("Final counts:", counts())


if __name__ == "__main__":
    main()
