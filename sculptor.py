"""
The Air Sculptor vision core: hand tracking, a short calibration "scan",
voice-driven colour, air-drawing, and export to a watertight STL.

This module is UI-agnostic. `SculptorProcessor.process(frame_bgr)` is handed to
the camera thread; it returns the annotated frame. The Tkinter app reads state
off the processor (status text, point count) and calls save()/calibrate()/etc.
"""

import threading

import cv2
import numpy as np
import trimesh

from core.theme import BGR, PALETTE


def _create_hands():
    """Create a MediaPipe Hands tracker, with a friendly error if this
    mediapipe build lacks the classic `solutions` API."""
    import mediapipe as mp
    if not hasattr(mp, "solutions"):
        raise RuntimeError(
            "This mediapipe build has no 'solutions' API. Install the pinned "
            "version:  pip install mediapipe==0.10.14")
    return mp.solutions.hands.Hands(
        max_num_hands=1, min_detection_confidence=0.6,
        min_tracking_confidence=0.6)


def _create_face():
    """MediaPipe FaceMesh for the emotion feature (optional)."""
    import mediapipe as mp
    if not hasattr(mp, "solutions"):
        raise RuntimeError("Install mediapipe==0.10.14 for the emotion feature.")
    return mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1, refine_landmarks=False,
        min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Colour words the user can speak ("make the leaves green") -> BGR for drawing.
COLOR_WORDS = {
    "red": (60, 60, 230), "crimson": (60, 60, 230),
    "orange": (60, 150, 240), "yellow": (80, 210, 240),
    "green": (90, 190, 120), "mint": (150, 210, 170),
    "blue": (210, 150, 90), "sky": (230, 190, 130),
    "purple": (200, 130, 180), "lavender": (228, 182, 201),
    "pink": (184, 180, 232), "rose": (160, 150, 220),
    "white": (245, 245, 245), "black": (40, 40, 40),
    "brown": (70, 95, 140), "gold": (90, 190, 220),
}

# Emotion -> pen-colour mood (BGR). Warm when happy, cool when sad, etc.
MOOD_COLORS = {
    "happy":     COLOR_WORDS["gold"],
    "sad":       COLOR_WORDS["blue"],
    "surprised": COLOR_WORDS["purple"],
    "neutral":   COLOR_WORDS["lavender"],
}


# ---------------- geometry (same tube builder as the CLI version) ----------

def resample_polyline(pts, n):
    pts = np.asarray(pts, float)
    if len(pts) < 2:
        return pts
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    if arc[-1] == 0:
        return pts[:1].repeat(n, axis=0)
    s = np.linspace(0, arc[-1], n)
    return np.stack([np.interp(s, arc, pts[:, d]) for d in range(3)], axis=1)


def smooth_polyline(pts, window=7):
    pts = np.asarray(pts, float)
    if len(pts) < window:
        return pts
    k = np.ones(window) / window
    out = np.stack([np.convolve(pts[:, d], k, mode="same") for d in range(3)],
                   axis=1)
    out[0], out[-1] = pts[0], pts[-1]
    return out


def tube_mesh(pts, radius, sides=20):
    pts = np.asarray(pts, float)
    n = len(pts)
    T = np.zeros_like(pts)
    T[1:-1] = pts[2:] - pts[:-2]
    T[0], T[-1] = pts[1] - pts[0], pts[-1] - pts[-2]
    T /= (np.linalg.norm(T, axis=1, keepdims=True) + 1e-12)

    ref = np.array([0, 0, 1.0])
    if abs(np.dot(T[0], ref)) > 0.9:
        ref = np.array([0, 1.0, 0])
    N = np.cross(T[0], ref); N /= np.linalg.norm(N) + 1e-12
    normals = [N]
    for i in range(1, n):
        axis = np.cross(T[i - 1], T[i]); s = np.linalg.norm(axis)
        if s < 1e-8:
            normals.append(normals[-1]); continue
        axis /= s
        ang = np.arccos(np.clip(np.dot(T[i - 1], T[i]), -1, 1))
        p = normals[-1]
        rot = (p * np.cos(ang) + np.cross(axis, p) * np.sin(ang)
               + axis * np.dot(axis, p) * (1 - np.cos(ang)))
        normals.append(rot / (np.linalg.norm(rot) + 1e-12))
    normals = np.array(normals)

    theta = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    ct, st = np.cos(theta), np.sin(theta)
    rings = np.empty((n, sides, 3))
    for i in range(n):
        B = np.cross(T[i], normals[i])
        rings[i] = pts[i] + radius * (ct[:, None] * normals[i] + st[:, None] * B)
    verts = rings.reshape(-1, 3)

    faces = []
    for i in range(n - 1):
        for j in range(sides):
            a = i * sides + j; b = i * sides + (j + 1) % sides
            c = (i + 1) * sides + j; d = (i + 1) * sides + (j + 1) % sides
            faces += [[a, b, d], [a, d, c]]
    c0, c1 = len(verts), len(verts) + 1
    verts = np.vstack([verts, pts[0], pts[-1]])
    base = (n - 1) * sides
    for j in range(sides):
        faces += [[c0, (j + 1) % sides, j],
                  [c1, base + j, base + (j + 1) % sides]]

    mesh = trimesh.Trimesh(vertices=verts, faces=np.array(faces), process=True)
    mesh.fix_normals()
    return mesh


# ---------------- the processor -------------------------------------------

CALIB_FRAMES = 150        # ~5-8 s of samples


class SculptorProcessor:
    def __init__(self, profile=None, create_hands=True, recognizer=None,
                 emotion=None):
        self._hands = _create_hands() if create_hands else None
        self._lock = threading.Lock()

        # Optional trained gesture recognizer (ml/). When present and ready,
        # gestures drive the app; otherwise we fall back to the pinch rule.
        self.recognizer = recognizer
        self.gesture_mode = bool(recognizer and getattr(recognizer, "available", False))
        self.gesture_label = "idle"
        self.save_requested = False        # app polls + clears this

        # Optional trained emotion recognizer (ml/face/). When present, the pen
        # colour follows your mood and the detected emotion shows on screen.
        self.emotion = emotion
        self.emotion_mode = bool(emotion and getattr(emotion, "available", False))
        self._face = _create_face() if (self.emotion_mode and create_hands) else None
        self.emotion_label = "neutral"

        self.path = []
        self.drawing = False
        self.mode = "idle"                 # "idle" | "calibrating" | "sculpting"
        self.color = COLOR_WORDS["lavender"]
        self.caption = ""
        self.status = "Press Calibrate to scan your hand"

        # personalization (identity)
        self.pinch_on = 0.32               # ratio of pinch dist to hand span
        self.pinch_off = 0.45
        self.hand_scale = None
        if profile:
            self.pinch_on = profile.get("pinch_on", self.pinch_on)
            self.pinch_off = profile.get("pinch_off", self.pinch_off)
            self.hand_scale = profile.get("hand_scale")
            self.mode = "sculpting"
            self.status = "Welcome back — pinch to draw"

        self._calib_ratios = []
        self._calib_spans = []

    # ---- public controls (called from the UI thread) ----
    def start_calibration(self):
        with self._lock:
            self._calib_ratios, self._calib_spans = [], []
            self.mode = "calibrating"
            self.status = "Scanning… show your hand and pinch slowly a few times"

    def clear(self):
        with self._lock:
            self.path = []

    def set_color_from_text(self, text):
        """Parse a spoken/typed phrase for a colour word; set caption too."""
        t = text.lower()
        for word, bgr in COLOR_WORDS.items():
            if word in t.split() or word in t:
                with self._lock:
                    self.color = bgr
                    self.caption = text
                return word
        with self._lock:
            self.caption = text
        return None

    def profile(self):
        return {"pinch_on": self.pinch_on, "pinch_off": self.pinch_off,
                "hand_scale": self.hand_scale}

    def has_stroke(self):
        return len(self.path) >= 8

    def save_stl(self, out_path, depth_gain=2.0, radius_mm=3.0, loop=False):
        """Export the stroke as a printable STL. If loop=True, attach a small
        ring at the top so the piece is wearable (keychain / pendant)."""
        with self._lock:
            pts = np.array(self.path, float)
        if len(pts) < 8:
            return None
        pts[:, 2] *= depth_gain
        pts[:, 1] = 1.0 - pts[:, 1]
        pts -= pts.mean(axis=0)
        pts *= 100.0 / (np.ptp(pts, axis=0).max() + 1e-9)
        pts = smooth_polyline(resample_polyline(pts, max(40, len(pts))), 7)
        mesh = tube_mesh(pts, radius=radius_mm, sides=20)

        if loop:
            # a torus ring sitting just above the topmost point of the sculpture
            ring = trimesh.creation.torus(
                major_radius=radius_mm * 2.0, minor_radius=radius_mm * 0.7)
            top = mesh.vertices[mesh.vertices[:, 1].argmax()]
            ring.apply_translation([top[0], top[1] + radius_mm * 2.0, top[2]])
            # overlapping solids; slicers merge them fine for printing
            mesh = trimesh.util.concatenate([mesh, ring])

        mesh.export(out_path)
        return mesh.is_watertight

    # ---- per-frame processing (camera thread) ----
    def process(self, frame):
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # emotion -> mood colour (optional, runs alongside hand tracking)
        if self._face is not None:
            fres = self._face.process(rgb)
            if fres.multi_face_landmarks:
                lm = fres.multi_face_landmarks[0].landmark
                pts = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
                label, _ = self.emotion.push(pts)
                if label:
                    self.emotion_label = label
                    with self._lock:
                        self.color = MOOD_COLORS.get(label, self.color)

        res = self._hands.process(rgb)

        if res.multi_hand_landmarks:
            lm = res.multi_hand_landmarks[0].landmark
            idx, thb = lm[8], lm[4]
            span = float(np.hypot(lm[0].x - lm[9].x, lm[0].y - lm[9].y)) + 1e-6
            ratio = float(np.hypot(idx.x - thb.x, idx.y - thb.y)) / span

            if self.mode == "calibrating":
                self._calibrate_step(ratio, span)
            elif self.mode == "sculpting":
                if self.gesture_mode:
                    self._gesture_step(lm)
                else:
                    self._sculpt_step(lm, ratio, w, h, frame)

            px, py = int(idx.x * w), int(idx.y * h)
            dot = BGR["pink_dk"] if self.drawing else BGR["sage"]
            cv2.circle(frame, (px, py), 9, dot, -1)
            cv2.circle(frame, (px, py), 9, (255, 255, 255), 1)

        self._draw_overlay(frame, w, h)
        return frame

    def _calibrate_step(self, ratio, span):
        self._calib_ratios.append(ratio)
        self._calib_spans.append(span)
        n = len(self._calib_ratios)
        self.status = f"Scanning your hand…  {min(n, CALIB_FRAMES)}/{CALIB_FRAMES}"
        if n >= CALIB_FRAMES:
            r = np.array(self._calib_ratios)
            lo, hi = np.percentile(r, 15), np.percentile(r, 70)
            self.pinch_on = float(lo + 0.15 * (hi - lo))
            self.pinch_off = float(self.pinch_on + 0.18)
            self.hand_scale = float(np.median(self._calib_spans))
            self.mode = "sculpting"
            self.status = "Calibrated! Pinch to draw ✏️"

    def _sculpt_step(self, lm, ratio, w, h, frame):
        if not self.drawing and ratio < self.pinch_on:
            self.drawing = True
        elif self.drawing and ratio > self.pinch_off:
            self.drawing = False
        if self.drawing:
            idx = lm[8]
            with self._lock:
                self.path.append((idx.x, idx.y, idx.z))

    def _gesture_step(self, lm):
        """Drive the app from the trained gesture classifier."""
        pts = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
        label, conf = self.recognizer.push(pts)
        if label is None:
            return
        self.gesture_label = label
        self.status = f"gesture: {label} ({conf:.0%})"
        if label == "draw":
            self.drawing = True
            idx = lm[8]
            with self._lock:
                self.path.append((idx.x, idx.y, idx.z))
        elif label == "stop":
            self.drawing = False
        elif label == "clear":
            self.drawing = False
            with self._lock:
                self.path = []
        elif label == "undo":
            self.drawing = False
            with self._lock:
                self.path = self.path[:-20]   # drop the last ~20 points
        elif label == "save":
            self.drawing = False
            self.save_requested = True       # app picks this up and exports
        else:                                 # idle
            self.drawing = False

    def _draw_overlay(self, frame, w, h):
        # soft translucent top bar for status text
        bar = frame.copy()
        cv2.rectangle(bar, (0, 0), (w, 40), BGR["bg"], -1)
        cv2.addWeighted(bar, 0.55, frame, 0.45, 0, frame)
        cv2.putText(frame, self.status, (14, 26),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, BGR["text"], 1, cv2.LINE_AA)

        # current stroke, drawn in the chosen colour
        with self._lock:
            path = list(self.path)
            color = self.color
            caption = self.caption
        if len(path) > 1:
            p2 = np.array([[int(p[0] * w), int(p[1] * h)] for p in path])
            for i in range(1, len(p2)):
                cv2.line(frame, tuple(p2[i - 1]), tuple(p2[i]), color, 3,
                         cv2.LINE_AA)

        # swatch + caption (what the user said they're making)
        cv2.rectangle(frame, (14, h - 34), (40, h - 12), color, -1)
        cv2.rectangle(frame, (14, h - 34), (40, h - 12), (255, 255, 255), 1)
        if caption:
            cv2.putText(frame, caption[:48], (50, h - 16),
                        cv2.FONT_HERSHEY_DUPLEX, 0.55, BGR["text"], 1, cv2.LINE_AA)
        if self.emotion_mode:
            cv2.putText(frame, f"mood: {self.emotion_label}", (w - 220, h - 16),
                        cv2.FONT_HERSHEY_DUPLEX, 0.6, color, 1, cv2.LINE_AA)

    def close(self):
        if self._hands is not None:
            self._hands.close()
        if self._face is not None:
            self._face.close()
