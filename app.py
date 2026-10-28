"""
Air Sculptor — draw in the air, export a 3D print. Pastel desktop app with
consent gating, a hand-scan calibration, offline voice colour + assistant, and
local-only session storage.

Run:  python app.py
"""

import argparse
import os
import sys
import tkinter as tk
from tkinter import ttk

import cv2
from PIL import Image, ImageTk

from core import consent, storage, session_end, theme
from core.camera import CameraThread
from core.voice import Listener, Speaker
from core.assistant import Assistant
from faq import FAQ
from sculptor import SculptorProcessor

APP_ID = "air_sculptor"
APP_NAME = "Air Sculptor"
VIDEO_W = 640
QUESTION_CUES = ("?", "how ", "what ", "why ", "buddy", "help", "hey")


def load_recognizer():
    """Try to load the trained gesture model from ml/. Returns a recognizer or
    None. Never raises — if anything is missing the app uses the pinch rule."""
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "ml"))
        from gesture_recognizer import GestureRecognizer
        rec = GestureRecognizer()
        return rec if rec.available else None
    except Exception:
        return None


def load_emotion():
    """Try to load the trained emotion model from ml/face/. Returns a recognizer
    or None; never raises."""
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "ml", "face"))
        from emotion_recognizer import EmotionRecognizer
        e = EmotionRecognizer()
        return e if e.available else None
    except Exception:
        return None


class App:
    def __init__(self, root, store, consent_rec, use_gestures=False,
                 use_emotion=False):
        self.root = root
        self.store = store
        self.mic_on = bool(consent_rec.get("mic"))
        self.session_dir = store.new_session_dir()

        recognizer = load_recognizer() if use_gestures else None
        emotion = load_emotion() if use_emotion else None
        profile = store.load_profile()
        self.proc = SculptorProcessor(profile=profile, recognizer=recognizer,
                                      emotion=emotion)
        self.assistant = Assistant(FAQ)
        self.speaker = Speaker()

        fam, pal = theme.apply(root)
        self.fam, self.pal = fam, pal
        root.title(APP_NAME)

        root.columnconfigure(0, weight=1)         # video column grows
        root.columnconfigure(1, minsize=320, weight=0)
        root.rowconfigure(0, weight=1)
        root.minsize(900, 560)

        # --- left: video (fills its pane; scaled each frame in _tick) ---
        left = ttk.Frame(root, padding=14, style="TFrame")
        left.grid(row=0, column=0, sticky="nsew")
        left.rowconfigure(0, weight=1)
        left.columnconfigure(0, weight=1)
        self.video = tk.Label(left, bg=pal["surface_2"], bd=0)
        self.video.grid(row=0, column=0, sticky="nsew")

        # --- right: controls + chat ---
        self._build_sidebar(root)

        # --- camera ---
        self.cam = CameraThread(self.proc.process)
        self.cam.start()

        # --- voice ---
        self.listener = None
        if self.mic_on:
            self.listener = Listener(self._on_voice_text)
            self.listener.start()

        if self.proc.gesture_mode:
            hello = ("Hi! 🤍 Gesture mode is on — calibrate, then use your "
                     "trained gestures (draw / stop / clear / save).")
        else:
            hello = ("Hi! 🤍 Calibrate first, then pinch to draw. "
                     "Ask me anything here or out loud.")
        self._log("Buddy", hello)
        self.root.protocol("WM_DELETE_WINDOW", self.end_session)
        self._tick()

    # ---------------- UI ----------------
    def _build_sidebar(self, root):
        pal, fam = self.pal, self.fam
        side = ttk.Frame(root, padding=(16, 14), style="TFrame")
        side.grid(row=0, column=1, sticky="nsew")
        side.rowconfigure(6, weight=1)

        ttk.Label(side, text=APP_NAME, style="Title.TLabel").grid(sticky="w")
        self.status = ttk.Label(side, text="Ready", style="Soft.TLabel",
                                wraplength=290)
        self.status.grid(sticky="w", pady=(2, 12))

        row = ttk.Frame(side, style="TFrame"); row.grid(sticky="ew")
        ttk.Button(row, text="Calibrate", style="Lav.TButton",
                   command=self.proc.start_calibration).grid(row=0, column=0, padx=3)
        ttk.Button(row, text="Clear", style="Primary.TButton",
                   command=self.proc.clear).grid(row=0, column=1, padx=3)
        ttk.Button(row, text="Save", style="Sage.TButton",
                   command=self.save).grid(row=0, column=2, padx=3)

        self.keychain = tk.BooleanVar(value=False)
        ttk.Checkbutton(side, text="🔑  Keychain mode (add a loop to print & wear)",
                        variable=self.keychain).grid(sticky="w", pady=(8, 0))

        mic_txt = "🎙️ voice on" if self.mic_on else "🎙️ voice off (mic not allowed)"
        ttk.Label(side, text=mic_txt, style="Soft.TLabel").grid(
            sticky="w", pady=(10, 4))

        ttk.Label(side, text="Ask Buddy", style="Card.TLabel").grid(
            sticky="w", pady=(8, 2))
        self.chat = tk.Text(side, height=12, width=34, wrap="word", bd=0,
                            bg=pal["surface"], fg=pal["text"], font=(fam, 10),
                            padx=10, pady=8, highlightthickness=1,
                            highlightbackground=pal["line"], state="disabled")
        self.chat.grid(sticky="nsew", pady=(0, 6))

        entry_row = ttk.Frame(side, style="TFrame"); entry_row.grid(sticky="ew")
        entry_row.columnconfigure(0, weight=1)
        self.entry = tk.Entry(entry_row, bd=0, bg=pal["surface_2"],
                              fg=pal["text"], font=(fam, 11),
                              highlightthickness=1, highlightbackground=pal["line"])
        self.entry.grid(row=0, column=0, sticky="ew", ipady=6, padx=(0, 6))
        self.entry.bind("<Return>", lambda e: self._send_typed())
        ttk.Button(entry_row, text="Send", style="Primary.TButton",
                   command=self._send_typed).grid(row=0, column=1)

        ttk.Button(side, text="End session", style="Deny.TButton",
                   command=self.end_session).grid(sticky="ew", pady=(12, 0))

    def _tick(self):
        if not self.cam.ok:
            self.status.config(text=self.cam.error or "Camera error")
        frame = self.cam.latest()
        if frame is not None:
            h, w = frame.shape[:2]
            # scale the video to the current size of its pane, keeping aspect
            avail_w = max(self.video.winfo_width(), 320)
            avail_h = max(self.video.winfo_height(), 240)
            scale = min(avail_w / w, avail_h / h)
            disp = cv2.resize(frame, (max(1, int(w * scale)),
                                      max(1, int(h * scale))))
            img = Image.fromarray(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB))
            self._imgtk = ImageTk.PhotoImage(img)
            self.video.config(image=self._imgtk)
            self.status.config(text=self.proc.status)
        # honor the "save" gesture (debounced: fires once per trigger)
        if getattr(self.proc, "save_requested", False):
            self.proc.save_requested = False
            self.save()
        self.root.after(33, self._tick)

    # ---------------- chat ----------------
    def _log(self, sender, text):
        self.chat.config(state="normal")
        tag = "me" if sender == "You" else "bot"
        self.chat.insert("end", f"{sender}: ", (tag,))
        self.chat.insert("end", text + "\n\n")
        self.chat.tag_config("me", foreground=self.pal["pink_dk"],
                             font=(self.fam, 10, "bold"))
        self.chat.tag_config("bot", foreground=self.pal["lavender"],
                             font=(self.fam, 10, "bold"))
        self.chat.see("end")
        self.chat.config(state="disabled")

    def _send_typed(self):
        text = self.entry.get().strip()
        if not text:
            return
        self.entry.delete(0, "end")
        self._ask(text, speak=False)

    def _ask(self, text, speak):
        self._log("You", text)
        reply = self.assistant.reply(text)
        self._log("Buddy", reply)
        if speak:
            self.speaker.say(reply)

    # ---------------- voice ----------------
    def _on_voice_text(self, text):
        # called from the listener thread -> hop to the UI thread
        self.root.after(0, lambda: self._handle_voice(text))

    def _handle_voice(self, text):
        # narration always updates the pen colour + on-screen caption
        self.proc.set_color_from_text(text)
        # if it sounds like a question / wake word, treat it as a chat turn
        if any(cue in text.lower() for cue in QUESTION_CUES):
            self._ask(text, speak=True)

    # ---------------- actions ----------------
    def save(self):
        if not self.proc.has_stroke():
            self.status.config(text="Draw a longer stroke before saving")
            return
        loop = bool(self.keychain.get())
        out = os.path.join(self.session_dir,
                           "air_keychain.stl" if loop else "air_sculpture.stl")
        watertight = self.proc.save_stl(out, loop=loop)
        self.status.config(text=f"Saved ✓  (watertight={watertight})")
        self._log("Buddy", "Saved your "
                  + ("keychain (with a loop to thread)! 🔑" if loop
                     else "sculpture to this session. 🤍"))

    def end_session(self):
        self.cam.stop()
        if self.listener:
            self.listener.stop()
        self.speaker.stop()
        self.proc.close()
        self.root.destroy()

        choice = session_end.ask(APP_NAME)
        summary = self.store.apply_choice(
            choice, profile=self.proc.profile(), session_dir=self.session_dir)
        print(summary)


def main():
    ap = argparse.ArgumentParser(description=APP_NAME)
    ap.add_argument("--gestures", action="store_true",
                    help="use the trained ML gesture recognizer (ml/) instead "
                         "of the pinch rule; falls back to pinch if no model")
    ap.add_argument("--emotion", action="store_true",
                    help="use the trained emotion model (ml/face/) to colour the "
                         "pen by your mood; ignored if no model is trained")
    args = ap.parse_args()

    store = storage.Store(APP_ID)
    rec = consent.ensure_consent(store, APP_NAME, uses_mic=True)
    if not rec:
        print("No permission given — Air Sculptor will not start.")
        sys.exit(0)

    root = tk.Tk()
    App(root, store, rec, use_gestures=args.gestures, use_emotion=args.emotion)
    root.mainloop()


if __name__ == "__main__":
    main()
