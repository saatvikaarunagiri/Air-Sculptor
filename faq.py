"""Knowledge base for the in-app assistant (Air Sculptor)."""

FAQ = [
    (["draw", "drawing", "pinch", "start"],
     "Pinch your thumb and index finger together and move your hand — that "
     "draws. Separate them to stop. Calibrate first so it fits your hand."),
    (["calibrate", "calibration", "scan", "setup"],
     "Click 'Calibrate', then show your hand to the camera and pinch slowly a "
     "few times for ~6 seconds. It learns your hand so tracking feels natural."),
    (["save", "saving", "export", "stl", "print"],
     "Click 'Save' (or press S). It writes a 3D-printable .stl into your saved "
     "session folder. Open it in any slicer to print."),
    (["colour", "color", "green", "voice", "say", "speak", "narrate"],
     "Turn on the mic and just say what you're making, e.g. 'the leaves should "
     "be green'. I'll set the pen colour and show your words on screen."),
    (["clear", "undo", "restart", "delete stroke"],
     "Click 'Clear' to erase the current stroke and start over."),
    (["data", "privacy", "stored", "store", "saved", "upload", "private"],
     "Everything runs on your computer and nothing is uploaded. At the end you "
     "choose what to keep: this session, your identity, both, or nothing."),
    (["identity", "profile", "remember"],
     "Your 'identity' is just your hand-calibration profile, kept locally so "
     "next time is faster. You can delete it anytime from the end screen."),
    (["depth", "flat", "3d", "z"],
     "Depth from one camera is approximate, so sculptures look a bit flat in Z. "
     "You can raise the depth gain in the code, or add a second camera."),
    (["help", "stuck", "how", "what can"],
     "I can explain drawing, calibration, saving, voice colours, and privacy. "
     "What are you trying to do?"),
    (["quit", "exit", "close", "end"],
     "Click 'End session' (or close the window). You'll be asked what to keep "
     "before anything is saved."),
]
