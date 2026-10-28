# Example outputs (facial-expression model)

Produced by running the pipeline on **synthetic** face data
(`make_synthetic_data.py` → `train.py` → `evaluate.py`). Even synthetic faces
overlap a little, so you get a realistic ~93% rather than a fake 100% — the
confusion matrix shows the model mixing up *happy* and *neutral*, which is a very
real failure mode.

Your numbers on real webcam data (your own expressions) will differ and are the
ones to put in your report.
