# OMR Evaluator — Hackathon Edition

Quick start:
1. pip install -r requirements.txt
2. streamlit run app.py
3. Open the local URL shown by streamlit (usually http://localhost:8501)

Provide:
- Number of questions
- Choices per question (e.g., 4 for A-D)
- Paste / upload answer key (JSON or "1:A,2:C,...")
- Upload OMR image(s)
- Click "Run Evaluation"

Outputs: Annotated images, CSV, PDF report.

Calibration: tune fill threshold and bubble area if detection misses bubbles from phone photos.

Good luck!
