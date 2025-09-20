# app.py
import streamlit as st
import json
import pandas as pd
from omr.processor import evaluate_image, evaluate_batch
from omr.report import generate_pdf_for_results
import io
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="OMR Evaluator", layout="wide")
st.title("Automated OMR Evaluator — Hackathon Edition 🚀")
st.markdown("Upload OMR sheets (images). Provide answer key and the sheet layout (number of questions & choices).")

# Sidebar inputs
st.sidebar.header("Settings")
num_questions = st.sidebar.number_input("Number of questions", min_value=1, value=10)
choices_per_q = st.sidebar.number_input("Choices per question (A,B,C...)", min_value=2, max_value=6, value=4)
fill_threshold = st.sidebar.slider("Fill threshold (0-1)", 0.1, 0.9, 0.5, 0.01)
min_area = st.sidebar.number_input("Min bubble area px", value=700)
max_area = st.sidebar.number_input("Max bubble area px", value=4000)

st.sidebar.markdown("---")
st.sidebar.markdown("**Answer key input**")
key_text = st.sidebar.text_area("Paste answer key like: 1:A,2:C,3:B", height=90)
key_file = st.sidebar.file_uploader("Or upload JSON (e.g. {\"1\":\"A\",\"2\":\"B\"})", type=["json"])

def parse_answer_key(text, uploaded):
    ak = {}
    if uploaded is not None:
        try:
            ak = json.load(uploaded)
            return {str(k): str(v).upper() for k,v in ak.items()}
        except Exception as e:
            st.sidebar.error("Invalid JSON answer key.")
    if text:
        # parse pairs
        parts = text.replace(" ", "").split(",")
        for p in parts:
            if ":" in p:
                q,a = p.split(":")
                try:
                    ak[str(int(q))] = str(a).upper()
                except:
                    ak[q] = str(a).upper()
    return ak

answer_key = parse_answer_key(key_text, key_file)

st.sidebar.markdown("---")
st.sidebar.markdown("**Upload OMR images**")
uploaded_files = st.file_uploader("Upload one or many images", accept_multiple_files=True, type=['png','jpg','jpeg'])

if st.sidebar.button("Run Evaluation"):
    if not uploaded_files:
        st.error("Upload at least one OMR image")
    elif len(answer_key) == 0:
        st.error("Provide an answer key (sidebar)")
    else:
        st.info(f"Running evaluation on {len(uploaded_files)} image(s)...")
        results = []
        for uf in uploaded_files:
            try:
                score, total, per_q, annotated, thresh = evaluate_image(
                    uf,
                    num_questions=num_questions,
                    choices_per_question=choices_per_q,
                    answer_key=answer_key,
                    fill_threshold=fill_threshold,
                    min_area=min_area,
                    max_area=max_area,
                    debug=False
                )
                results.append({"file": uf.name, "score": score, "total": total, "per_q": per_q, "annotated": annotated})
            except Exception as e:
                st.error(f"Error processing {uf.name}: {e}")

        # Show table
        df = pd.DataFrame([{"file": r['file'], "score": r['score'], "total": r['total']} for r in results])
        st.subheader("Results summary")
        st.dataframe(df)

        # Allow CSV download
        csv_buf = io.BytesIO()
        df.to_csv(csv_buf, index=False)
        csv_buf.seek(0)
        st.download_button("Download CSV", data=csv_buf, file_name="omr_results.csv", mime="text/csv")

        # Generate PDF report bytes
        pdf_bytes = generate_pdf_for_results(results)
        st.download_button("Download PDF report", data=pdf_bytes, file_name="omr_report.pdf", mime="application/pdf")

        # Show annotated images one by one
        st.subheader("Annotated images")
        for r in results:
            st.markdown(f"**{r['file']} — Score: {r['score']}/{r['total']}**")
            # convert BGR -> RGB
            img = cv2.cvtColor(r['annotated'], cv2.COLOR_BGR2RGB)
            st.image(img, use_column_width=True)

st.markdown("---")
st.markdown("**Calibration tips:** If bubbles are not detected correctly, increase/decrease `min bubble area`, `max bubble area`, or `fill threshold`. If sheets are tilted heavily, ensure your sheets include a clear border (or crop them manually).")

st.markdown("Made for quick hackathon demos. For production: add template config, validation, robust clustering and more tests.")
