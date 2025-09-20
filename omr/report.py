# omr/report.py
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from PIL import Image
import io
import cv2
import json
import pandas as pd

def annotated_image_to_png_bytes(bgr_img):
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    buf.seek(0)
    return buf

def generate_pdf_for_results(results_list, out_path=None):
    """
    results_list: list of dicts with keys: file, score, total, per_q, annotated (numpy bgr)
    If out_path is None, returns bytes content; else writes to path and returns None.
    """
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    for res in results_list:
        # header
        c.setFont("Helvetica-Bold", 16)
        title = f"OMR Result - {res.get('file','student')}"
        c.drawString(50, height - 50, title)
        c.setFont("Helvetica", 12)
        c.drawString(50, height - 70, f"Score: {res.get('score')}/{res.get('total')}")

        # annotated image
        img_buf = annotated_image_to_png_bytes(res['annotated'])
        img_reader = ImageReader(img_buf)
        # maintain aspect ratio, fit
        iw, ih = Image.open(img_buf).size
        w = 480
        h = int(ih * (w / float(iw)))
        c.drawImage(img_reader, 50, height - 100 - h, width=w, height=h)

        # small table: first 20 questions
        c.setFont("Helvetica", 10)
        y = height - 120 - h
        c.drawString(50, y, "Q#  Sel  Corr  Result")
        y -= 14
        for pq in res['per_q'][:40]:  # show up to 40 to avoid overflow
            line = f"{pq['question']:>2}    {str(pq['selected'] or '-') :>3}    {str(pq['correct'] or '-') :>4}    {'OK' if pq['is_correct'] else 'WR'}"
            c.drawString(50, y, line)
            y -= 12
            if y < 50:
                c.showPage()
                y = height - 50
        c.showPage()

    c.save()
    buffer.seek(0)
    data = buffer.read()
    if out_path:
        with open(out_path, "wb") as f:
            f.write(data)
        return None
    return data
