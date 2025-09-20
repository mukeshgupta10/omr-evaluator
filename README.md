# 📝 Automated OMR Evaluator – Hackathon Project 🚀

An automated **OMR (Optical Mark Recognition) evaluation system** built with **Python + OpenCV + Streamlit**.  

Upload scanned/phone-captured answer sheets, provide an answer key, and get instant results with:
- ✅ Scores
- 🖼️ Annotated answer sheets
- 📊 CSV summaries
- 📄 PDF reports

---

## ✨ Features
- Upload **single or multiple OMR sheets**
- Automatic detection of filled bubbles
- Evaluation against provided answer key
- Downloadable CSV + PDF reports
- Adjustable settings for bubble size & fill threshold

---

## 🚀 Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py

Open the local URL shown by Streamlit (usually http://localhost:8501
)

Provide:

Number of questions

Choices per question (e.g., 4 for A–D)

Paste / upload answer key (JSON or "1:A,2:C,...")

Upload OMR image(s)

Click Run Evaluation

Outputs:

✅ Annotated images

📊 CSV report

📄 PDF report

⚙️ Calibration: Tune fill threshold and bubble area in sidebar if detection misses bubbles from phone photos.

📸 Demo Screenshots
Upload OMR sheets

Results summary

Annotated OMR

PDF Report

📦 Tech Stack

Python 3.10+

OpenCV
 – Image Processing

Streamlit
 – Web UI

NumPy
 – Math

Pandas
 – CSV Export

ReportLab
 – PDF Generation
  
All uploaded in demo_screenshots 

📂 Project Structure
omr-evaluator/
├─ app.py              # Streamlit app (UI)
├─ omr/
│  ├─ processor.py     # Core OMR detection & evaluation
│  └─ report.py        # PDF report generator
├─ requirements.txt
├─ README.md
└─ demo_screenshots/   # Screenshots for demo (add your images here)

📜 License

MIT License – Free to use and modify.

🤝 Team & Credits

Built with ❤️ at Code4Edtech Challenge by Innomatics Research lab by:

Mukesh Gupta (GitHub : https://github.com/mukeshgupta10 )
Ashutosh Kumar Jha (Linkedin : https://www.linkedin.com/in/ashutosh-kumar-819887331?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app  )



---

