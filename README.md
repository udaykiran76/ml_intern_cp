# ml_intern_cp
# Machine Learning with Python — Internship Capstone Projects

**Author:** Reddy Uday Kiran Reddy

This repository contains both capstone projects completed as part of the Machine Learning with Python
internship — a minor project and a major project, each demonstrating a complete, real-world ML workflow
from raw data through to a working, testable application.

---

## 📁 Projects

### 1. Minor Project — Spam Email Classifier

A text classification system that detects spam vs. legitimate (ham) messages, using TF-IDF feature
extraction and a tuned SVM classifier, achieving ~98% accuracy with strong precision/recall balance.

**Covers:** text preprocessing (NLTK), TF-IDF vectorization, model comparison (Naive Bayes, Logistic
Regression, SVM), hyperparameter tuning (GridSearchCV), and full evaluation (confusion matrix,
precision/recall/F1).

📁 [View project folder](./minor_spam_project)
🔗 **Live demo:** https://spamemailclassifiermp.streamlit.app/

---

### 2. Major Project — Face Recognition Attendance System

A real-time attendance system that detects faces, converts them into facial embeddings using a pretrained
Dlib model, matches them against a registered database of known individuals, and logs attendance with a
timestamp — including a live registration flow and a dashboard for reviewing records.

**Covers:** face detection (Haar Cascade), facial embedding extraction (`face_recognition`/Dlib),
classifier comparison (KNN vs SVM), live webcam-based registration and recognition, and a full Streamlit
dashboard (registration, attendance-taking, and record viewing).

📁 [View project folder](./major_frs_project)
⚠️ **No live demo:** this project is submitted as complete source code, runnable locally. See the
[project README](./major_frs_project/README.md#-deployment-note) for details on why, and full local
run instructions.

---

## 🛠️ Common Tools Across Both Projects

Python · Pandas · NumPy · scikit-learn · Jupyter Notebook / Google Colab · Streamlit

Each project folder contains its own detailed README covering objective, dataset, methodology, results,
key insights, and instructions to run it locally.
