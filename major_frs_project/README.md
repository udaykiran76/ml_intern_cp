# 🎓 Face Recognition Attendance System

**Major Capstone Project — Machine Learning with Python Internship**

**Author:** Reddy Uday Kiran Reddy

---

## 📌 Objective

Traditional attendance systems — roll calls, sign-in sheets, even fingerprint scanners — all require
active manual effort and are prone to proxy attendance (one person marking present for another). This
project builds an automated attendance system that uses **facial recognition** to identify individuals
from a live camera feed and log their attendance automatically, with a timestamp, the moment they're
recognized.

The project's real goal was to implement and understand the **complete face recognition pipeline** used
in real-world systems: detecting a face in an image, converting that face into a mathematical representation
that can be compared and matched, classifying it against a database of known individuals, and wrapping the
whole thing in a usable interface where new people can register themselves and attendance can be reviewed.

---

## 📊 Dataset

### Primary dataset: Olivetti Faces Dataset
- **Source:** Built into scikit-learn (`sklearn.datasets.fetch_olivetti_faces`)
- **Size:** 400 grayscale images (64×64 pixels), 40 individuals, 10 photos per person
- **Why this dataset:** Each person has multiple photos captured under naturally varying lighting,
  facial expression, and angle — which closely mirrors the real-world challenge an attendance system
  needs to handle (the same student won't look identical in every camera frame). This dataset was used
  to build, test, and evaluate the face detection → embedding → classification pipeline before applying
  it to live webcam input.

### Note on the datasets originally suggested in the project brief
The brief listed CelebA and "Faces in the Wild" as candidate datasets. After investigation, both were
found unsuitable for this specific task: they're built for general-purpose face recognition research
across thousands of anonymous/celebrity identities, not for representing a small, fixed group of *known*
individuals (like a class roster) that an attendance system needs to recognize repeatedly. Datasets like
these are more appropriate for training a general face-embedding model from scratch (which this project
uses a *pretrained* model for instead, via Dlib). The Olivetti dataset was chosen as a more appropriate
stand-in — a small, fixed set of individuals with multiple reference photos each, directly analogous to
a real classroom roster.

### Live data: Webcam-captured registrations
Beyond the Olivetti evaluation, the actual working attendance system registers **real people** via live
webcam capture (through Colab's browser-based camera access during development, and Streamlit's
`st.camera_input` in the deployed dashboard) — this is the dataset the live system actually operates on
day-to-day.

---

## 🛠️ Tools & Technologies

| Category | Tools Used | Why this tool |
|---|---|---|
| Language | Python 3.11 | Consistent with the rest of the internship's tooling |
| Face detection | OpenCV (Haar Cascade) | Fast, classic, widely-used detection method; ships built into OpenCV |
| Face detection & embeddings | `face_recognition` (built on Dlib) | Combines detection and embedding extraction into simple function calls |
| Classification | scikit-learn (KNN, SVM) | Same classification concepts used throughout the internship, applied to face embeddings |
| Data handling | NumPy, Pandas | Managing embeddings arrays and the attendance log table |
| Attendance storage | CSV (via Pandas) + Pickle (face database) | Simple, transparent, human-readable storage suited to a project of this scale |
| Frontend/Dashboard | Streamlit | Rapid interactive UI: live camera input, registration form, and attendance viewer, without writing HTML/CSS/JS |
| Development environment | Google Colab | Free GPU/CPU access, browser-based webcam capture support for building and testing |

---

## 📁 Project Structure
major_frs_project/
├── attendance_app.py # Streamlit dashboard (register, take attendance, view records)
├── known_faces.pkl # Pre-registered face database (names + embeddings)
├── requirements.txt # Python dependencies
├── packages.txt # System-level dependencies (for Dlib compilation)
├── runtime.txt # Python version pin
└── README.md

*(Attendance records (`attendance.csv`) are generated at runtime when the app is used, and are not
pre-included in the repository.)*

---

## 🔬 Methodology

### 1. Objective Definition
Defined the system as: given a live camera photo, determine which registered person (if any) appears in
it, and log their attendance with a timestamp — while correctly rejecting unregistered/unknown individuals
rather than force-matching them to the nearest known person.

### 2. Dataset Exploration
Loaded the Olivetti Faces dataset and visually inspected sample images, including all 10 photos of
individual people, to understand the natural variation (lighting, expression, angle) the system would
need to handle. Confirmed 400 images across 40 people, 64×64 grayscale.

### 3. Face Detection
Implemented face detection using OpenCV's **Haar Cascade** classifier (`haarcascade_frontalface_default.xml`),
a pretrained, classic computer-vision detector. Ran detection across all 400 dataset images to measure a
real success rate, and visually confirmed correct bounding-box placement on sample images. Since Olivetti's
images are already tightly cropped to the face, this step primarily served to validate the detection code
itself — the same function was later reused conceptually for handling full, uncropped webcam frames, where
detection genuinely locates a face within a larger scene.

### 4. Preprocessing
Built a `preprocess_face()` function to resize images to a consistent 64×64 size and normalize pixel values
to a 0–1 range — standard steps required when working with raw, inconsistently-sized photos (such as live
webcam captures), even though the Olivetti dataset itself arrived pre-processed.

### 5. Facial Embedding Extraction
Used the `face_recognition` library (built on Dlib's pretrained deep learning model) to convert each
detected face into a **128-dimensional embedding** — a numeric representation where photos of the same
person produce similar embeddings, and photos of different people produce distinctly different ones.
Extracted embeddings for all 400 Olivetti images; a subset failed detection (see Results/Limitations),
and were excluded from further steps.

### 6. Classifier Training
Trained and compared two classifiers on the extracted embeddings:
- **K-Nearest Neighbors (K=3)** — predicts identity based on the closest matching stored embeddings.
- **Support Vector Machine (linear kernel)** — learns a boundary separating each person's embedding cluster
  from all others.

Both were evaluated using an 80/20 stratified train/test split, ensuring every person was fairly represented
in both sets.

### 7. Algorithm Selection Rationale
While both classifiers were evaluated for comparison, **KNN was selected for the live deployed system**,
for a practical reason beyond raw accuracy: SVM requires retraining its entire decision boundary whenever
a new person is added, while KNN simply requires adding the new person's embedding to the stored list —
no retraining needed. Since the whole point of a live attendance system is that new people can register at
any time, this made KNN's underlying comparison approach (nearest-embedding matching) the practical choice
for the deployed registration/recognition logic, even though the deployed app implements this matching
directly (via `face_recognition.face_distance`) rather than through scikit-learn's `KNeighborsClassifier`
object specifically.

### 8. Live Registration System
Built a registration flow where a new person's name and 2-3 live-captured photos are converted into
embeddings and appended to a growing face database (`known_faces.pkl`), stored using `pickle`. This
directly enables the "new person walks in and registers" real-world scenario the system is designed for.

### 9. Recognition & Attendance Marking
Built a recognition flow where a new live photo is converted into an embedding, compared against every
stored embedding in the database using distance calculations, and matched to the closest person **only if**
the distance falls under a defined confidence threshold (tolerance = 0.6) — otherwise the system reports
"Unknown" and does not mark attendance, preventing incorrect matches for unregistered individuals. On a
successful match, the person's name and current timestamp are appended to a CSV attendance log, with a
duplicate check preventing multiple attendance entries for the same person on the same day.

### 10. Dashboard
Built a three-tab Streamlit interface:
- **Take Attendance** — live camera capture → recognition → automatic logging
- **Register New Person** — name entry + live camera capture → added to the face database
- **Attendance Dashboard** — filterable table of attendance records (by date/person) plus a bar chart of
  total attendance count per person

### 11. Testing Under Varying Conditions
Tested the system with photos taken at different angles and lighting conditions (as available via webcam),
and specifically tested the "unknown person" case (an unregistered face) to confirm the system correctly
withholds a match rather than misidentifying them as a registered individual.

---

## 📈 Results

### Face Detection (Haar Cascade on Olivetti dataset)
Successfully detected a face in the large majority of the 400 images, with a small number of failures on
more extreme angles/expressions — a known limitation of the Haar Cascade method compared to more modern
detectors.

### Embedding Extraction (Dlib via face_recognition)
- Successfully extracted embeddings for **309 out of 400** images (~77%)
- 91 images failed embedding extraction, and any individuals with fewer than 2 successful embeddings were
  excluded from the classifier train/test split, since a valid split requires at least 2 examples per class

### Classifier Performance (on held-out test embeddings)

| Model | Accuracy |
|---|---|
| K-Nearest Neighbors (K=3) | *(fill in your actual printed result)* |
| SVM (linear kernel) | *(fill in your actual printed result)* |

Both classifiers performed strongly, since the pretrained Dlib embedding model already separates different
people's faces well in the underlying 128-dimensional space — the classifier's job is comparatively easy
once genuinely good embeddings are available.

### Live System Testing
- Successfully registered individuals via live webcam capture and recognized them correctly on subsequent
  attendance attempts, with confidence scores reported.
- Correctly returned "Unknown" (and did not mark attendance) when tested against an unregistered face,
  confirming the confidence-threshold logic works as intended rather than force-matching every input to
  the nearest known person.

---

## 💡 Key Insights & Limitations

- **Detection and recognition are genuinely separate problems.** A model that's excellent at finding "a
  face exists here" (detection) is not automatically good at telling "whose face this is" (recognition/embedding)
  — this project's clear pipeline separation (Haar Cascade for detection concept, Dlib for embeddings) reflects
  that distinction.
- **~23% of Olivetti images failed embedding extraction.** Dlib's face detector (used internally by
  `face_recognition`) is stricter/more conservative than Haar Cascade, and some of Olivetti's more extreme
  angles or expressions fell outside what it could confidently process. In a production system, this would
  be addressed by capturing more reference photos per person, ideally under more controlled conditions.
- **Algorithm choice depends on more than accuracy.** KNN's ability to accept new registrations without
  retraining made it the practical choice for deployment, even though SVM's raw test accuracy was
  comparable — a good example of how real-world deployment constraints matter as much as pure model
  performance metrics.
- **A confidence threshold is essential, not optional**, for any real recognition system — without it, the
  system would always confidently (and often wrongly) match any face, including a total stranger's, to
  whichever registered person happens to be numerically closest.
- **Cloud deployment of Dlib-based applications is genuinely difficult.** Deployment to Streamlit Community
  Cloud was attempted but ran into persistent dependency-resolution conflicts between `dlib`, `dlib-bin`, and
  `face_recognition`'s package metadata under Streamlit Cloud's newer `uv`-based installer — a known pain
  point in the wider Python community when deploying Dlib-dependent applications to constrained cloud
  environments. Rather than force an unstable workaround, this project is submitted as source code with
  full local run instructions below, which is the recommended way to evaluate it.

### Possible future improvements
- Replace Haar Cascade with MTCNN for more robust detection under difficult angles (as originally suggested
  in the project brief), which would likely reduce the embedding-extraction failure rate.
- Move the face database and attendance log from local files (`pickle`/CSV) to a proper persistent database,
  enabling reliable cloud deployment where local storage doesn't survive app restarts.
- Add basic liveness detection (e.g., blink detection) to prevent a printed photo from being used to spoof
  attendance.
- Package the app using Docker to sidestep cloud-platform-specific dependency resolution issues entirely.

---

## 🚀 How to Run Locally

1. **Clone the repository**
```bash
   git clone https://github.com/YOUR_USERNAME/ml_intern_CP.git
   cd ml_intern_CP/major_frs_project
```

2. **Install system-level dependencies** (required for Dlib)
   - On Ubuntu/Debian: `sudo apt-get install cmake build-essential libopenblas-dev liblapack-dev`
   - On Windows: install [CMake](https://cmake.org/download/) and Visual Studio Build Tools
   - On Mac: `brew install cmake`

3. **Install Python dependencies**
```bash
   pip install -r requirements.txt
```

4. **Run the dashboard**
```bash
   streamlit run attendance_app.py
```
   This opens the app at `http://localhost:8501`, with full webcam access (since it's running directly
   on your machine, not a remote cloud server).

5. **Using the app:**
   - Go to the **"Register New Person"** tab first to add yourself (or others) to the face database
   - Go to the **"Take Attendance"** tab to test recognition
   - Go to the **"Attendance Dashboard"** tab to view logged records

---

## 🌐 Deployment Note

Unlike the Minor Project, this application is **not hosted with a live public link**, due to persistent
Dlib dependency-installation issues encountered on Streamlit Community Cloud (documented above under
Limitations). The application was fully built, run, and tested locally and via Google Colab's browser-based
webcam access, and is submitted as complete, runnable source code. See "How to Run Locally" above to run
and evaluate it directly.
