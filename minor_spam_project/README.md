# 📧 Spam Email Classifier

**Minor Capstone Project — Machine Learning with Python Internship**
**Author:** Reddy Uday Kiran Reddy

🔗 **Live Demo:** https://spamemailclassifiermp.streamlit.app/

## 📌 Objective

Spam messages are one of the most common real-world problems machine learning is used to solve — email
providers, SMS carriers, and messaging apps all rely on automated classifiers to filter unwanted content
before it ever reaches a user. This project builds that same kind of system from scratch: a model that
reads a message's text and predicts whether it is **Spam** (unwanted, often malicious or promotional) or
**Ham** (a normal, legitimate message).

Beyond just building a working classifier, this project's goal was to go through the **complete, realistic
ML workflow** a data scientist would actually follow: collecting and inspecting real data, cleaning noisy
text, converting it into a format a model can learn from, comparing multiple algorithms fairly, tuning the
best one, evaluating it honestly, and finally deploying it as a live, usable web application — not just a
notebook that sits unused.

-## 📊 Dataset

- **Name:** SMS Spam Collection Dataset
- **Source:** [UCI Machine Learning Repository / Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- **Total messages:** 5,572
  - Ham (legitimate): 4,825 (~87%)
  - Spam: 747 (~13%)
- **Format:** Tab-separated file with two columns — `label` (ham/spam) and `message` (raw text)

### Why this dataset matters for evaluation

The dataset is **imbalanced** — there are roughly 6.5 times more ham messages than spam ones. This is
actually realistic: in real life, most messages people receive are legitimate, and spam is the minority.
But it creates a trap for evaluation. A "lazy" model that always predicts "ham," no matter what the message
says, would still score around 87% accuracy — while being completely useless, since it would never catch a
single spam message. This is exactly why, throughout this project, **accuracy alone was never trusted as
the deciding metric** — precision, recall, and F1-score (explained in the Results section) were used to get
an honest picture of how the model actually performs on the minority "spam" class, which is the class that
actually matters most to catch correctly.

---

## 🛠️ Tools & Technologies

| Category | Tools Used | Why this tool |
|---|---|---|
| Language | Python 3.11 | Industry-standard language for ML, with a mature ecosystem of libraries |
| Data handling | Pandas, NumPy | Loading, exploring, and manipulating the dataset efficiently |
| Text preprocessing | NLTK | Provides ready-made stopword lists and reliable tokenization |
| Feature extraction | scikit-learn's TfidfVectorizer | Converts raw text into meaningful numeric features |
| Modeling | scikit-learn (Naive Bayes, Logistic Regression, SVM) | Well-established, well-documented algorithms suited to text classification |
| Model tuning | GridSearchCV (5-fold cross-validation) | Systematically finds the best hyperparameters instead of guessing |
| Visualization | Matplotlib, Seaborn | Understanding class distribution and comparing model performance visually |
| Frontend | Streamlit | Turns a Python script into an interactive web app without needing HTML/CSS/JS |
| Deployment | Streamlit Community Cloud | Free, permanent hosting directly from a GitHub repository |
| Development environment | Google Colab, Jupyter Notebook | Cell-based execution makes it easy to inspect data and results at every step |
| Model persistence | joblib | Saves a trained model's exact learned state so it doesn't need retraining every time |

---

## 📁 Project Structure

spam-email-classifier/
├── data/
│ └── sms.tsv # Raw labeled dataset
├── notebooks/
│ └── spam_classifier.ipynb # Full documented notebook (main deliverable)
├── app.py # Streamlit frontend application
├── spam_model.pkl # Trained, tuned SVM model
├── tfidf_vectorizer.pkl # Fitted TF-IDF vectorizer
├── requirements.txt # Python dependencies
├── runtime.txt # Python version for deployment
└── README.md


---

## 🔬 Methodology

### 1. Data Collection

The SMS Spam Collection dataset was loaded directly into a Pandas DataFrame. Before doing anything else,
the data was inspected — checking its shape, viewing sample rows, and counting how many messages fell into
each class. This "look before you model" step matters because it's what revealed the class imbalance
mentioned above, which shaped every evaluation decision made later in the project.

### 2. Text Preprocessing

Raw text is messy and inconsistent from a model's point of view — the same underlying word can appear in
many different forms ("FREE", "free", "Free!!!"), and filler words add noise without adding meaning. Each
message was cleaned through the following steps, in this specific order:

1. **Lowercasing** — ensures "FREE" and "free" are treated as the exact same word, rather than being seen
   as two unrelated tokens.
2. **Punctuation removal** — strips out characters like `!`, `?`, `,` since punctuation alone rarely helps
   distinguish spam from ham, and leaving it in would fragment words unnecessarily.
3. **Tokenization** — splits each cleaned sentence into individual words (e.g., `"win free prize"` becomes
   `["win", "free", "prize"]`), which is the format needed before removing stopwords or vectorizing.
4. **Stopword removal** — filters out extremely common English words ("the", "is", "a", "and", etc.) that
   appear in almost every message regardless of whether it's spam or ham, and therefore add noise rather
   than useful signal.

The result is a cleaned version of every message that keeps only the words most likely to carry real
meaning for classification.

### 3. Feature Extraction — TF-IDF

Machine learning models work with numbers, not raw words, so the cleaned text needed to be converted into
a numeric format. **TF-IDF (Term Frequency–Inverse Document Frequency)** was chosen over a simpler approach
(like just counting word occurrences) because it weighs words more intelligently:

- **Term Frequency** rewards words that appear often *within a specific message* — if "prize" appears
  multiple times in one message, that's a strong signal for that message specifically.
- **Inverse Document Frequency** penalizes words that appear in *almost every* message across the whole
  dataset — since such words don't help tell messages apart, their weight is reduced.

The combined effect is that distinctive, spam-indicating words (like "winner", "urgent", "claim", "free")
end up with high TF-IDF scores in spam messages, while generic words that appear everywhere get
automatically down-weighted — without needing to manually decide which words matter.

### 4. Train/Test Split

The dataset was split into 80% training data and 20% testing data. The test set is never shown to the
model during training — it exists purely to check, honestly, how the model performs on messages it has
never encountered before, simulating real-world use. The split used `stratify=y`, which guarantees the
80/20 split preserves the same ~87/13 ham/spam ratio in *both* the training and test sets — without this,
a random split could accidentally leave very few spam examples in the test set, making evaluation unreliable.

### 5. Model Training & Comparison

Rather than assuming one algorithm would work best, three different classic algorithms — all reasonable
choices for text classification — were trained on identical data and compared fairly:

- **Multinomial Naive Bayes** — a probability-based classifier that's historically one of the most common
  choices for spam filtering specifically, since it works well with word-frequency-style data and trains
  very quickly even on limited data.
- **Logistic Regression** — a simple, interpretable linear classifier that serves as a strong general
  baseline for binary (spam/ham) classification problems.
- **Support Vector Machine (linear kernel)** — chosen because it doesn't just find *a* boundary between
  spam and ham, it finds the boundary that maximizes the margin between the two classes, which often makes
  it more robust to slightly unusual or borderline messages than a plain linear classifier.

### 6. Hyperparameter Tuning

Rather than using SVM's default settings, `GridSearchCV` was used to search across several values of `C`
(0.1, 1, 10, 100) — the parameter that controls how strictly the model tries to fit the training data. This
used 5-fold cross-validation, meaning each candidate value of `C` was tested across 5 different train/validation
splits of the training data and averaged, rather than trusting a single lucky/unlucky split. The search was
explicitly optimized for **F1-score** rather than accuracy, since F1-score balances precision and recall
together — which matters more for this imbalanced, spam-detection task than raw accuracy would.

### 7. Evaluation

Each trained model was evaluated on the untouched test set using:
- **Confusion matrix** — to see exactly how many spam messages were correctly caught, how many were missed,
  and how many legitimate messages were incorrectly flagged.
- **Precision** — of everything the model labeled "spam," how much of it was genuinely spam (measures false
  alarm rate).
- **Recall** — of all the actual spam messages in the test set, how many did the model successfully catch
  (measures how much spam slips through).
- **F1-score** — a single combined score balancing precision and recall, useful for comparing models overall.

### 8. Testing on New Samples

Beyond the formal test set, the final tuned model was also tested against a handful of completely new,
hand-written messages — including some deliberately spam-like ("Congratulations! You won a free iPhone...")
and some deliberately normal ("Hey, are we still meeting for lunch tomorrow?"). This step exists to sanity-check
that the model generalizes to genuinely new writing styles, not just the statistical patterns of the original
dataset.

### 9. Deployment

The final trained SVM model and the fitted TF-IDF vectorizer were both saved using `joblib`, which preserves
their exact learned internal state so they never need to be retrained again. These were then loaded inside
a Streamlit application (`app.py`), which provides a simple web interface: a text box where a user pastes a
message, and a "Classify" button that runs the same cleaning → vectorizing → predicting pipeline used during
training, then displays the result with a confidence percentage. This was deployed publicly through Streamlit
Community Cloud, connected directly to this GitHub repository, giving a permanent live link rather than a
temporary local demo.

---

## 📈 Results

| Model | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| Naive Bayes | 0.961 | 1.00 | 0.71 | 0.83 |
| Logistic Regression | 0.950 | 1.00 | 0.62 | 0.77 |
| **SVM (tuned, linear kernel)** | **0.980** | **0.99** | **0.86** | **0.92** |

### Interpreting these results

**Naive Bayes** achieved perfect precision (1.00) — meaning every single message it labeled "spam" really
was spam, with zero false alarms. But its recall (0.71) shows it missed nearly 3 in every 10 actual spam
messages, letting them through as "ham." This is the classic behavior of an overly cautious model: it only
flags something as spam when extremely confident, at the cost of catching less spam overall.

**Logistic Regression** showed a similar pattern to Naive Bayes but with even lower recall (0.62), missing
almost 4 in 10 spam messages — making it the weakest of the three for this specific task.

**SVM**, after tuning, achieved the best balance: very high precision (0.99, almost no false alarms) *and*
noticeably better recall (0.86) than the other two models — meaning it catches significantly more real spam
while still rarely misflagging legitimate messages. This is why SVM was selected as the final model for
deployment: in a real spam filter, both extremes are costly — missing spam is annoying and potentially
dangerous, while flagging real messages as spam risks a user missing something important — and SVM handled
that tradeoff best out of the three.

---

## 💡 Key Insights

- The dataset's class imbalance (87% ham / 13% spam) meant **accuracy alone would have been a misleading
  metric** — it was necessary to look at precision, recall, and F1-score specifically for the spam class to
  understand real performance.
- **Naive Bayes**, despite being a classic go-to algorithm for spam detection historically, underperformed
  SVM here specifically on recall — it was too conservative about labeling messages as spam.
- **TF-IDF** proved effective without any manual feature engineering — distinctive spam vocabulary naturally
  rose to the top through the term-frequency/inverse-document-frequency weighting, without needing to
  hand-pick "spammy" keywords.
- **Deployment surfaced real-world engineering issues** that don't show up while just working in a notebook:
  the exact scikit-learn version used during training (`1.6.1`) had to be pinned in `requirements.txt`, since
  a newer version on the deployment server caused the saved model to fail loading. Similarly, the deployment
  platform's default Python version (3.14) was too new for scikit-learn's pre-built installers, causing
  extremely slow builds — explicitly setting Python 3.11 in the deployment settings fixed this. These weren't
  modeling problems at all, but they were just as important to solve as the machine learning itself.

### Possible future improvements

- Add bigrams/trigrams to TF-IDF (e.g., treating "click here" as one feature rather than two separate words),
  which could capture spam phrasing patterns more precisely than single words alone.
- Try ensemble methods like Random Forest or Gradient Boosting for comparison against the three models tested here.
- Address the class imbalance more directly using oversampling techniques such as SMOTE, to see if recall
  improves further without sacrificing precision.
- Expand training data beyond SMS-length text to full-length emails, which have different structure
  (subject lines, greetings, signatures) that this SMS-trained model has never seen.

---

## 🚀 How to Run Locally

1. **Clone the repository**
```bash
   git clone https://github.com/YOUR_USERNAME/spam-email-classifier.git
   cd spam-email-classifier
```

2. **Install dependencies**
```bash
   pip install -r requirements.txt
```

3. **Run the notebook** (to see the full training process, all outputs, and evaluation)
```bash
   jupyter notebook notebooks/spam_classifier.ipynb
```

4. **Run the Streamlit app locally**
```bash
   streamlit run app.py
```
   This opens the app automatically at `http://localhost:8501`.

---

## 🌐 Live Application

Try the deployed classifier here: **https://spamemailclassifiermp.streamlit.app/**

Simply type or paste any message and click "Classify" to see whether it's predicted as Spam or Ham, along
with the model's confidence percentage.
