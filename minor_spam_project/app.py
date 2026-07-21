import streamlit as st
import joblib
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import os

nltk_data_path = os.path.join(os.path.expanduser("~"), "nltk_data")
nltk.data.path.append(nltk_data_path)
nltk.download('stopwords', download_dir=nltk_data_path)
nltk.download('punkt', download_dir=nltk_data_path)
nltk.download('punkt_tab', download_dir=nltk_data_path)

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "spam_model.pkl"))
vectorizer = joblib.load(os.path.join(BASE_DIR, "tfidf_vectorizer.pkl"))
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words]
    return " ".join(tokens)

st.set_page_config(page_title="Spam Email Classifier", page_icon="📧")
st.title("📧 Spam Email Classifier")
st.write("Enter a message below and I'll tell you if it's Spam or Ham (not spam).")

user_input = st.text_area("Type or paste a message here:")

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter a message first.")
    else:
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        probability = model.predict_proba(vectorized)[0]
        if prediction == 1:
            st.error(f"🚨 This looks like SPAM ({probability[1]*100:.1f}% confidence)")
        else:
            st.success(f"✅ This looks like HAM — not spam ({probability[0]*100:.1f}% confidence)")
