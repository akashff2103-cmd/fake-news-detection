import streamlit as st
import joblib
import re
import string
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords

# Load Model & Vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

stop_words = set(stopwords.words('english'))

# Text Preprocessing Function
def preprocess(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

# Streamlit UI
st.set_page_config(page_title="Fake News Detector", layout="centered")

st.title("📰 Fake News Detection App")
st.markdown("### Enter a News Article (Title + Content):")

# Input box
user_input = st.text_area("Paste the news article here", height=250)

# Button to Predict
if st.button("Check News"):
    if user_input.strip() == "":
        st.warning("Please enter some news text!")
    else:
        clean_text = preprocess(user_input)
        text_vector = vectorizer.transform([clean_text])
        prediction = model.predict(text_vector)

        if prediction[0] == 1:
            st.success("✅ This news is **Real**.")
        else:
            st.error("❌ This news is **Fake**.")

# Footer
st.markdown("---")
st.markdown("Created by Dhaval Kanpariya · Powered by Streamlit and ML 🎯")
