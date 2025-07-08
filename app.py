import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import model_from_json
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
import re
import nltk

nltk.download('stopwords')
nltk.download('wordnet')

with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

def load_model():
    with open('question_classification_model.json', 'r') as json_file:
        loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)
    model.load_weights('question_classification_weights.weights.h5')
    return model

model = load_model()

def clean_text(text):
    stop = set(stopwords.words('english'))
    for wh in ['who', 'what', 'when', 'why', 'how', 'which', 'where', 'whom']:
        stop.discard(wh)

    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = text.lower()
    words = text.split()
    words = [word for word in words if word not in stop]
    lem = WordNetLemmatizer()
    words = [lem.lemmatize(word, pos='v') for word in words]
    return ' '.join(words)

st.title("Question Classification")
st.write("Masukkan pertanyaan anda dibawah")

user_input = st.text_area("Pertanyaan : ", "")

if st.button("Klasifikasi"):
    if user_input.strip() == "":
        st.warning("Please Enter Text First.")
    else:
        cleaned = clean_text(user_input)
        tfidf_input = vectorizer.transform([cleaned])
        prediction = model.predict(tfidf_input)
        predicted_class = np.argmax(prediction, axis=1)
        label = label_encoder.inverse_transform(predicted_class)[0]
        st.success(f"Jenis Pertanyaan: **{label}**")
