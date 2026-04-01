import streamlit as st
import pickle
import re
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ==============================
# CONFIG
# ==============================
MAX_LEN = 200
THRESHOLD = 0.5


# ==============================
# LOAD MODEL & TOKENIZER
# ==============================
@st.cache_resource
def load_resources():
    try:
        model = load_model("cnn_toxic_model.h5")
        with open("tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model/tokenizer: {e}")
        return None, None

model, tokenizer = load_resources()


# ==============================
# TEXT CLEANING
# ==============================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)   # remove URLs
    text = re.sub(r"@\w+", "", text)      # remove mentions
    text = re.sub(r"[^a-zA-Z0-9!? ]", "", text)
    return text


# ==============================
# PREDICTION FUNCTION
# ==============================
def predict_toxicity(text):
    text = clean_text(text)

    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=MAX_LEN)

    prediction = model.predict(padded)
    prob = float(prediction.squeeze())

    label = "Toxic" if prob > THRESHOLD else "Not Toxic"

    return label, prob


# ==============================
# STREAMLIT UI
# ==============================
st.set_page_config(
    page_title="Toxic Comment Detector",
    page_icon="",
    layout="centered"
)

st.title("Toxic Comment Detection")
st.markdown("Detect whether a comment is **toxic or safe** using a CNN model.")

# Sidebar
st.sidebar.header("About")
st.sidebar.write("""
- Model: CNN (Deep Learning)
- Task: Binary Text Classification
- Output: Toxic / Not Toxic
""")

# Input
user_input = st.text_area("Enter your comment:", height=150)

# Example buttons
col1, col2 = st.columns(2)

with col1:
    if st.button("Try Toxic Example"):
        user_input = "You are the worst person ever!"

with col2:
    if st.button("Try Safe Example"):
        user_input = "Hope you have a great day!"

# Prediction
if st.button("Analyze Comment"):

    if model is None or tokenizer is None:
        st.stop()

    if user_input.strip() == "":
        st.warning("Please enter a comment.")
    else:
        label, prob = predict_toxicity(user_input)

        st.subheader("Result")

        # Severity logic
        if prob > 0.8:
            st.error(f"Highly Toxic ({prob:.2f})")
        elif prob > 0.6:
            st.warning(f"Moderately Toxic ({prob:.2f})")
        elif prob > THRESHOLD:
            st.info(f"Slightly Toxic ({prob:.2f})")
        else:
            st.success(f"Not Toxic ({prob:.2f})")

        # Progress bar
        st.progress(min(max(prob, 0.0), 1.0))

        # Cleaned text preview (for debugging/demo)
        with st.expander("View Processed Text"):
            st.write(clean_text(user_input))


# Footer
st.markdown("---")
st.caption("Built with Streamlit | CNN Toxicity Detection Project")