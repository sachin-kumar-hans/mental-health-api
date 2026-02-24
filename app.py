"""import tkinter as tk
import joblib

# Load model and vectorizer
model = joblib.load("mentalh_model.pkl")
vectorizer = joblib.load("vectorizr.pkl")

def predict_text():
    user_input = text_entry.get("1.0", tk.END)
    cleaned = user_input.lower()
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)
    result_label.config(text="Prediction: " + prediction[0])

# Create window
window = tk.Tk()
window.title("Mental Health Classifier")
window.geometry("500x400")

# Text box
text_entry = tk.Text(window, height=10, width=50)
text_entry.pack(pady=10)

# Predict button
predict_button = tk.Button(window, text="Predict", command=predict_text)
predict_button.pack(pady=10)

# Result label
result_label = tk.Label(window, text="", font=("Arial", 14))
result_label.pack(pady=10)

window.mainloop()
"""
"""from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load("mentalh_model.pkl")
vectorizer = joblib.load("vectorizr.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = ""

    if request.method == "POST":
        user_text = request.form["text"]

        cleaned = user_text.lower()
        vector = vectorizer.transform([cleaned])
        result = model.predict(vector)

        prediction = result[0]

    return render_template("app.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
"""
import streamlit as st
import joblib

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Mental Health Classifier",
    page_icon="🧠",
    layout="centered"
)

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("mentalh_model.pkl")
vectorizer = joblib.load("vectorizr.pkl")

# -----------------------------
# Custom CSS Styling (Website Look)
# -----------------------------
st.markdown("""
<style>
body {
    background-color: #f4f6f9;
}
.main {
    background-color: #ffffff;
    padding: 40px;
    border-radius: 15px;
    box-shadow: 0px 0px 20px rgba(0,0,0,0.1);
}
.title {
    text-align: center;
    font-size: 38px;
    font-weight: bold;
    color: #2E86C1;
}
.subtitle {
    text-align: center;
    font-size: 18px;
    color: #555;
}
.result {
    font-size: 24px;
    font-weight: bold;
    color: #E74C3C;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Website Header
# -----------------------------
st.markdown('<div class="title">🧠 Mental Health Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Enter your thoughts below and check mental state</div>', unsafe_allow_html=True)

st.write("")

# -----------------------------
# Text Input Section
# -----------------------------
user_text = st.text_area("Enter your text here:", height=150)

st.write("")

# -----------------------------
# Prediction Button
# -----------------------------
if st.button("🔍 Predict", use_container_width=True):

    if user_text.strip() != "":
        cleaned = user_text.lower()
        vector = vectorizer.transform([cleaned])
        result = model.predict(vector)
        prediction = result[0]

        st.markdown(f'<div class="result">Prediction: {prediction}</div>', unsafe_allow_html=True)

    else:
        st.warning("Please enter some text before predicting.")