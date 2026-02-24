from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
import joblib

app = FastAPI()

# Load model once at startup
model = joblib.load("mentalh_model.pkl")
vectorizer = joblib.load("vectorizr.pkl")

# HTML + CSS embedded
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>AI Mental Health Prediction</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background: linear-gradient(to right, #74ebd5, #9face6);
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
        }
        .card {
            background: white;
            padding: 40px;
            border-radius: 15px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.2);
            width: 500px;
            text-align: center;
        }
        textarea {
            width: 100%;
            height: 120px;
            padding: 10px;
            border-radius: 8px;
            border: 1px solid #ccc;
        }
        button {
            margin-top: 15px;
            padding: 10px 20px;
            border: none;
            border-radius: 8px;
            background-color: #4CAF50;
            color: white;
            cursor: pointer;
        }
        button:hover {
            background-color: #45a049;
        }
        .result {
            margin-top: 20px;
            font-size: 20px;
            font-weight: bold;
            color: #333;
        }
    </style>
</head>
<body>
    <div class="card">
        <h2>🧠 Mental Health Prediction</h2>
        <form method="post">
            <textarea name="text" placeholder="Enter your thoughts here..."></textarea><br>
            <button type="submit">Predict</button>
        </form>
        {result}
    </div>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def home():
    return HTML_PAGE.format(result="")

@app.post("/", response_class=HTMLResponse)
def predict(text: str = Form(...)):
    cleaned = text.lower()
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]

    result_html = f"<div class='result'>Prediction: {prediction}</div>"
    return HTML_PAGE.format(result=result_html)