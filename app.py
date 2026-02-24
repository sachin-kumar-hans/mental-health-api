from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
import joblib

app = FastAPI()

# Load model
model = joblib.load("mentalh_model.pkl")
vectorizer = joblib.load("vectorizr.pkl")

# HTML Template (inside Python)
html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>AI Mental Health Prediction</title>
    <style>
        body {
            font-family: Arial;
            background: #f4f6f9;
            text-align: center;
            padding: 50px;
        }
        .container {
            background: white;
            padding: 40px;
            border-radius: 15px;
            box-shadow: 0px 0px 15px rgba(0,0,0,0.1);
            width: 50%;
            margin: auto;
        }
        textarea {
            width: 100%;
            height: 120px;
            padding: 10px;
        }
        button {
            padding: 10px 20px;
            background: #2E86C1;
            color: white;
            border: none;
            border-radius: 5px;
            margin-top: 15px;
            cursor: pointer;
        }
        button:hover {
            background: #1f6391;
        }
        .result {
            margin-top: 20px;
            font-size: 20px;
            color: #E74C3C;
            font-weight: bold;
        }
    </style>
</head>
<body>

<div class="container">
    <h2>🧠 AI Mental Health Prediction</h2>
    <form method="post">
        <textarea name="text" placeholder="Enter your thoughts here..."></textarea><br>
        <button type="submit">Predict</button>
    </form>
    {result_section}
</div>

</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def home():
    return html_content.format(result_section="")

@app.post("/", response_class=HTMLResponse)
def predict(text: str = Form(...)):
    cleaned = text.lower()
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]

    result_html = f'<div class="result">Prediction: {prediction}</div>'
    return html_content.format(result_section=result_html)