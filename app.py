from flask import Flask, render_template, request
import numpy as np
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load models
knn_model = joblib.load("models/knn_model.pkl")
ann_model = load_model("models/ann_model.h5")
cnn_model = load_model("models/cnn_model.h5")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    model_type = request.form["model"]

    study_hours = float(request.form["study_hours"])
    stress_level = float(request.form["stress_level"])
    sleep_hours = float(request.form["sleep_hours"])
    screen_time = float(request.form["screen_time"])

    features = np.array([
        study_hours,
        stress_level,
        sleep_hours,
        screen_time
    ]).reshape(1, -1)

    if model_type == "knn":
        prediction = knn_model.predict(features)

    elif model_type == "ann":
        prediction = ann_model.predict(features)
        prediction = np.argmax(prediction, axis=1)

    elif model_type == "cnn":
        features = features.reshape(
            features.shape[0],
            features.shape[1],
            1
        )
        prediction = cnn_model.predict(features)
        prediction = np.argmax(prediction, axis=1)

    classes = {
        0: "Low",
        1: "Medium",
        2: "High"
    }

    result = classes[int(prediction[0])]

    return render_template(
        "index.html",
        prediction_text=f"Burnout Risk: {result}"
    )

if __name__ == "__main__":
    app.run(debug=True)
