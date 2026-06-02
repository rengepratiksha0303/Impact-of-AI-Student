# AI Student Impact Prediction

A Flask-based Machine Learning web application that predicts student burnout risk using:

- KNN (K-Nearest Neighbors)
- ANN (Artificial Neural Network)
- CNN (Convolutional Neural Network)

## Features

- Predict Burnout Risk Level
- Compare KNN, ANN, and CNN models
- User-friendly web interface
- Flask backend

## Dataset

The model is trained on the AI Student Impact Dataset.

Target Variable:

Burnout_Risk_Level

Classes:

- Low
- Medium
- High

## Installation

Clone repository:

```bash
git clone https://github.com/yourusername/student-impact-predictor.git
cd student-impact-predictor
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run application:

```bash
python app.py
```

Open browser:

```text
http://127.0.0.1:5000
```

## Project Structure

```text
student-impact-predictor/
│
├── app.py
├── requirements.txt
├── README.md
├── templates/
│   └── index.html
│
└── models/
    ├── knn_model.pkl
    ├── ann_model.h5
    └── cnn_model.h5
```

## Technologies Used

- Python
- Flask
- Scikit-Learn
- TensorFlow
- Pandas
- NumPy

## Author

Pratiksha Renge
