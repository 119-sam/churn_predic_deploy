from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle
from sklearn.base import BaseEstimator, TransformerMixin

# CleanFixer class (used in training pipeline)
class CleanFixer(BaseEstimator, TransformerMixin):
    def __init__(self, city_col='City', threshold=100):
        self.city_col = city_col
        self.threshold = threshold
        self.valid_cities = None

    def fit(self, X, y=None):
        if self.city_col in X.columns:
            city_counts = X[self.city_col].value_counts()
            self.valid_cities = city_counts[city_counts >= self.threshold].index.tolist()
        return self

    def transform(self, X):
        X = X.copy()
        X['Total Charges'] = pd.to_numeric(X['Total Charges'], errors='coerce')
        X['Total Charges'] = X['Total Charges'].fillna(X['Monthly Charges'] * X['Tenure Months'])

        if self.city_col in X.columns and self.valid_cities is not None:
            X[self.city_col] = X[self.city_col].apply(lambda c: c if c in self.valid_cities else 'Other')

        return X

# Load the model
model = pickle.load(open("model_2.pkl", "rb"))

# Expected columns
expected_columns = [
    'City', 'Gender', 'Senior Citizen', 'Partner', 'Dependents',
    'Tenure Months', 'Phone Service', 'Multiple Lines', 'Internet Service',
    'Online Security', 'Online Backup', 'Device Protection', 'Tech Support',
    'Streaming TV', 'Streaming Movies', 'Contract', 'Paperless Billing',
    'Payment Method', 'Monthly Charges', 'Total Charges', 'Churn Score', 'CLTV'
]

# Flask app setup
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    print("🔍 RAW form data received:")
    for key in request.form:
        print(f"{key} → {repr(request.form[key])}")

    # Get input values from form
    input_data = []
    for col in expected_columns:
        field_name = col.lower().replace(" ", "_")
        value = request.form.get(field_name)
        input_data.append(value)

    # Convert numeric fields
    numeric_cols = ['Tenure Months', 'Monthly Charges', 'Total Charges', 'Churn Score', 'CLTV']
    for idx, col in enumerate(expected_columns):
        if col in numeric_cols:
            try:
                input_data[idx] = float(input_data[idx])
            except:
                return f"Invalid input for {col}. Please enter a numeric value."

    # Create input DataFrame
    input_df = pd.DataFrame([input_data], columns=expected_columns)
    print("\n================= MODEL INPUT CHECK =================")
    print(input_df)
    print("=====================================================")

    # Predict probabilities
    proba = model.predict_proba(input_df)[0]
    print(f" Model's Raw Probabilities: {proba}")

    #  Use custom threshold if you want:
    threshold = 0.4  
    prediction = 1 if proba[1] > threshold else 0

    result_label = "Churn" if prediction == 1 else "Not Churn"
    confidence = proba[1] if prediction == 1 else proba[0]

    print(f"Model's Prediction: {prediction} ({result_label}) with confidence {confidence:.2f}")

    return render_template(
        'index.html',
        prediction_text=f' Prediction: {result_label} ({confidence*100:.1f}% confidence)'
    )

if __name__ == '__main__':
    app.run(debug=True)



