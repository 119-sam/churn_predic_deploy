# 📊 Customer Churn Prediction - ML Web App Deployment

A Machine Learning web application that predicts whether a customer will churn (leave the service) based on various attributes like city, tenure, contract type, monthly charges, etc. This model was built using `scikit-learn`, deployed using `Flask`, and hosted on `Render`.

🔗 **Live Demo:** [https://churn-predic-deploy-render.onrender.com](https://churn-predic-deploy-render.onrender.com)

---

## 🚀 Features

- 📈 Predicts customer churn based on form inputs.
- 🧠 ML model trained with custom data preprocessing using `CleanFixer` class.
- 🌐 Deployed using Flask + Render.
- 📝 Handles missing values and city grouping for low-frequency values.
- 📊 Displays model prediction, confidence score, and class label.

---

## 📂 Project Structure
├── app.py # Flask application
├── model_2.pkl # Trained ML model with pipeline
├── requirements.txt # Python dependencies
├── templates/
│ └── index.html # Web page template


## 💡 How It Works

- User submits details via a web form.
- The input is preprocessed (e.g., filling missing values, encoding cities).
- The trained model predicts whether the customer is likely to churn.
- Output is shown with confidence level.

---

## 🧠 Model Information

- **Model Used:** RandomForestClassifier
- **Preprocessing:** Custom `CleanFixer` class
- **Accuracy:** *77%*  
- **Precision:** *56%*  
- **Recall:** *88%*  
- **F1 Score:** *68%*
---

## 🛠 Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- Flask
- HTML/CSS
- Render (for deployment)

🧠 Author
Sam
Feel free to ⭐ the repo if you liked it!

