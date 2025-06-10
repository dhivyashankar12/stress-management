Stress Prediction System – Federated Learning

A privacy-focused machine learning application that predicts human stress levels using physiological and lifestyle data. It uses **Federated Learning** to ensure data privacy by training models locally and only sharing model updates—not raw data.

---

## 📌 Project Highlights

- 🔐 **Federated Learning** architecture using Flask.
- 🧪 Trained on 5 algorithms to select the best-performing model.
- 📉 Power BI dashboard connected to real-time data (`dashboard_data.csv`).
- 🧠 Final model deployed using TensorFlow with saved `model.h5`.
- 📁 Clean structure with `shared`, `server`, and `client` components.

---

## 🧠 Machine Learning Models Compared

- ✅ Neural Network (TensorFlow)
- ✅ Logistic Regression
- ✅ Random Forest Classifier
- ✅ Support Vector Machine (SVM)
- ✅ K-Nearest Neighbors (KNN)

The model with the highest accuracy is saved and used for prediction (`model.h5`).
