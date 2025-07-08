# 💰 Insurance Cost Prediction App

This project uses an **Artificial Neural Network (ANN)** to predict a customer's **medical insurance charges** based on personal attributes like age, BMI, smoking status, and region.

Built using **TensorFlow, Streamlit, and scikit-learn**, the app enables real-time prediction through a clean UI.

---

## 🚀 Features

- 🧠 ANN-based regression model (Keras Sequential API)
- 📊 Uses tabular features like age, BMI, region, smoking status
- ⚖️ Inputs are preprocessed, encoded, and scaled with `StandardScaler`
- 💻 Web interface with Streamlit
- 💸 Predicts real-time insurance costs in ₹

---

## 📁 Project Structure

```
insurance-cost-predictor/
├── app.py                         # Streamlit UI
├── model/
│   ├── insurance\_ann\_model.h5     # Trained ANN model
│   └── scaler.pkl                 # StandardScaler object
├── 01\_insurance\_ann\_training.ipynb   # Model training notebook
├── 02\_single\_inference.ipynb         # Local test input predictor
├── insurance.csv                 # Dataset (from Kaggle)
├── requirements.txt              # Python dependencies
````

---

## 🧠 Tech Stack

| Layer          | Tool/Library              |
|----------------|---------------------------|
| Modeling       | TensorFlow / Keras        |
| Preprocessing  | scikit-learn              |
| Web App        | Streamlit                 |
| Dataset Source | Kaggle                    |

---

## 📦 Installation

```bash
git clone https://github.com/your-username/insurance-cost-predictor.git
cd insurance-cost-predictor
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
streamlit run app.py
````

---

## 📊 Dataset

* [Medical Cost Personal Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance)

---

## 📌 Sample Prediction

> Example:
>
> * Age: 45
> * BMI: 27.5
> * Smoker: Yes
> * Region: Southeast
>   → 🧠 **Predicted Cost: ₹32,352.22**

---

## 🪄 Future Enhancements

* Model evaluation (R², RMSE on test set)
* Export prediction results to CSV
* Add confidence intervals or SHAP explainability
* Host on Streamlit Cloud

---

## 📜 License

MIT © 2025 Lakshay Jain
