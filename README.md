# 💰 AI Estimated Salary Predictor

An end-to-end **Artificial Neural Network (ANN) regression project** that predicts a customer's estimated annual salary using selected banking and demographic information.

## 📌 Project Overview

This project demonstrates how an **Artificial Neural Network can be used for regression** to predict a continuous numerical value.

**Target:** `EstimatedSalary`

### Input Features

The final model uses only these 9 original customer features:

- `CreditScore`
- `Age`
- `Geography`
- `Tenure`
- `HasCrCard`
- `Balance`
- `IsActiveMember`
- `Gender`
- `NumOfProducts`

> **Note:** `Exited` is intentionally excluded from both model training and the Streamlit prediction interface.

## 🎯 Objectives

1. Prepare customer data for machine learning.
2. Select the required salary-prediction features.
3. Encode categorical variables.
4. Scale model inputs consistently.
5. Build an ANN regression model.
6. Train the model using Mean Squared Error (MSE).
7. Evaluate the model using MAE, RMSE, and R².
8. Save the trained model and preprocessing objects.
9. Deploy the model through Streamlit.
10. Provide real-time estimated salary predictions.

## 📊 Dataset

The project uses the **Churn Modelling** customer dataset.

### Features Used

| Feature | Description |
|---|---|
| `CreditScore` | Customer's credit score |
| `Age` | Customer age |
| `Geography` | Customer's country |
| `Tenure` | Number of years with the bank |
| `HasCrCard` | Whether the customer has a credit card |
| `Balance` | Customer's bank balance |
| `IsActiveMember` | Whether the customer is an active member |
| `Gender` | Customer gender |
| `NumOfProducts` | Number of bank products used |

### Target

```text
EstimatedSalary
```

### Excluded Columns

```text
RowNumber
CustomerId
Surname
Exited
```

## 🧠 ANN Regression Model

### Architecture

```text
Input Layer
     ↓
Dense Layer — 64 neurons — ReLU
     ↓
Dense Layer — 32 neurons — ReLU
     ↓
Output Layer — 1 neuron — Linear
     ↓
EstimatedSalary
```

Because `EstimatedSalary` is continuous, the output layer uses:

```python
Dense(1, activation="linear")
```

A sigmoid output is not appropriate for salary regression because sigmoid restricts predictions to the 0–1 range.

## ⚙️ Data Preprocessing

The same preprocessing pipeline is used during training and prediction.

### Feature Selection

```python
feature_columns = [
    "CreditScore",
    "Age",
    "Geography",
    "Tenure",
    "HasCrCard",
    "Balance",
    "IsActiveMember",
    "Gender",
    "NumOfProducts"
]

X = df_data[feature_columns].copy()
```

### Categorical Encoding

- `Gender` is encoded numerically.
- `Geography` is one-hot encoded into columns such as:
  - `Geography_France`
  - `Geography_Germany`
  - `Geography_Spain`

Thus, the 9 original features become multiple numerical inputs for the ANN.

### Feature Scaling

`StandardScaler` is fitted on the training data and the same saved scaler is used for Streamlit predictions.

```python
scaler.fit_transform(X_train)
```

and during prediction:

```python
scaler.transform(input_data)
```

## 🏋️ Model Training

The dataset is split into training and testing data using:

```python
train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)
```

### Training Configuration

| Setting | Value |
|---|---|
| Problem | Regression |
| Model | Artificial Neural Network |
| Hidden Layers | 2 |
| Hidden Neurons | 64 → 32 |
| Hidden Activation | ReLU |
| Output Neurons | 1 |
| Output Activation | Linear |
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Loss | MSE |
| Metric | MAE |
| Test Size | 20% |
| Random State | 42 |

Early stopping is used to restore the best validation weights.

## 📏 Evaluation Metrics

### MAE — Mean Absolute Error

MAE measures the average absolute difference between actual and predicted salary.

For example, an MAE of `$15,000` means the predictions have an average absolute error of approximately `$15,000` on the evaluated data.

### RMSE — Root Mean Squared Error

RMSE measures prediction error while giving greater influence to larger errors.

### R² — R-Squared

R² measures the proportion of variation in the target explained by the model on the evaluated dataset.

## 🌐 Streamlit Application

The Streamlit application provides an interactive interface for:

- Credit Score
- Geography
- Gender
- Age
- Tenure
- Balance
- Number of Products
- Has Credit Card
- Active Member

There is intentionally **no `Exited` input**.

Click:

```text
🔮 Predict Estimated Salary
```

to generate a salary estimate.

## 🔄 Prediction Pipeline

```text
User Input
    ↓
Validation
    ↓
Gender Encoding
    ↓
Geography One-Hot Encoding
    ↓
Exact Training Feature Order
    ↓
StandardScaler
    ↓
ANN Model
    ↓
Linear Output
    ↓
Estimated Salary
```

## 📁 Project Structure

```text
AI-Estimated-Salary-Predictor/
│
├── Ann_Salary_Estimation.py
├── train_salary_model_fixed.py
│
├── estimated_salary_model.keras
├── gender_encoder.pkl
├── geography_encoder.pkl
├── scaler.pkl
├── feature_names.pkl
│
├── Churn_Modelling(3).csv
├── requirements.txt
└── README.md
```

### File Descriptions

| File | Purpose |
|---|---|
| `Ann_Salary_Estimation.py` | Streamlit prediction application |
| `train_salary_model_fixed.py` | ANN training and evaluation |
| `estimated_salary_model.keras` | Saved trained ANN |
| `gender_encoder.pkl` | Saved Gender encoder |
| `geography_encoder.pkl` | Saved Geography encoder |
| `scaler.pkl` | Saved StandardScaler |
| `feature_names.pkl` | Saved model feature order |
| `Churn_Modelling(3).csv` | Dataset |
| `requirements.txt` | Python dependencies |
| `README.md` | Project documentation |

## 🛠️ Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- TensorFlow
- Keras
- Streamlit

## 📦 Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

Or:

```bash
pip install pandas numpy scikit-learn tensorflow streamlit
```

## 🚀 Running the Project

### 1. Train the Model

```bash
python train_salary_model_fixed.py
```

This generates:

```text
estimated_salary_model.keras
gender_encoder.pkl
geography_encoder.pkl
scaler.pkl
feature_names.pkl
```

### 2. Run Streamlit

```bash
streamlit run Ann_Salary_Estimation.py
```

The application will open in your browser.

## ⚠️ Important: Retrain After Changing Features

If `Exited` is removed from the feature set, the model must be retrained.

Do not use an old model trained with `Exited`.

The model, encoders, scaler, and feature-name list should all come from the same training run.

## 🐛 Previous `$1.00` Prediction Issue

An earlier version produced approximately `$1.00` for repeated predictions.

The main issue was an incorrect classification configuration for a regression problem. The corrected model uses:

```python
Dense(1, activation="linear")
```

with:

```python
loss="mse"
```

The Streamlit application also applies the same encoding, feature ordering, and scaling pipeline used during training.

## 🔍 Model Input Verification

The application verifies that the processed feature count matches the trained ANN:

```python
if scaled_input.shape[1] != model.input_shape[-1]:
    st.error("Feature mismatch")
```

It also preserves the exact feature order saved during training.

## 💡 Limitations

The model produces an **estimate**, not a guaranteed future salary.

Prediction quality depends on:

- Dataset quality
- Available features
- Feature relationships
- Model architecture
- Training process
- Data distribution
- Generalization to new customers

## 🎓 Learning Outcomes

This project demonstrates practical experience with:

- Regression
- Artificial Neural Networks
- TensorFlow/Keras
- Feature selection
- Categorical encoding
- One-hot encoding
- Feature scaling
- Train/test splitting
- MSE
- MAE
- RMSE
- R²
- Early stopping
- Model persistence
- Consistent preprocessing
- Streamlit deployment
- Interactive machine-learning applications

## 👨‍💻 Author

**Malik Muddaser Hasnain**

Full Stack Web Developer | AI Engineering Enthusiast

Areas of interest:

- Full Stack Web Development
- Artificial Intelligence
- Machine Learning
- Deep Learning
- Supervised Learning
- Regression
- Classification
- Artificial Neural Networks
- Recurrent Neural Networks
- AI-powered Applications

## ⭐ Project Summary

**AI Estimated Salary Predictor** is an end-to-end deep-learning application demonstrating how an ANN regression model can be trained and deployed for continuous salary prediction.

```text
Dataset
   ↓
Feature Selection
   ↓
Categorical Encoding
   ↓
Feature Scaling
   ↓
ANN Regression
   ↓
Model Evaluation
   ↓
Model Saving
   ↓
Streamlit Deployment
   ↓
Estimated Salary
```
