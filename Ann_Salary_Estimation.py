
import streamlit as st
import pandas as pd
import numpy as np
from tensorflow import keras
from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Salary Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PROFESSIONAL LIGHT BLUE THEME
# ============================================================

st.markdown("""
<style>

    /* Main application background */
    .stApp {
        background-color: #EAF6FF;
    }

    /* Main content */
    .main {
        background-color: #EAF6FF;
    }

    /* Header */
    .main-title {
        text-align: center;
        color: #0B3C5D;
        font-size: 44px;
        font-weight: 800;
        margin-top: 10px;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #486581;
        font-size: 18px;
        margin-bottom: 25px;
    }

    /* Section headings */
    .section-title {
        color: #0B3C5D;
        font-size: 25px;
        font-weight: 700;
        margin-top: 10px;
        margin-bottom: 15px;
    }

    /* Prediction card */
    .prediction-card {
        background: white;
        padding: 30px;
        border-radius: 20px;
        text-align: center;
        border: 1px solid #C7E3F5;
        box-shadow: 0 5px 18px rgba(0, 60, 100, 0.10);
        margin-top: 25px;
        margin-bottom: 20px;
    }

    .prediction-label {
        color: #486581;
        font-size: 20px;
        font-weight: 600;
    }

    .prediction-value {
        color: #087E8B;
        font-size: 46px;
        font-weight: 800;
        margin-top: 8px;
    }

    /* Info cards */
    .info-card {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #C7E3F5;
        box-shadow: 0 3px 12px rgba(0, 60, 100, 0.07);
        text-align: center;
    }

    .info-title {
        color: #486581;
        font-size: 15px;
    }

    .info-value {
        color: #0B3C5D;
        font-size: 24px;
        font-weight: 700;
    }

    /* Buttons */
    .stButton > button {
        width: 100%;
        height: 52px;
        border-radius: 12px;
        font-size: 17px;
        font-weight: 700;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #DFF1FF;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #627D98;
        font-size: 14px;
        padding: 15px;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model_path = (
        Path(__file__).parent /
        "estimated_salary_model.keras"
    )

    if not model_path.exists():
        raise FileNotFoundError(
            "estimated_salary_model.keras was not found "
            "in the same folder as app.py."
        )

    return keras.models.load_model(model_path)


try:

    model = load_model()

except Exception as e:

    st.error("❌ Model Loading Error")
    st.error(str(e))

    st.info(
        "Make sure estimated_salary_model.keras "
        "is located beside app.py."
    )

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🤖 AI Salary Predictor")

    st.write(
        "A neural-network-based application "
        "for estimated salary prediction."
    )

    st.divider()

    st.markdown("### 🧠 Model Details")

    st.write("**Problem:** Regression")

    st.write("**Target:** EstimatedSalary")

    st.write("**Architecture:** 64 → 32 → 1")

    st.write("**Optimizer:** Adam")

    st.write("**Loss:** MSE")

    st.write("**Output:** Linear")

    st.divider()

    st.markdown("### 📊 Input Features")

    st.write("Credit Score")
    st.write("Geography")
    st.write("Gender")
    st.write("Age")
    st.write("Tenure")
    st.write("Balance")
    st.write("Number of Products")
    st.write("Credit Card")
    st.write("Active Member")
    st.write("Exited")

    st.divider()

    st.caption(
        "Built with Python • TensorFlow • Keras • Streamlit"
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">'
    '💰 AI Estimated Salary Predictor'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Professional Neural Network Based Salary Prediction Dashboard'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# CUSTOMER INPUT
# ============================================================

st.markdown(
    '<div class="section-title">'
    '👤 Customer Information'
    '</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)


# ============================================================
# COLUMN 1
# ============================================================

with col1:

    credit_score = st.number_input(
        "💳 Credit Score",
        min_value=300,
        max_value=850,
        value=650,
        step=1
    )

    geography = st.selectbox(
        "🌍 Geography",
        ["France", "Germany", "Spain"]
    )

    gender = st.selectbox(
        "👤 Gender",
        ["Female", "Male"]
    )

    age = st.number_input(
        "🎂 Age",
        min_value=18,
        max_value=100,
        value=35,
        step=1
    )


# ============================================================
# COLUMN 2
# ============================================================

with col2:

    tenure = st.number_input(
        "📅 Tenure",
        min_value=0,
        max_value=10,
        value=5,
        step=1
    )

    balance = st.number_input(
        "💵 Balance",
        min_value=0.0,
        max_value=300000.0,
        value=50000.0,
        step=1000.0
    )

    num_products = st.number_input(
        "📦 Number of Products",
        min_value=1,
        max_value=4,
        value=1,
        step=1
    )


# ============================================================
# COLUMN 3
# ============================================================

with col3:

    has_credit_card = st.selectbox(
        "💳 Has Credit Card",
        [0, 1],
        format_func=lambda x:
        "Yes" if x == 1 else "No"
    )

    active_member = st.selectbox(
        "⚡ Active Member",
        [0, 1],
        format_func=lambda x:
        "Yes" if x == 1 else "No"
    )

    exited = st.selectbox(
        "🚪 Exited",
        [0, 1],
        format_func=lambda x:
        "Yes" if x == 1 else "No"
    )


st.divider()


# ============================================================
# BUTTONS
# ============================================================

button_col1, button_col2 = st.columns(2)


with button_col1:

    predict_button = st.button(
        "🔮 Predict Estimated Salary",
        use_container_width=True
    )


with button_col2:

    reset_button = st.button(
        "🔄 Reset",
        use_container_width=True
    )


# ============================================================
# RESET
# ============================================================

if reset_button:

    st.rerun()


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if credit_score < 300 or credit_score > 850:

        st.error(
            "Credit Score must be between 300 and 850."
        )

        st.stop()


    if age < 18:

        st.error(
            "Age must be at least 18."
        )

        st.stop()


    # --------------------------------------------------------
    # Gender Encoding
    # --------------------------------------------------------

    gender_encoded = (
        0
        if gender == "Female"
        else 1
    )


    # --------------------------------------------------------
    # Geography One-Hot Encoding
    # --------------------------------------------------------

    geography_france = (
        1
        if geography == "France"
        else 0
    )

    geography_germany = (
        1
        if geography == "Germany"
        else 0
    )

    geography_spain = (
        1
        if geography == "Spain"
        else 0
    )


    # --------------------------------------------------------
    # CREATE MODEL INPUT
    # --------------------------------------------------------

    input_data = pd.DataFrame({

        "CreditScore": [credit_score],

        "Geography_France": [
            geography_france
        ],

        "Geography_Germany": [
            geography_germany
        ],

        "Geography_Spain": [
            geography_spain
        ],

        "Gender": [
            gender_encoded
        ],

        "Age": [
            age
        ],

        "Tenure": [
            tenure
        ],

        "Balance": [
            balance
        ],

        "NumOfProducts": [
            num_products
        ],

        "HasCrCard": [
            has_credit_card
        ],

        "IsActiveMember": [
            active_member
        ],

        "Exited": [
            exited
        ]

    })


    # --------------------------------------------------------
    # FEATURE COUNT CHECK
    # --------------------------------------------------------

    expected_features = model.input_shape[-1]

    actual_features = input_data.shape[1]


    if actual_features != expected_features:

        st.error(
            "❌ Model/Input Feature Mismatch"
        )

        st.write(
            f"Model expects: {expected_features}"
        )

        st.write(
            f"App provides: {actual_features}"
        )

        st.write(
            "Input features:"
        )

        st.write(
            input_data.columns.tolist()
        )

        st.stop()


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    st.write("Input sent to model:")
    st.dataframe(input_data)

    try:

        prediction = model.predict(
            input_data,
            verbose=0
        )

        predicted_salary = float(
            prediction[0][0]
        )

    except Exception as e:

        st.error(
            "❌ Prediction Error"
        )

        st.error(str(e))

        st.stop()


    # --------------------------------------------------------
    # PREDICTION CARD
    # --------------------------------------------------------
    if not model_path.exists():
        raise FileNotFoundError(
            "estimated_salary_model.keras was not found "
            "in the same folder as roles.py."
        )
       # ========================================================
    # SUMMARY METRICS
    # ========================================================

    

    # ========================================================
    # PREDICTION INTERPRETATION
    # ========================================================

    st.subheader("📊 Prediction Summary")

    st.write(
        f"The neural network estimated an annual salary "
        f"of **${predicted_salary:,.2f}** based on the "
        f"customer information provided."
    )


    # ========================================================
    # INPUT DATA
    # ========================================================

    with st.expander(
        "📋 View Processed Model Input"
    ):

        st.dataframe(
            input_data,
            use_container_width=True
        )


    # ========================================================
    # DOWNLOAD RESULT
    # ========================================================

    result_data = pd.DataFrame({

        "CreditScore": [credit_score],

        "Geography": [geography],

        "Gender": [gender],

        "Age": [age],

        "Tenure": [tenure],

        "Balance": [balance],

        "NumOfProducts": [
            num_products
        ],

        "HasCrCard": [
            has_credit_card
        ],

        "IsActiveMember": [
            active_member
        ],

        "Exited": [
            exited
        ]
       
    })


    csv = result_data.to_csv(
        index=False
    ).encode("utf-8")


    st.download_button(
        label="📥 Download Prediction Report",
        data=csv,
        file_name="salary_prediction_report.csv",
        mime="text/csv",
        use_container_width=True
    )


# ============================================================
# ABOUT MODEL
# ============================================================

with st.expander(
    "🧠 About This AI Model"
):

    st.write(
        """
        This application uses a neural network regression model
        to estimate a customer's salary.

        **Model Architecture**

        Input Layer → 64 neurons → 32 neurons → 1 output neuron

        **Output Layer**

        The final neuron uses a linear activation because
        EstimatedSalary is a continuous numerical value.

        **Loss Function**

        Mean Squared Error (MSE) is used during training.
        MSE penalizes large prediction errors more strongly.

        **Evaluation Metrics**

        The model can be evaluated using:

        • MAE — Mean Absolute Error

        • RMSE — Root Mean Squared Error

        • R² — Coefficient of Determination
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    '<div class="footer">'
    '💰 AI Estimated Salary Predictor | '
    'TensorFlow + Keras + Streamlit'
    '</div>',
    unsafe_allow_html=True
)
