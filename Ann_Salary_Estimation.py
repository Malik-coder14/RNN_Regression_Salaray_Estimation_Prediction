import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from tensorflow import keras

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="AI Salary Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).parent

# ============================================================
# THEME
# ============================================================
st.markdown("""
<style>
.stApp { background-color: #EAF6FF; }
.main { background-color: #EAF6FF; }

.main-title {
    text-align: center;
    color: #0B3C5D;
    font-size: 44px;
    font-weight: 800;
}
.subtitle {
    text-align: center;
    color: #486581;
    font-size: 18px;
    margin-bottom: 25px;
}
.section-title {
    color: #0B3C5D;
    font-size: 25px;
    font-weight: 700;
}
.prediction-card {
    background: white;
    padding: 30px;
    border-radius: 20px;
    text-align: center;
    border: 1px solid #C7E3F5;
    box-shadow: 0 5px 18px rgba(0, 60, 100, 0.10);
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
}
section[data-testid="stSidebar"] {
    background-color: #DFF1FF;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD ALL MODEL ASSETS
# ============================================================
@st.cache_resource
def load_assets():
    model = keras.models.load_model(
        BASE_DIR / "estimated_salary_model.keras"
    )

    with open(BASE_DIR / "gender_encoder.pkl", "rb") as f:
        gender_encoder = pickle.load(f)

    with open(BASE_DIR / "geography_encoder.pkl", "rb") as f:
        geography_encoder = pickle.load(f)

    with open(BASE_DIR / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    with open(BASE_DIR / "feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)

    return (
        model,
        gender_encoder,
        geography_encoder,
        scaler,
        feature_names
    )

try:
    (
        model,
        gender_encoder,
        geography_encoder,
        scaler,
        feature_names
    ) = load_assets()
except Exception as e:
    st.error("Model/preprocessing files could not be loaded.")
    st.exception(e)
    st.stop()

# ============================================================
# HEADER
# ============================================================
st.markdown(
    '<div class="main-title">💰 AI Estimated Salary Predictor</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="subtitle">'
    'ANN Regression Model — Estimated Salary Prediction'
    '</div>',
    unsafe_allow_html=True
)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## 🤖 Model Information")
    st.write("**Problem:** Regression")
    st.write("**Target:** EstimatedSalary")
    st.write("**Architecture:** 64 → 32 → 1")
    st.write("**Output:** Linear")
    st.write("**Loss:** MSE")
    st.write("**Input scaling:** StandardScaler")

    st.divider()
    st.write("**Model input features:**")
    for feature in feature_names:
        st.write(f"• {feature}")

# ============================================================
# INPUTS
# ============================================================
st.markdown(
    '<div class="section-title">👤 Customer Information</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

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

with col3:
    has_credit_card = st.selectbox(
        "💳 Has Credit Card",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    active_member = st.selectbox(
        "⚡ Active Member",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    # Exited is retained because the training notebook used it.
    exited = st.selectbox(
        "🚪 Exited",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

st.divider()

# ============================================================
# PREPROCESSING FUNCTION
# ============================================================
def prepare_input():
    raw = pd.DataFrame({
        "CreditScore": [credit_score],
        "Geography": [geography],
        "Gender": [gender],
        "Age": [age],
        "Tenure": [tenure],
        "Balance": [balance],
        "NumOfProducts": [num_products],
        "HasCrCard": [has_credit_card],
        "IsActiveMember": [active_member],
        "Exited": [exited]
    })

    # Use the SAME encoders used during training
    raw["Gender"] = gender_encoder.transform(
        raw[["Gender"]]
    ).ravel()

    geo_array = geography_encoder.transform(
        raw[["Geography"]]
    )
    geo_columns = geography_encoder.get_feature_names_out(
        ["Geography"]
    )

    geo_df = pd.DataFrame(
        geo_array,
        columns=geo_columns,
        index=raw.index
    )

    processed = pd.concat(
        [raw.drop(columns=["Geography"]), geo_df],
        axis=1
    )

    # Critical: force EXACT training feature order
    processed = processed.reindex(
        columns=feature_names,
        fill_value=0
    )

    # Critical: apply the SAME StandardScaler used during training
    scaled = scaler.transform(processed).astype("float32")

    return raw, processed, scaled

# ============================================================
# PREDICT
# ============================================================
predict_button = st.button(
    "🔮 Predict Estimated Salary",
    use_container_width=True
)

if predict_button:
    try:
        raw_input, processed_input, scaled_input = prepare_input()

        # Safety checks
        if scaled_input.shape[1] != model.input_shape[-1]:
            st.error(
                f"Feature mismatch: model expects "
                f"{model.input_shape[-1]} features but app created "
                f"{scaled_input.shape[1]}."
            )
            st.stop()

        prediction = model.predict(
            scaled_input,
            verbose=0
        )

        predicted_salary = float(np.asarray(prediction).ravel()[0])

        st.markdown(
            f"""
            <div class="prediction-card">
                <div class="prediction-label">
                    Estimated Annual Salary
                </div>
                <div class="prediction-value">
                    ${predicted_salary:,.2f}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.success(
            "Prediction generated using the same encoding and scaling "
            "pipeline used during model training."
        )

        with st.expander("🔍 View processed model input"):
            st.dataframe(processed_input, use_container_width=True)

        with st.expander("📐 View scaled model input"):
            st.dataframe(
                pd.DataFrame(
                    scaled_input,
                    columns=feature_names
                ),
                use_container_width=True
            )

    except Exception as e:
        st.error("Prediction failed.")
        st.exception(e)

# ============================================================
# EXPLANATION
# ============================================================
with st.expander("🧠 Why was the $1.00 prediction happening?"):
    st.write(
        """
        The original training notebook was still using a classification
        configuration: the output layer used sigmoid and the loss was
        binary_crossentropy. That restricts the network output to the
        0–1 range, so it cannot directly predict a salary such as
        $50,000 or $100,000.

        The corrected model uses one linear output neuron and MSE loss.
        The Streamlit app also applies the saved StandardScaler before
        calling model.predict(), exactly as was done during training.
        """
    )
