import streamlit as st
import pandas as pd
import joblib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Wine Quality Prediction",
    page_icon="🍷",
    layout="centered"
)

# ---------------- CUSTOM STYLING ---------------- 
st.markdown("""
<style>
/* App background */
.stApp {
    background: linear-gradient(135deg, #fff0f0, #ffe6e6);
}

/* Main title */
.big-title {
    font-size: 42px;
    font-weight: bold;
    color: #7a0019;
    text-align: center;
    margin-bottom: 5px;
}

/* Subtitle */
.sub-text {
    text-align: center;
    font-size: 18px;
    color: #4d0000;
    margin-bottom: 25px;
}

/* Card container */
.card {
    background-color: white;
    padding: 25px;
    border-radius: 18px;
    box-shadow: 0px 8px 20px rgba(0,0,0,0.15);
    margin-bottom: 20px;
}

/* Button */
.stButton>button {
    background: linear-gradient(90deg, #7a0019, #b3002d);
    color: white;
    border-radius: 12px;
    height: 3.2em;
    font-size: 18px;
    font-weight: bold;
}
.stButton>button:hover {
    background: linear-gradient(90deg, #b3002d, #7a0019);
}

/* Success box */
.stAlert {
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<div class="big-title">🍷 Wine Quality Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">Predict wine quality using Machine Learning</div>', unsafe_allow_html=True)

# ---------------- LOAD MODEL & SCALER ----------------
@st.cache_resource
def load_model():
    model = joblib.load("finalized_RF_model.sav")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_model()

# ---------------- INPUT CARD ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("### 🍷 Enter Wine Chemical Properties")

col1, col2 = st.columns(2)

with col1:
    fixed_acidity = st.slider("Fixed Acidity", 0.0, 20.0, 7.4)
    volatile_acidity = st.slider("Volatile Acidity", 0.0, 2.0, 0.7)
    citric_acid = st.slider("Citric Acid", 0.0, 1.0, 0.0)
    residual_sugar = st.slider("Residual Sugar", 0.0, 50.0, 0.6)
    chlorides = st.slider("Chlorides", 0.0, 1.0, 0.09)
    free_sulfur_dioxide = st.slider("Free Sulfur Dioxide", 0.0, 100.0, 15.0)

with col2:
    total_sulfur_dioxide = st.slider("Total Sulfur Dioxide", 0.0, 300.0, 98.0)
    density = st.slider("Density", 0.9, 1.5, 1.0)
    ph = st.slider("pH", 2.0, 5.0, 3.0)
    sulphates = st.slider("Sulphates", 0.0, 5.0, 0.6)
    alcohol = st.slider("Alcohol", 0.0, 20.0, 10.5)

st.markdown('</div>', unsafe_allow_html=True)

# ---------------- PREDICTION ----------------
if st.button("🍾 Predict Wine Quality", use_container_width=True):
    input_data = pd.DataFrame([[
        fixed_acidity,
        volatile_acidity,
        citric_acid,
        residual_sugar,
        chlorides,
        free_sulfur_dioxide,
        total_sulfur_dioxide,
        density,
        ph,
        sulphates,
        alcohol
    ]], columns=[
        'fixed acidity', 'volatile acidity', 'citric acid',
        'residual sugar', 'chlorides', 'free sulfur dioxide',
        'total sulfur dioxide', 'density', 'pH',
        'sulphates', 'alcohol'
    ])

    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)[0]

    # 🎈 Celebration
    st.balloons()

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("## 🍷 Prediction Result")
    st.success(f"Predicted Wine Quality Score: **{round(prediction)}**")

    if prediction >= 7:
        st.markdown("### ✅ Excellent Quality Wine 🍾")
    elif prediction >= 5:
        st.markdown("### 🙂 Average Quality Wine")
    else:
        st.markdown("### ⚠️ Low Quality Wine")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("✨ Built with Streamlit & Machine Learning | Wine Quality Prediction 🍷")