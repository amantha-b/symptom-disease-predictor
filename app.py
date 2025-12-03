# app.py
import numpy as np
import pandas as pd
import streamlit as st
import joblib

# ----------------------------
# 1. Load pre-trained model
# ----------------------------

@st.cache_resource
def load_model():
    model = joblib.load("disease_model.pkl")
    symptom_cols = joblib.load("symptom_columns.pkl")
    return model, symptom_cols

rf_model, symptom_columns = load_model()

# ----------------------------
# 2. Page config
# ----------------------------

st.set_page_config(
    page_title="Symptom-based Disease Predictor",
    page_icon=None,
    layout="wide",
)

# ----------------------------
# 3. Custom CSS
# ----------------------------

st.markdown(
    """
    <style>
    .main {
        padding-top: 1.5rem;
    }
    .big-title {
        font-size: 2.2rem;
        font-weight: 700;
    }
    .subtitle {
        color: #9ca3af;
        font-size: 0.95rem;
    }
    .card {
        padding: 1.5rem;
        border-radius: 1rem;
        background: #020617;
        border: 1px solid #1e293b;
    }
    .metric-card {
        padding: 0.8rem 1rem;
        border-radius: 0.75rem;
        background: #0f172a;
        border: 1px solid #1f2937;
        font-size: 0.9rem;
    }
    .footer-text {
        color: #6b7280;
        font-size: 0.8rem;
        margin-top: 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# 4. Header
# ----------------------------

st.markdown('<div class="big-title">Symptom-based Disease Predictor</div>', unsafe_allow_html=True)
st.markdown(
    """
    <p class="subtitle">
    This tool uses a machine learning model to provide an estimated disease prediction
    based on selected symptoms. It is intended for academic and demonstration purposes only.
    </p>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")

# ----------------------------
# Technology Stack Section
# ----------------------------

st.sidebar.title("Technology Stack")

st.sidebar.markdown(
    """
    **Programming Language:**  
    - Python 3.x  

    **Machine Learning:**  
    - Scikit-learn (Random Forest Classifier)  
    - NumPy  
    - Pandas  

    **Model Serving:**  
    - Joblib (for model serialization)  

    **Web Framework:**  
    - Streamlit  

    **User Interface:**  
    - Custom CSS styling  
    - Streamlit theme overrides  

    **Deployment (local):**  
    - Streamlit CLI  
    """
)


# ----------------------------
# 5. Info cards
# ----------------------------

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(
        '<div class="metric-card"><strong>Model</strong><br>Random Forest Classifier</div>',
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        f'<div class="metric-card"><strong>Total Symptoms</strong><br>{len(symptom_columns)}</div>',
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        '<div class="metric-card"><strong>Output Format</strong><br>Disease + Probability Estimates</div>',
        unsafe_allow_html=True,
    )

# ----------------------------
# 6. Layout: Symptoms (left) / Prediction (right)
# ----------------------------

left_col, right_col = st.columns([1.2, 1])

with left_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Select Symptoms")

    st.markdown(
        "Start typing to search for symptoms. Select one or more symptoms from the list below."
    )

    selected_symptoms = st.multiselect(
        "Symptoms:",
        options=symptom_columns,
        help="You can select multiple symptoms.",
    )

    st.markdown("#### Selected Symptoms")
    if selected_symptoms:
        for s in selected_symptoms:
            st.write(f"- {s}")
    else:
        st.info("No symptoms selected.")
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------
# 7. Prediction logic
# ----------------------------

def predict_disease(symptom_list):
    input_vector = np.zeros(len(symptom_columns), dtype=int)

    for s in symptom_list:
        if s in symptom_columns:
            idx = symptom_columns.index(s)
            input_vector[idx] = 1

    input_df = pd.DataFrame([input_vector], columns=symptom_columns)

    prediction = rf_model.predict(input_df)[0]
    probs = rf_model.predict_proba(input_df)[0]
    classes = rf_model.classes_

    top_indices = np.argsort(probs)[::-1][:3]
    top_results = [(classes[i], round(probs[i] * 100, 2)) for i in top_indices]

    return prediction, top_results

with right_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Disease Prediction")

    if st.button("Predict Disease", use_container_width=True):
        if not selected_symptoms:
            st.warning("Please select at least one symptom before predicting.")
        else:
            pred_disease, top3 = predict_disease(selected_symptoms)

            st.success(f"Predicted Disease: {pred_disease}")

            st.markdown("#### Top 3 Estimated Probabilities:")
            top_df = pd.DataFrame(
                {
                    "Disease": [d for d, _ in top3],
                    "Probability (%)": [p for _, p in top3],
                }
            )
            st.dataframe(top_df, use_container_width=True)
            st.caption(
                "These values represent the model's probability distribution across all disease classes."
            )
    else:
        st.info("Click 'Predict Disease' to view the results.")

    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------
# 8. Footer
# ----------------------------

st.markdown(
    '<div class="footer-text">© 2025 Disease Prediction Demo · Developed by Manish Kumar Kondoju · Academic Use Only.</div>',
    unsafe_allow_html=True,
)
