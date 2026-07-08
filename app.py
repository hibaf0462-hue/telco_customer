import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Customer Churn Predictor", page_icon="📶", layout="centered")

@st.cache_resource
def load_artifacts():
    model = joblib.load("best_model.pkl")
    scaler = joblib.load("scaler.pkl")
    with open("meta.json") as f:
        meta = json.load(f)
    return model, scaler, meta

model, scaler, meta = load_artifacts()
FEATURE_COLUMNS = meta["feature_columns"]
CATEGORY_OPTIONS = meta["category_options"]
NUM_COLS = meta["num_cols"]

st.title("📶 Customer Churn Predictor")
st.write(
    "Fill in a customer's details below and this app will predict whether "
    f"they are likely to churn, using a **{meta['best_model_name']}** model "
    "trained on the telco customer dataset."
)

with st.expander("ℹ️ About this model"):
    st.write("Accuracy / F1 of each model evaluated during training:")
    st.dataframe(pd.DataFrame(meta["results"]).set_index("model"), use_container_width=True)

st.divider()

# ----------------------------------------------------------------------------
# Input form
# ----------------------------------------------------------------------------
with st.form("customer_form"):
    st.subheader("Customer profile")

    c1, c2 = st.columns(2)
    with c1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox("Senior citizen?", ["No", "Yes"])
        partner = st.selectbox("Has a partner?", ["No", "Yes"])
        dependents = st.selectbox("Has dependents?", ["No", "Yes"])
    with c2:
        tenure = st.slider("Tenure (months with company)", 0, 72, 12)
        monthly_charges = st.number_input("Monthly charges ($)", min_value=0.0, max_value=200.0, value=70.0, step=0.5)
        total_charges = st.number_input(
            "Total charges ($)", min_value=0.0, max_value=10000.0,
            value=float(round(monthly_charges * tenure, 2)), step=1.0
        )

    st.subheader("Services")
    c3, c4 = st.columns(2)
    with c3:
        phone_service = st.selectbox("Phone service?", ["No", "Yes"])
        multiple_lines = st.selectbox("Multiple lines?", CATEGORY_OPTIONS["MultipleLines"])
        internet_service = st.selectbox("Internet service", CATEGORY_OPTIONS["InternetService"])
        online_security = st.selectbox("Online security", CATEGORY_OPTIONS["OnlineSecurity"])
        online_backup = st.selectbox("Online backup", CATEGORY_OPTIONS["OnlineBackup"])
    with c4:
        device_protection = st.selectbox("Device protection", CATEGORY_OPTIONS["DeviceProtection"])
        tech_support = st.selectbox("Tech support", CATEGORY_OPTIONS["TechSupport"])
        streaming_tv = st.selectbox("Streaming TV", CATEGORY_OPTIONS["StreamingTV"])
        streaming_movies = st.selectbox("Streaming movies", CATEGORY_OPTIONS["StreamingMovies"])

    st.subheader("Account")
    c5, c6 = st.columns(2)
    with c5:
        contract = st.selectbox("Contract type", CATEGORY_OPTIONS["Contract"])
        paperless_billing = st.selectbox("Paperless billing?", ["No", "Yes"])
    with c6:
        payment_method = st.selectbox("Payment method", CATEGORY_OPTIONS["PaymentMethod"])

    submitted = st.form_submit_button("Predict churn", use_container_width=True)

# ----------------------------------------------------------------------------
# Build a feature row identical to the training pipeline, then predict
# ----------------------------------------------------------------------------
def build_feature_row(raw: dict) -> pd.DataFrame:
    row = {col: 0 for col in FEATURE_COLUMNS}

    row["gender"] = 1 if raw["gender"] == "Male" else 0
    row["SeniorCitizen"] = 1 if raw["senior_citizen"] == "Yes" else 0
    row["Partner"] = 1 if raw["partner"] == "Yes" else 0
    row["Dependents"] = 1 if raw["dependents"] == "Yes" else 0
    row["PhoneService"] = 1 if raw["phone_service"] == "Yes" else 0
    row["PaperlessBilling"] = 1 if raw["paperless_billing"] == "Yes" else 0

    row["tenure"] = raw["tenure"]
    row["MonthlyCharges"] = raw["monthly_charges"]
    row["TotalCharges"] = raw["total_charges"]

    onehot_map = {
        "MultipleLines": raw["multiple_lines"],
        "InternetService": raw["internet_service"],
        "OnlineSecurity": raw["online_security"],
        "OnlineBackup": raw["online_backup"],
        "DeviceProtection": raw["device_protection"],
        "TechSupport": raw["tech_support"],
        "StreamingTV": raw["streaming_tv"],
        "StreamingMovies": raw["streaming_movies"],
        "Contract": raw["contract"],
        "PaymentMethod": raw["payment_method"],
    }
    for prefix, value in onehot_map.items():
        col_name = f"{prefix}_{value}"
        if col_name in row:
            row[col_name] = 1

    df_row = pd.DataFrame([row], columns=FEATURE_COLUMNS)
    df_row[NUM_COLS] = scaler.transform(df_row[NUM_COLS])
    return df_row


if submitted:
    raw = dict(
        gender=gender, senior_citizen=senior_citizen, partner=partner, dependents=dependents,
        tenure=tenure, monthly_charges=monthly_charges, total_charges=total_charges,
        phone_service=phone_service, multiple_lines=multiple_lines, internet_service=internet_service,
        online_security=online_security, online_backup=online_backup, device_protection=device_protection,
        tech_support=tech_support, streaming_tv=streaming_tv, streaming_movies=streaming_movies,
        contract=contract, paperless_billing=paperless_billing, payment_method=payment_method,
    )

    X_new = build_feature_row(raw)
    pred = model.predict(X_new)[0]

    st.divider()
    st.subheader("Result")

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_new)[0][1]
        st.metric("Churn probability", f"{proba:.1%}")
        st.progress(min(max(proba, 0.0), 1.0))
    else:
        proba = None

    if pred == 1:
        st.error("⚠️ This customer is **likely to churn**.")
    else:
        st.success("✅ This customer is **likely to stay**.")

    with st.expander("See the exact feature row sent to the model"):
        st.dataframe(X_new, use_container_width=True)
