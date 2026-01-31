import streamlit as st
import requests
from PIL import Image
import pandas as pd
import io

st.set_page_config(layout="wide", page_title="DermalScan AI")

st.title("🧠 DermalScan AI – Skin Aging Analysis")

uploaded = st.file_uploader("Upload Face Image", type=["jpg","png","jpeg"])

if "history" not in st.session_state:
    st.session_state.history = []

if uploaded:
    col1,col2 = st.columns(2)

    with col1:
        st.subheader("Before Analysis")
        img = Image.open(uploaded)
        st.image(img, use_column_width=True)

    if st.button("🔍 Start Analysis"):
        files = {"image": uploaded.getvalue()}
        res = requests.post("http://127.0.0.1:5000/analyze", files=files).json()

        with col2:
            st.subheader("After Analysis")
            st.image(res["image"], use_column_width=True)

        st.success(f"Detected: {res['label']} | Age: {res['age']}")

        st.session_state.history.append(res)

        # 🔥 Prediction Table
        st.subheader("Prediction Breakdown")
        df = pd.DataFrame(
            res["breakdown"].items(),
            columns=["Condition","Confidence %"]
        )
        st.table(df)

# 📜 HISTORY
if st.session_state.history:
    st.subheader("📜 Analysis History")
    hist_df = pd.DataFrame(st.session_state.history)
    st.dataframe(hist_df[["label","confidence","age","time"]])

    csv = hist_df.to_csv(index=False).encode()
    st.download_button("⬇️ Download CSV", csv, "history.csv")
