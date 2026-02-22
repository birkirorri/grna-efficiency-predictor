import streamlit as st
import requests

API_URL = "https://grna-efficiency-predictor.onrender.com/predict"

st.set_page_config(
    page_title="gRNA Efficiency Predictor",
    page_icon="🧬",
    layout="centered"
)

st.title("🧬 gRNA On-Target Efficiency Predictor")
st.markdown("Predicts CRISPR Cas9 cleavage efficiency from a 30-nucleotide sequence.")

st.divider()

sequence = st.text_input(
    "Enter 30-mer sequence",
    max_chars=30,
    placeholder="e.g. CACCGGAGTCCGAGCAGAAGAAGAAGGTTT"
)

if st.button("Predict", type="primary"):
    if len(sequence) != 30:
        st.error(f"Sequence must be exactly 30 nucleotides. You entered {len(sequence)}.")
    else:
        with st.spinner("Predicting..."):
            try:
                response = requests.post(API_URL, json={"sequence": sequence})
                
                if response.status_code == 200:
                    result = response.json()
                    score = result["efficiency_score"]
                    category = result["category"]

                    st.divider()

                    # colour depends on category
                    color = {"High": "green", "Medium": "orange", "Low": "red"}[category]

                    st.metric("Efficiency Score", f"{score:.4f}")
                    st.markdown(f"**Category:** :{color}[{category}]")
                    st.progress(score)

                    st.divider()
                    st.caption(f"Model version: {result['model_version']} | Sequence: {result['sequence']}")

                elif response.status_code == 422:
                    st.error(response.json()["detail"])
                else:
                    st.error(f"API error: {response.status_code}")

            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to API. Make sure the API is running on localhost:8000.")