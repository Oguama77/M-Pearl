import streamlit as st
import requests

st.title("MamaPEARL MVP")

st.info("This tool helps predict the risk of late-onset preeclampsia. You can ask general medical questions, upload test results, or enter your data manually for a prediction.\n The results of a complete blood count, liver and kidney function tests and urine ACR are required for accurate prediction of late onset preeclmpsia.")

BACKEND_URL = "http://localhost:8000"

# --- Chatbot ---
st.header("Chat with our bot")
chat_input = st.text_input("Ask a general medical question or inquire about preeclampsia:")
if st.button("Send") and chat_input:
    with st.spinner("Thinking..."):
        try:
            response = requests.post(f"{BACKEND_URL}/chat", json={"message": chat_input})
            if response.ok:
                st.success(response.json()["response"])
            else:
                st.error(f"Error communicating with backend: {response.text}")
        except requests.exceptions.RequestException as e:
            st.error(f"Connection error: {e}")


# --- Image Extraction ---
st.header("Extract Variables from Test Result Images")
uploaded_files = st.file_uploader(
    "Upload one or more images of lab reports or patient charts",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)
if st.button("Extract Variables from Images") and uploaded_files:
    with st.spinner("Extracting data from images..."):
        try:
            files = [
                ("files", (file.name, file.getvalue(), file.type))
                for file in uploaded_files
            ]
            response = requests.post(f"{BACKEND_URL}/extract", files=files)
            if response.ok:
                result = response.json()
                if "prediction" in result:
                    st.success(f"Prediction: {result['prediction']}")
                elif "error" in result:
                    st.error(f"Error: {result['error']}")
                else:
                    st.error("Unexpected response from backend.")
            else:
                st.error(f"Error extracting variables: {response.text}")
        except requests.exceptions.RequestException as e:
            st.error(f"Connection error: {e}")

# --- Manual Prediction ---
st.header("Preeclampsia Predictor")

variable_list = [
    "Gestational age", "Albumin level", "Alkaline phosphate level",
    "Alanine transaminase level", "Aspartate transaminase level", "Blood urea nitrogen level",
    "Calcium level", "Cholesterol level", "Serum creatinine level", "C-reactive protein level",
    "Erythrocyte sedimentation rate", "Gamma-glutamyl transferase (GGT) level", "Glucose level",
    "Hemoglobin", "Potassium", "Magnesium", "Platelet count", "Total bilirubin",
    "Total CO2 (bicarbonate)", "Total protein", "Uric acid",
    "Urine albumin-to-creatinine ration", "Urine protein/creatinine ratio",
    "White blood cell count", "Systolic blood pressure", "Diastolic blood pressure",
    "Protein level in urine", "Height (cm)", "Maternal weight at pregnancy (kg)", "Fundal height (cm)"
]

mean_values = [
    30.383824, 0.047224, 2.114673, 59.325487, 13.121650, 16.193501, 5.908047, 5.209185, 149.911673, 0.410707,
    5.840941, 3.739687, 9.714550, 66.197603, 2.134625, 2.607922, 0.070634, 199.839308, 0.281055, 13.475048,
    3.761391, 2.503311, 31.665389, 0.203468, 8.857324, 111.964249, 67.989443, 963.371447, 160.779131, 60.139828, 58.334186
]

with st.expander("Manual Variable Entry (Advanced)"):
    st.markdown("Enter the values for each variable below. Leave blank to use the mean value.")
    manual_vars = []
    for i, (name, mean) in enumerate(zip(variable_list, mean_values)):
        val = st.text_input(f"{i+1}. {name}", key=f"var_{i}")
        manual_vars.append(val)
    if st.button("Predict Preeclampsia Risk (Manual Input)"):
        with st.spinner("Calculating risk..."):
            try:
                final_values = []
                for val, mean in zip(manual_vars, mean_values):
                    if val.strip() == "":
                        final_values.append(mean)
                    else:
                        final_values.append(float(val.strip()))
                st.markdown("**Variables used for prediction:**")
                for i, (name, val) in enumerate(zip(variable_list, final_values)):
                    st.write(f"{i+1}. {name}: {val}")
                response = requests.post(f"{BACKEND_URL}/predict", json={"variables": final_values})
                if response.ok:
                    st.success(response.json()["response"])
                else:
                    st.error(f"Error communicating with backend: {response.text}")
            except ValueError:
                st.error("Invalid input. Please enter only numbers or leave blank for mean.")
            except requests.exceptions.RequestException as e:
                st.error(f"Connection error: {e}") 