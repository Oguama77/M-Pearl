import os
import joblib
from fastapi import FastAPI, File, UploadFile, Body
from pydantic import BaseModel
from typing import List
import openai
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI
import base64
import re
import json

# Load environment variables from .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Load your ML model and scaler
MODEL_PATH = "model.pkl"  # Change this to your actual model filename
SCALER_PATH = "scaler.pkl"  # Path to your saved scaler
with open(MODEL_PATH, "rb") as f:
    preeclampsia_model = joblib.load(f)
with open(SCALER_PATH, "rb") as f:
    scaler = joblib.load(f)

# Set your OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")

# Define the ML tool
def predict_preeclampsia(variables: List[float]) -> str:
    scaled_vars = scaler.transform([variables])
    pred = preeclampsia_model.predict(scaled_vars)[0]
    if pred == 0:
        return "No risk of late-onset preeclampsia"
    elif pred == 1:
        return "Risk of late-onset preeclampsia"
    else:
        return f"Model output: {pred}"

# Define the agent's routing logic
def is_preeclampsia_query(query: str) -> bool:
    keywords = [
        "preeclampsia", "blood pressure", "proteinuria", "gestational age",
        "hypertension", "pregnancy", "late-onset", "risk factors"
    ]
    return any(word in query.lower() for word in keywords)

# LangChain LLM setup
llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)

# General LLM tool for healthcare Q&A
def general_healthcare_qa(query: str) -> str:
    system_prompt = (
        "You are a helpful and knowledgeable medical assistant. "
        "Provide detailed, clear, and informative answers to healthcare questions. "
        "Your responses should be about 200 words long, unless the question is very simple."
    )
    return llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query}
    ])

llm_tool = Tool(
    name="General Healthcare QA",
    func=general_healthcare_qa,
    description="Answers general healthcare and medical questions."
)

# Define tools for LangChain
preeclampsia_tool = Tool(
    name="preeclampsia_predictor",
    func=lambda x: predict_preeclampsia([float(i) for i in x.split(",")]),
    description="Predicts late-onset preeclampsia from patient variables. Input should be a comma-separated list of floats."
)

tools = [preeclampsia_tool, llm_tool]

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# FastAPI app
app = FastAPI()

class ChatInput(BaseModel):
    message: str

class FormInput(BaseModel):
    variables: List[float]

@app.post("/chat")
async def chat_endpoint(data: ChatInput):
    # Check if user is asking about preeclampsia risk specifically
    preeclampsia_keywords = [
        "preeclampsia", "preeclampsia risk", "risk of preeclampsia", 
        "check preeclampsia", "preeclampsia prediction", "late-onset preeclampsia"
    ]
    
    is_asking_about_risk = any(keyword in data.message.lower() for keyword in preeclampsia_keywords)
    
    if is_asking_about_risk:
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
        
        response_text = (
            "I can help you check your risk of late-onset preeclampsia. "
            "To provide an accurate assessment, I need the following 30 variables from your medical tests:\n\n"
        )
        
        for i, var in enumerate(variable_list, 1):
            response_text += f"{i}. {var}\n"
        
        response_text += (
            "\nPlease provide these values as comma-separated numbers in the order listed above. "
            "You can also upload an image of your test results using the 'Extract Variables from Test Result Image' feature above."
        )
        
        return {"response": response_text}
    else:
        # Handle general healthcare questions
        response = agent.invoke({"input": data.message})
        return {"response": response["output"]}

@app.post("/predict")
async def predict_endpoint(data: FormInput):
    response = predict_preeclampsia(data.variables)
    return {"response": response}

def extract_json_from_llm_output(text):
    # Remove Markdown code block if present
    match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    if match:
        text = match.group(1)
    # Remove any leading/trailing whitespace
    return text.strip()

@app.post("/extract")
async def extract_variables(files: List[UploadFile] = File(...)):
    image_b64_list = []
    for file in files:
        image_bytes = await file.read()
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        image_b64_list.append(image_b64)

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
    prompt = (
        "From the provided medical test result images, extract the numerical values for the following variables. "
        "If a variable is not present, use 'null'. Return the result as a JSON object where the keys are the variable names "
        f"from this list: {', '.join(variable_list)}. "
        "For example: {\"Gestational age\": 34, \"Albumin level\": 3.5, ...}"
    )

    # Compose the content for all images
    content = [{"type": "text", "text": prompt}]
    for image_b64 in image_b64_list:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}})

    from openai import OpenAI
    client = OpenAI(api_key=openai_api_key)
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are a medical assistant that extracts structured data from images."},
            {
                "role": "user",
                "content": content
            }
        ],
        max_tokens=512
    )
    raw_output = response.choices[0].message.content
    json_str = extract_json_from_llm_output(raw_output)
    try:
        extracted = json.loads(json_str)
        # Fill missing/null values with means
        final_values = []
        for i, var in enumerate(variable_list):
            val = extracted.get(var)
            if val is None or (isinstance(val, str) and val.lower() == "null"):
                final_values.append(mean_values[i])
            else:
                try:
                    final_values.append(float(val))
                except Exception:
                    final_values.append(mean_values[i])
        # Make prediction
        prediction = predict_preeclampsia(final_values)
        return {"prediction": prediction}
    except Exception as e:
        return {"error": f"Could not parse output as JSON: {json_str}. Error: {e}"}

mean_values = [
    30.383824, 0.047224, 2.114673, 59.325487, 13.121650, 16.193501, 5.908047, 5.209185, 149.911673, 0.410707,
    5.840941, 3.739687, 9.714550, 66.197603, 2.134625, 2.607922, 0.070634, 199.839308, 0.281055, 13.475048,
    3.761391, 2.503311, 31.665389, 0.203468, 8.857324, 111.964249, 67.989443, 963.371447, 160.779131, 60.139828, 58.334186
]

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

@app.post("/normalize_variables")
async def normalize_variables(user_input: str = Body(..., embed=True)):
    prompt = (
        "You are a medical assistant. The user will provide a list of variable names and values, possibly unordered and incomplete. "
        "Here is the required variable list, in order:\n"
        + "\n".join([f"{i+1}. {name}" for i, name in enumerate(variable_list)]) +
        "\n\nHere are the mean values for each variable, in order:\n"
        + ", ".join(str(x) for x in mean_values) +
        "\n\nGiven the user's input:\n" + user_input +
        "\n\nReturn a comma-separated list of 30 values in the correct order, using the user's values where available, and the mean for any missing variable. Output only the comma-separated list of 30 values."
    )
    result = llm.invoke([
        {"role": "system", "content": "You are a medical assistant that arranges variables for a machine learning model."},
        {"role": "user", "content": prompt}
    ])
    # Parse the result into a list of floats
    try:
        values = [float(x.strip()) for x in result.split(",") if x.strip()]
        if len(values) != 30:
            return {"error": f"Expected 30 values, got {len(values)}. Output: {result}"}
        return {"variables": values}
    except Exception as e:
        return {"error": f"Could not parse output: {result}. Error: {e}"} 