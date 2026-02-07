
from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)


pipeline = pickle.load(open("churn_pipeline.pkl", "rb"))

model = pipeline["model"]
scaler = pipeline["scaler"]
feature_names = pipeline["feature_names"]
encoders = pipeline["encoders"]

@app.route("/")
def home():
    
    return render_template("index.html", features=feature_names)

@app.route("/predict", methods=["POST"])
def predict():
    
    input_dict = {}
    for feature in feature_names:
        input_dict[feature] = request.form.get(feature)

    df_input = pd.DataFrame([input_dict])

    
    for col in df_input.columns:
        if col in encoders:
            
            if df_input[col][0] not in encoders[col].classes_:
                df_input[col][0] = encoders[col].classes_[0]
            df_input[col] = encoders[col].transform(df_input[col])

    
    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    for col in numeric_cols:
        df_input[col] = pd.to_numeric(df_input[col], errors='coerce').fillna(0)

    
    scaled = scaler.transform(df_input)

    
    prediction = model.predict(scaled)
    result = "Customer Will Churn ❌" if prediction[0] == 1 else "Customer Will Stay ✅"

    
    return render_template(
        "index.html",
        features=feature_names,
        prediction_text=result
    )

if __name__ == "__main__":
    app.run(debug=True)
