from flask import Flask, request, render_template, jsonify
from src.pipeline.prediction_pipeline import CustomData, PredictPipeline
import pandas as pd
import os
import sys

application = Flask(__name__)
app = application

@app.route('/')
def home_page():
    return render_template("index.html")

@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == "GET":
        csv_data = pd.read_csv(os.path.join("artifacts", "raw.csv"))
        return render_template("index.html", bike_name=csv_data["bike_name"].unique(), city=csv_data["city"].unique())

    else:
        data = CustomData(
            bike_name=request.form.get("bike_name"),
            city=request.form.get("city"),
            kms_driven=float(request.form.get("kms_driven")),
            owner=int(request.form.get("owner")),
            age=float(request.form.get("age")),
            power=float(request.form.get("power")),
            brand=int(request.form.get("brand"))
        )

        final_data = data.get_data_as_data_frame()
        predict_pipeline = PredictPipeline()
        pred = predict_pipeline.predict(final_data)

        return render_template("results.html", prediction_result=pred)

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
