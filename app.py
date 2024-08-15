import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model
regmodel = pickle.load(open("regmodel.pkl", "rb"))
scalar = pickle.load(open("scaling.pkl", "rb"))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    new_data = scalar.transform(np.array(list(data.values())).reshape(1, -1))
    output = regmodel.predict(new_data)
    return jsonify(output[0])

@app.route("/predict", methods=["POST"])
def predict():
    data = [float(val) for val in request.form.values()]
    final_input = scalar.transform(np.array(data).reshape(1,-1))
    output = regmodel.predict(final_input)[0]
    return render_template("home.html", prediction_text = "The predicted house price is {}".format(output))

if __name__ == "__main__":
    app.run(debug=True)