# from flask import Flask
# app = Flask(__name__)

# @app.route('/')
# def home():
#     return 'Hello, World!'

# if __name__ == '__main__':
#     app.run()
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import json
import numpy as np

app = Flask(__name__)
CORS(app)

# load the model from the saved file
model = joblib.load('path_to_your_model.pkl',trusted=True)

@app.route('/predict', methods=['POST'])
def predict():
    
    from_data = request.get_json(force=True)
    data = pd.DataFrame(from_data)
    # assumes you're passing in a JSON object with a single field called 'features'
    prediction = model.predict(data)
    loan_status_labels = {
        0: 'Not Approved',
        1: 'Approved'
    }
    predicted_loan_status = loan_status_labels[prediction[0]]

    return {"prediction":predicted_loan_status}
    

if __name__ == '__main__':
    app.run(debug=False)
