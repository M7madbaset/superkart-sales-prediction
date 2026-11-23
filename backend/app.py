
# Import necessary libraries
import numpy as np
import joblib  # For loading the serialized model
import pandas as pd  # For data manipulation
from flask import Flask, request, jsonify  # For creating the Flask API

# Initialize Flask app with a name
superkart_api = Flask("SuperKart")

# Load the serialized pipeline (pre-processing + model) saved in the notebook
model = joblib.load("superkart_model.joblib")

# The 10 input features the model expects (same names as used for training)
REQUIRED_FEATURES = [
    'Product_Weight', 'Product_Sugar_Content', 'Product_Allocated_Area', 'Product_MRP',
    'Store_Size', 'Store_Location_City_Type', 'Store_Type',
    'Product_Id_char', 'Store_Age_Years', 'Product_Type_Category',
]

# Define a route for the home page (health check)
@superkart_api.get('/')
def home():
    return "Welcome to the SuperKart System"

# Define an endpoint to predict sales for a single product-store record
@superkart_api.post('/v1/predict')
def predict_sales():
    try:
        # Get JSON data from the request (silent=True -> None instead of an exception on bad JSON)
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({'error': 'Request body must be a valid JSON object'}), 400

        # Validate that every required feature is present
        missing = [f for f in REQUIRED_FEATURES if f not in data]
        if missing:
            return jsonify({'error': 'Missing required fields', 'missing_fields': missing}), 400

        # Convert the input into a one-row DataFrame and predict
        input_data = pd.DataFrame([{f: data[f] for f in REQUIRED_FEATURES}])
        prediction = float(model.predict(input_data)[0])

        # Return the prediction as a JSON response
        return jsonify({'Sales': prediction})

    except Exception as e:
        return jsonify({'error': 'Prediction failed', 'details': str(e)}), 500

# Define an endpoint to predict sales for a batch of records (CSV upload)
@superkart_api.post('/v1/predictbatch')
def predict_sales_batch():
    try:
        # Get the uploaded CSV file from the request
        if 'file' not in request.files:
            return jsonify({'error': "No file uploaded. Send a CSV in the 'file' form field"}), 400

        # Read the file into a DataFrame
        input_data = pd.read_csv(request.files['file'])

        # Validate the columns; extra columns (e.g. Product_Id, Store_Id) are ignored
        missing = [f for f in REQUIRED_FEATURES if f not in input_data.columns]
        if missing:
            return jsonify({'error': 'CSV is missing required columns', 'missing_columns': missing}), 400
        input_data = input_data[REQUIRED_FEATURES]

        # Make predictions for the batch data
        predictions = model.predict(input_data).tolist()

        # Output: row index -> predicted sales
        return jsonify({str(i): round(pred, 2) for i, pred in enumerate(predictions)})

    except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
        return jsonify({'error': 'Could not read the uploaded CSV', 'details': str(e)}), 400
    except Exception as e:
        return jsonify({'error': 'Batch prediction failed', 'details': str(e)}), 500


# Run the Flask app (the Docker image uses gunicorn instead of this dev server)
if __name__ == '__main__':
    superkart_api.run(debug=True)
