from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)  # Enable CORS to allow cross-origin requests from your frontend

# Load the model and scaler (only once when the app starts)
model = joblib.load('monkeypox_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/predict', methods=['POST'])
def predict_monkeypox():
    try:
        # Get the JSON data from the request
        data = request.get_json()
        
        # Check if the symptoms field is in the incoming data
        if 'symptoms' not in data:
            return jsonify({"error": "Missing symptoms data"}), 400
        
        symptoms = data['symptoms']
        
        # Define expected keys for symptoms
        expected_keys = [
            'Systemic Illness', 'Rectal Pain', 'Sore Throat', 'Penile Oedema',
            'Oral Lesions', 'Solitary Lesion', 'Swollen Tonsils', 'HIV Infection',
            'Sexually Transmitted Infection'
        ]
        
        # Check if all expected keys are in symptoms
        if not all(key in symptoms for key in expected_keys):
            return jsonify({"error": "Missing expected symptoms keys"}), 400
        
        # Convert the symptoms to a DataFrame (similar to how you trained the model)
        symptoms_data = pd.DataFrame([symptoms])
        
        # Scale the symptoms data using the pre-trained scaler
        symptoms_scaled = scaler.transform(symptoms_data)
        
        # Predict using the model
        prediction = model.predict(symptoms_scaled)
        
        # Return the prediction result
        result = 'Positive for monkeypox' if prediction[0] == 1 else 'Negative for monkeypox'
        
        return jsonify({'result': result})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500  # Internal Server Error

if __name__ == '__main__':
    app.run(debug=True)
