from flask import Flask, request, jsonify
import joblib
import numpy as np

# ✅ Load model and encoder once
model = joblib.load('mental_health_model.pkl')
le_target = joblib.load('le_target.pkl')

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return "Mental Health Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    try:
        # Extract features from input
        age = int(data['Age'])
        gender = int(data['Gender_encoded'])
        treatment = int(data['treatment_encoded'])
        work_interfere = int(data['work_interfere_encoded'])

        # Make prediction
        features = np.array([[age, gender, treatment, work_interfere]])
        prediction = model.predict(features)
        label = le_target.inverse_transform(prediction)

        return jsonify({
            # 'mental_health_consequence_encoded': int(prediction[0]),
            'Mental_health_unstable': label[0]
        })

    except Exception as e:
        return jsonify({'error': str(e)})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
