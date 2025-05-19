from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)

# Load model
with open('online_food_deliverys.pkl', 'rb') as f:
    classifier = pickle.load(f)

# Mapping dictionary
my_dictionary = {
    'male': 1, 'female': 0,
    'single': 2, 'married': 0, 'prefer not to say': 1,
    'student': 3, 'employee': 0, 'self employeed': 2, 'house wife': 1,
    'no income': 4, '25001 to 50000': 1, 'more than 50000': 3,
    '10001 to 25000': 0, 'below rs.10000': 2,
    'graduate': 0, 'post graduate': 2, 'ph.d': 1,
    'school': 3, 'uneducated': 4,
    'yes': 1, 'no': 0,
    'positive': 1, 'negative': 0
}

def salary_converter(my_sal):
    try:
        sal = int(my_sal)
        if sal < 10000: return 2
        elif 10001 <= sal <= 25000: return 0
        elif 25001 <= sal <= 50000: return 1
        else: return 3
    except: 
        return 2  # default value

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Convert all values
        processed = [
            int(data['age']),
            my_dictionary[data['gender'].lower()],
            my_dictionary[data['marital_status'].lower()],
            my_dictionary[data['occupation'].lower()],
            salary_converter(data['monthly_income']),
            my_dictionary[data['education'].lower()],
            int(data['family_size']),
           
            my_dictionary[data['output'].lower()]
        ]

        prediction = classifier.predict([processed])[0]
        return jsonify({'result': 'POSITIVE' if prediction == 1 else 'NEGATIVE'})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=False)
