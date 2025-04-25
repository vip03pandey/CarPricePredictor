from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)


car = pd.read_csv('Cleaned Car.csv')


with open('car.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def index():
    companies = sorted(car['company'].unique())
    car_models = sorted(car['name'].unique())
    years = sorted(car['year'].unique(), reverse=True)
    fuel_types = sorted(car['fuel_type'].unique())
    return render_template('index.html', companies=companies, car_models=car_models, years=years, fuel_types=fuel_types)

@app.route('/predict', methods=['POST'])
def predict():
    try:

        company = request.form['company']
        car_model = request.form['car_model']
        year = int(request.form['year'])
        fuel_type = request.form['fuel_type']
        kms_driven = float(request.form['kms_driven'])


        input_data = pd.DataFrame({
            'company': [company],
            'name': [car_model],
            'year': [year],
            'fuel_type': [fuel_type],
            'kms_driven': [kms_driven]
        })


        predicted_price = model.predict(input_data)[0]


        predicted_price = round(predicted_price, 2)

        return jsonify({'prediction': predicted_price, 'status': 'success'})
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'})

if __name__ == '__main__':
    app.run(debug=True)