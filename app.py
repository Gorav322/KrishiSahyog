from flask import Flask, request, jsonify, render_template
import pickle
import os
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import io
import base64
import numpy as np
import matplotlib

# Set matplotlib to use Agg backend for server environments
matplotlib.use('Agg')

app = Flask(__name__)

# Route to render HTML page
@app.route('/')
def index():
    return render_template('Aakashvaani.html')

# Define ARIMA forecast function
def arima_forecast(data, periods):
    model = ARIMA(data, order=(5, 1, 0))  # You can tweak the (p,d,q) values for your dataset
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=periods)
    return forecast

# Crop prediction route
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Changed to read JSON
    crop = data.get('crop_name')  
    state = data.get('state_name')
    area = data.get('area')

    try:
        area = float(area)
    except ValueError:
        return jsonify({'error': 'Invalid area value. Must be a number.'})

    seasons = ['kharif', 'rabi', 'total']
    predictions = {}

    for season in seasons:
        model_path = f'models/{crop}_{season}_model.pkl'
        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as model_file:
                    model = pickle.load(model_file)
                prediction = model.predict([[area]])[0]
                predictions[season] = round(prediction * 10, 2)  # Convert tons to quintals
            except Exception as e:
                predictions[season] = f'Error loading model: {str(e)}'
        else:
            predictions[season] = 'Model not found'

    if not predictions:
        return jsonify({'error': f'No models found for crop: {crop}'})

    return jsonify({'predictions': predictions})

# Temperature forecast route
@app.route('/temperature_forecast', methods=['POST'])
def temperature_forecast():
    forecast_years = 5
    try:
        # Load temperature data
        min_temp_data = pd.read_csv(r"D:\datasets\TEMP_ANNUAL_MIN_1901-2021.csv")
        mean_temp_data = pd.read_csv(r"D:\datasets\TEMP_ANNUAL_MEAN_1901-2021.csv")
        max_temp_data = pd.read_csv(r"D:\datasets\TEMP_ANNUAL_MAX_1901-2021.csv")

        # Combine the datasets into one DataFrame
        min_temp_data['Temperature_Type'] = 'Min'
        mean_temp_data['Temperature_Type'] = 'Mean'
        max_temp_data['Temperature_Type'] = 'Max'

        combined_data = pd.concat([min_temp_data, mean_temp_data, max_temp_data])

        # Clean and prepare data
        combined_data = combined_data.dropna(subset=['YEAR'])
        combined_data['YEAR'] = combined_data['YEAR'].astype(int)
        combined_data['YEAR'] = pd.to_datetime(combined_data['YEAR'].astype(str) + '-12-31')
        combined_data.set_index('YEAR', inplace=True)
        combined_data['ANNUAL'] = pd.to_numeric(combined_data['ANNUAL'], errors='coerce')
        combined_data.sort_index(inplace=True)

        temperature_types = ['Min', 'Mean', 'Max']
        forecast_results = {}
        img_list = []

        for temp_type in temperature_types:
            temp_data = combined_data[combined_data['Temperature_Type'] == temp_type]['ANNUAL'].dropna()

            if len(temp_data) < 2:
                forecast_results[temp_type] = 'Not enough data to forecast.'
                continue

            forecast = arima_forecast(temp_data, forecast_years)
            future_dates = pd.date_range(start='2024', periods=forecast_years, freq='A')

            forecast_results[temp_type] = dict(zip(future_dates.year, forecast))

            # Plot the forecast and historical data
            plt.figure(figsize=(10, 6))
            plt.plot(temp_data.index, temp_data.values, label=f'Historical {temp_type} Temperatures')
            plt.plot(future_dates, forecast, label=f'{forecast_years}-Year Forecast')
            plt.title(f'{temp_type} Temperature Forecast')
            plt.xlabel('Year')
            plt.ylabel('Temperature (°C)')
            plt.legend()

            # Save plot to base64 string
            img = io.BytesIO()
            plt.savefig(img, format='png')
            img.seek(0)
            img_base64 = base64.b64encode(img.getvalue()).decode('utf8')
            img_list.append(f'<img src="data:image/png;base64,{img_base64}" alt="{temp_type} Temperature Forecast"/>')
            plt.close()

        return jsonify({'forecast': forecast_results, 'images': img_list})

    except Exception as e:
        return jsonify({'error': str(e)})

# Rainfall forecast route
@app.route('/rainfall_forecast', methods=['POST'])
def rainfall_forecast():
    data = request.json  # Changed to read JSON
    state = data.get('state_name')
    forecast_years = 5

    try:
        # Load rainfall data
        rainfall_data = pd.read_csv(r"C:\Users\GORAV\flask_project\data\Sub_Division_IMD_2017.csv")
        state_data = rainfall_data[rainfall_data['SUBDIVISION'].str.contains(state, case=False)]

        if state_data.empty:
            return jsonify({'error': f'No data found for state: {state}'})

        state_data = state_data[['YEAR', 'ANNUAL']]
        state_data.dropna(inplace=True)
        state_data['YEAR'] = pd.to_datetime(state_data['YEAR'].astype(str) + '-12-31')
        state_data.set_index('YEAR', inplace=True)

        # Forecast rainfall using ARIMA
        forecast = arima_forecast(state_data['ANNUAL'], forecast_years)
        future_years = pd.date_range(start='2024', periods=forecast_years, freq='A')
        forecast_results = {}

        for year, rainfall in zip(future_years.year, forecast):
            if rainfall < 750:
                classification = "Below Normal"
                recommendation = "Consider drought-resistant crops."
            elif rainfall < 1000:
                classification = "Normal"
                recommendation = "Proceed with normal crop choices."
            else:
                classification = "Above Normal"
                recommendation = "Good conditions for water-intensive crops."

            forecast_results[year] = {
                'rainfall': round(rainfall, 2),
                'classification': classification,
                'recommendation': recommendation
            }

        return jsonify({'forecast': forecast_results})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)