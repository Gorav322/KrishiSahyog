import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import numpy as np

# Function to forecast using ARIMA
def arima_forecast(data, periods):
    model = ARIMA(data, order=(5, 1, 0))  # Example order; adjust based on your data
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=periods)
    return forecast

# Define thresholds for classification based on average rainfall
def get_thresholds(avg_rainfall):
    drought_threshold = 0.8 * avg_rainfall  # Less than 80% of average = drought
    heavy_rain_threshold = 1.5 * avg_rainfall  # More than 150% of average = heavy rainfall
    return drought_threshold, heavy_rain_threshold

# Load the data from CSV
def load_rainfall_data(file_path):
    df = pd.read_csv(file_path)
    return df

# Prepare the rainfall data for the ARIMA model
def prepare_rainfall_data(df):
    # Assuming columns: 'SUBDIVISION', 'YEAR', 'ANNUAL'
    rainfall_data = {}
    for _, row in df.iterrows():
        state = row['SUBDIVISION']
        year = row['YEAR']
        annual_rainfall = row['ANNUAL']
        
        if state not in rainfall_data:
            rainfall_data[state] = {}
        rainfall_data[state][year] = annual_rainfall
    return rainfall_data

# Calculate average rainfall for each region
def calculate_average_rainfall(rainfall_data):
    average_rainfall_data = {}
    for state, data in rainfall_data.items():
        average_rainfall = np.mean(list(data.values()))
        average_rainfall_data[state] = average_rainfall
    return average_rainfall_data

# Function to classify and recommend actions based on predicted rainfall
def generate_friendly_predictions(state, years, rainfall_data, average_rainfall_data):
    # Check if the state is available in the data
    if state not in rainfall_data:
        print(f"Rainfall data not available for {state}.")
        return
    
    historical_data = list(rainfall_data[state].values())
    avg_rainfall = average_rainfall_data[state]
    drought_threshold, heavy_rain_threshold = get_thresholds(avg_rainfall)
    
    # Forecast future rainfall starting from 2025
    forecast = arima_forecast(historical_data, years)
    forecast_years = range(2025, 2025 + years)
    
    for i, year in enumerate(forecast_years):
        rain = forecast[i]

        # Classify the rainfall
        if rain < drought_threshold:
            classification = "Drought"
            recommendation = "Consider using drought-resistant crops and irrigation systems."
        elif rain > heavy_rain_threshold:
            classification = "Heavy Rainfall"
            recommendation = "Prepare for potential waterlogging. Improve drainage systems or plant water-tolerant crops."
        else:
            classification = "Normal Rainfall"
            recommendation = "Follow standard practices for the growing season."

        # Print out the farmer-friendly prediction
        print(f"**{year} Prediction for {state}**: {classification} with {rain:.2f} mm of rainfall.")
        print(f"Recommendation: {recommendation}")
        print("\n")

# Load and prepare the data
file_path = r"C:\Users\GORAV\Downloads\Sub_Division_IMD_2017.csv"  # Update with your actual file path
df = load_rainfall_data(file_path)
rainfall_data = prepare_rainfall_data(df)
average_rainfall_data = calculate_average_rainfall(rainfall_data)

# Input from user
state_input = input("Enter the state: ")
years_input = int(input("Enter the number of years for prediction (e.g., 2): "))

# Generate predictions for the input state and years
generate_friendly_predictions(state_input, years_input, rainfall_data, average_rainfall_data)
