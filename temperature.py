import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Load temperature data from CSV files
# Replace 'min_temp_file.csv', 'mean_temp_file.csv', 'max_temp_file.csv' with your actual file paths
min_temp_data = pd.read_csv(r"D:\datasets\TEMP_ANNUAL_MIN_1901-2021.csv")
mean_temp_data = pd.read_csv(r"D:\datasets\TEMP_ANNUAL_MEAN_1901-2021.csv")
max_temp_data = pd.read_csv(r"D:\datasets\TEMP_ANNUAL_MAX_1901-2021.csv")

# Combine the datasets into one DataFrame
min_temp_data['Temperature_Type'] = 'Min'
mean_temp_data['Temperature_Type'] = 'Mean'
max_temp_data['Temperature_Type'] = 'Max'

combined_data = pd.concat([min_temp_data, mean_temp_data, max_temp_data])

# Check for missing values in 'YEAR' column
print(combined_data['YEAR'].isna().sum())  # Number of missing values

# Option 1: Drop rows with missing 'YEAR' values
combined_data = combined_data.dropna(subset=['YEAR'])

# Option 2: Fill missing 'YEAR' values with a placeholder (e.g., 1900) if appropriate
# combined_data['YEAR'] = combined_data['YEAR'].fillna(1900)

# Convert 'YEAR' column to integer
combined_data['YEAR'] = combined_data['YEAR'].astype(int)

# Convert 'YEAR' column to datetime, setting the end of year as the date
combined_data['YEAR'] = pd.to_datetime(combined_data['YEAR'].astype(str) + '-12-31')

# Set the 'YEAR' column as index
combined_data.set_index('YEAR', inplace=True)

# Convert 'ANNUAL' column to numeric, forcing errors to NaN
combined_data['ANNUAL'] = pd.to_numeric(combined_data['ANNUAL'], errors='coerce')

# Ensure data is sorted by index
combined_data.sort_index(inplace=True)

# Define ARIMA forecast function
def arima_forecast(data, periods):
    model = ARIMA(data, order=(5, 1, 0))  # Example order; adjust based on your data
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=periods)
    return forecast

# Number of years to forecast
forecast_years = 5
forecast_start_year = 2025

# Filter data for each temperature type and forecast
temperature_types = ['Min', 'Mean', 'Max']

for temp_type in temperature_types:
    temp_data = combined_data[combined_data['Temperature_Type'] == temp_type]['ANNUAL'].dropna()
    
    if len(temp_data) < 2:  # Check if there is enough data to forecast
        print(f"Not enough data to forecast {temp_type} temperatures.")
        continue
    
    # Generate forecast starting from forecast_start_year
    forecast = arima_forecast(temp_data, forecast_years)
    
    # Generate future date range for plotting forecast
    future_dates = pd.date_range(start=pd.Timestamp(forecast_start_year, 12, 31), periods=forecast_years, freq='A')
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(temp_data.index, temp_data, label=f'Historical {temp_type} Temperature')
    plt.plot(future_dates, forecast, label='Forecast', color='red')
    plt.xlabel('Year')
    plt.ylabel('Temperature (°C)')
    plt.title(f'{temp_type} Temperature Forecast')
    plt.legend()
    plt.show()
    
    # Print forecast values
    for year, temp in zip(future_dates.year, forecast):
        print(f"Predicted {temp_type.lower()} temperature for {year}: {temp:.2f} °C")
