import pandas as pd

def load_msp_data(file_path):
    return pd.read_csv(file_path)

def get_msp(msp_data, crop, variety, season):
    crop_data = msp_data[(msp_data['Commodity'] == crop) & (msp_data['Variety'].fillna('NA') == variety)]
    if crop_data.empty:
        return "No MSP provided for this crop"
    return crop_data[season].values[0]

def calculate_msp(crop, predictions, msp_file_path):
    msp_data = load_msp_data(msp_file_path)
    msp_results = {}
    for season_key in ['kharif', 'rabi', 'total']:
        if season_key in predictions:
            yield_value = predictions[season_key]
            variety = 'Common'  # Use appropriate variety or make dynamic
            msp_value = get_msp(msp_data, crop, variety, '2023-24')  # Update year if needed
            msp_results[season_key] = {
                'predicted_yield': yield_value,
                'predicted_msp': msp_value
            }
        else:
            msp_results[season_key] = 'No MSP data available'
    return msp_results
