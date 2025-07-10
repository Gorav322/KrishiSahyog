import pandas as pd
from sklearn.linear_model import LinearRegression

def preprocess_crop_data(df):
    # Remove extra whitespace characters from column names
    df.columns = df.columns.str.strip()
    
    # Define the expected columns
    expected_columns = ['State', 'Season', 'Area_2017_18', 'Area_2018_19', 'Area_2019_20', 
                        'Area_2020_21', 'Area_2021_22', 'Production_2017_18', 'Production_2018_19', 
                        'Production_2019_20', 'Production_2020_21', 'Production_2021_22', 
                        'Yield_2017_18', 'Yield_2018_19', 'Yield_2019_20', 'Yield_2020_21', 
                        'Yield_2021_22']
    
    # Adjust column names according to the dataset structure
    if len(df.columns) <= len(expected_columns):
        df.columns = expected_columns[:len(df.columns)]
    else:
        raise ValueError(f"Column length mismatch: Expected {len(expected_columns)} columns but found {len(df.columns)}.")
    
    df = df.iloc[1:].reset_index(drop=True)
    
    # Convert columns to numeric, handle conversion warnings
    for col in df.columns[2:]:
        df.loc[:, col] = pd.to_numeric(df[col], errors='coerce')
    
    # Fill missing values
    df.loc[:, df.columns[2:]] = df.fillna(df.median(numeric_only=True))
    
    # ETelanganansure 'Season' column is treated as string
    df.loc[:, 'Season'] = df['Season'].astype(str).str.strip()
    df.loc[:, 'Season'] = df['Season'].replace('nan', '')
    
    return df

def train_models(crop_data):
    models = {}
    for crop_name, df in crop_data.items():
        models[crop_name] = {}
        seasons = df['Season'].unique()
        for season in seasons:
            season_data = df[df['Season'].str.strip().str.lower() == season.strip().lower()]
            X = season_data[['Area_2021_22']].values  # Features
            y = season_data['Yield_2021_22'].values   # Target
            
            if len(X) > 0:  # Check if there's data for the season
                model = LinearRegression()
                model.fit(X, y)
                models[crop_name][season] = model
    return models

def predict_yield(model, area, crop_name):
    # Predict yield
    prediction = model.predict([[area]])
    return prediction[0]

def predict_yield_for_state(state_name, land_size, models, crop_data):
    state_name = state_name.strip().lower()  # Normalize state name
    
    for crop_name, data in crop_data.items():
        data['State'] = data['State'].str.strip().str.lower()  # Normalize state names in the data
        state_data = data[data['State'] == state_name]
        
        if state_data.empty:
            print(f"No data available for state: {state_name} in crop: {crop_name}")
            continue
        
        print(f"\n{crop_name} yield predictions for {state_name}:")
        for season in state_data['Season'].unique():
            season_data = state_data[state_data['Season'].str.strip().str.lower() == season.strip().lower()]
            if season in models[crop_name]:
                model = models[crop_name][season]
                yield_prediction = predict_yield(model, land_size, crop_name)
                print(f"Season: {season}, Predicted yield for {crop_name} with land size {land_size}: {yield_prediction} quintals")
            else:
                print(f"No model available for season: {season} for crop: {crop_name}")

def main():
    # Load and preprocess crop data
    crop_yield_files = {
        'bajra': r"C:\Users\GORAV\flask_project\data\bajra.csv",
    'barley': r"C:\Users\GORAV\flask_project\data\barley.csv",
    'gram': r"C:\Users\GORAV\flask_project\data\gram.csv",
    'jowar': r"C:\Users\GORAV\flask_project\data\jowar.csv",
    'kharif cc': r"C:\Users\GORAV\flask_project\data\kharif cc.csv",
    'kharif cereals': r"C:\Users\GORAV\flask_project\data\kharif cereals.csv",
    'kharif foodgrains': r"C:\Users\GORAV\flask_project\data\kharif foodgrains.csv",
    'kharif pulses': r"C:\Users\GORAV\flask_project\data\kharif pulses.csv",
    'lentil': r"C:\Users\GORAV\flask_project\data\lentil.csv",
    'maize': r"C:\Users\GORAV\flask_project\data\maize.csv",
    'moong': r"C:\Users\GORAV\flask_project\data\moong.csv",
    'rabi cc': r"C:\Users\GORAV\flask_project\data\rabi cc.csv",
    'rabi cereals': r"C:\Users\GORAV\flask_project\data\rabi cereals.csv",
    'rabi foodgrains': r"C:\Users\GORAV\flask_project\data\rabi foodgrains.csv",
    'rabi pulses': r"C:\Users\GORAV\flask_project\data\rabi pulses.csv",
    'ragi': r"C:\Users\GORAV\flask_project\data\ragi.csv",
    'small millets': r"C:\Users\GORAV\flask_project\data\small millets.csv",
    'total cc': r"C:\Users\GORAV\flask_project\data\total cc.csv",
    'total cereals': r"C:\Users\GORAV\flask_project\data\total cereals.csv",
    'total foodgrains': r"C:\Users\GORAV\flask_project\data\total foodgrains.csv",
    'total pulses': r"C:\Users\GORAV\flask_project\data\total pulses.csv",
    'tur': r"C:\Users\GORAV\flask_project\data\tur.csv",
    'urad': r"C:\Users\GORAV\flask_project\data\urad.csv",
    'rice': r"C:\Users\GORAV\flask_project\data\rice.csv",
    'wheat': r"C:\Users\GORAV\flask_project\data\wheat.csv",
    }

    crop_data = {}
    for crop_name, file_path in crop_yield_files.items():
        df = pd.read_csv(file_path)
        crop_data[crop_name] = preprocess_crop_data(df)

    # Train models
    models = train_models(crop_data)

    # Get state name and land size from user
    state_name = input("Enter the state name: ").strip().lower()
    land_size = float(input("Enter the land size (in hectares): ").strip())
    
    # Predict yields for the specified state
    predict_yield_for_state(state_name, land_size, models, crop_data)

if __name__ == "__main__":
    main()