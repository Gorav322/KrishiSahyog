import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle
import os

def preprocess_crop_data(df):
    df.columns = df.columns.str.strip()
    
    expected_columns = ['State', 'Season', 'Area_2017_18', 'Area_2018_19', 'Area_2019_20', 
                        'Area_2020_21', 'Area_2021_22', 'Production_2017_18', 'Production_2018_19', 
                        'Production_2019_20', 'Production_2020_21', 'Production_2021_22', 
                        'Yield_2017_18', 'Yield_2018_19', 'Yield_2019_20', 'Yield_2020_21', 
                        'Yield_2021_22']
    
    if len(df.columns) <= len(expected_columns):
        df.columns = expected_columns[:len(df.columns)]
    else:
        raise ValueError(f"Column length mismatch: Expected {len(expected_columns)} columns but found {len(df.columns)}.")
    
    df = df.iloc[1:].reset_index(drop=True)
    
    numeric_columns = df.columns[2:]
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
    
    df['Season'] = df['Season'].astype(str).str.strip()
    df['Season'] = df['Season'].replace('nan', '')
    
    return df

def train_models(crop_data):
    models = {}
    for crop_name, crop_info in crop_data.items():
        models[crop_name] = {'models': {}, 'data': crop_info['data']}
        
        available_seasons = crop_info['data']['Season'].str.strip().str.lower().unique()
        print(f"Processing crop: {crop_name}")
        print(f"Available seasons: {available_seasons}")

        for season in available_seasons:
            print(f"Trying to train model for season: {season}")
            season_data = crop_info['data'][crop_info['data']['Season'].str.strip().str.lower() == season]
            
            print(f"Data for season '{season}':")
            print(season_data.head())
            
            if season_data.empty:
                print(f"No data available for season: {season} in crop: {crop_name}")
                continue
            
            X = season_data[['Area_2021_22']].values
            y = season_data['Yield_2021_22'].values
            
            if len(X) > 0:
                model = RandomForestRegressor()
                model.fit(X, y)
                model_filename = f"{crop_name}_{season}_model.pkl"
                model_path = os.path.join('models', model_filename)
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                print(f"Model for {crop_name} ({season}) saved to: {model_path}")
    
    return models

def save_models(models, filename='models.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(models, f)

def main():
    crop_yield_files = {
        'Bajra': r"C:\Users\GORAV\flask_project\data\bajra.csv",
        'Barley': r"C:\Users\GORAV\flask_project\data\barley.csv",
        'Gram': r"C:\Users\GORAV\flask_project\data\gram.csv",
        'Jowar': r"C:\Users\GORAV\flask_project\data\jowar.csv",
        'Kharif CC': r"C:\Users\GORAV\flask_project\data\kharif cc.csv",
        'Kharif Cereals': r"C:\Users\GORAV\flask_project\data\kharif cereals.csv",
        'Kharif Foodgrains': r"C:\Users\GORAV\flask_project\data\kharif foodgrains.csv",
        'Kharif Pulses': r"C:\Users\GORAV\flask_project\data\kharif pulses.csv",
        'Lentil': r"C:\Users\GORAV\flask_project\data\lentil.csv",
        'Maize': r"C:\Users\GORAV\flask_project\data\maize.csv",
        'Moong': r"C:\Users\GORAV\flask_project\data\moong.csv",
        'Rabi CC': r"C:\Users\GORAV\flask_project\data\rabi cc.csv",
        'Rabi Cereals': r"C:\Users\GORAV\flask_project\data\rabi cereals.csv",
        'Rabi Foodgrains': r"C:\Users\GORAV\flask_project\data\rabi foodgrains.csv",
        'Rabi Pulses': r"C:\Users\GORAV\flask_project\data\rabi pulses.csv",
        'Ragi': r"C:\Users\GORAV\flask_project\data\ragi.csv",
        'Small Millets': r"C:\Users\GORAV\flask_project\data\small millets.csv",
        'Total CC': r"C:\Users\GORAV\flask_project\data\total cc.csv",
        'Total Cereals': r"C:\Users\GORAV\flask_project\data\total cereals.csv",
        'Total Foodgrains': r"C:\Users\GORAV\flask_project\data\total foodgrains.csv",
        'Total Pulses': r"C:\Users\GORAV\flask_project\data\total pulses.csv",
        'Tur': r"C:\Users\GORAV\flask_project\data\tur.csv",
        'Urad': r"C:\Users\GORAV\flask_project\data\urad.csv",
        'Rice': r"C:\Users\GORAV\flask_project\data\rice.csv",
        'Wheat': r"C:\Users\GORAV\flask_project\data\wheat.csv",
    }

    crop_data = {}
    for crop_name, file_path in crop_yield_files.items():
        df = pd.read_csv(file_path)
        df = preprocess_crop_data(df)
        
        crop_data[crop_name] = {
            'models': {}, 
            'data': df
        }
    
    models = train_models(crop_data)
    save_models(models)

if __name__ == '__main__':
    main()
