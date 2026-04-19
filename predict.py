import joblib
import pandas as pd
import numpy as np
import os

class AnomalyPredictor:
    def __init__(self, model_dir='model'):
        self.model_path = os.path.join(model_dir, 'rf_model.pkl')
        self.scaler_path = os.path.join(model_dir, 'scaler.pkl')
        self.region_enc_path = os.path.join(model_dir, 'region_encoder.pkl')
        self.dwelling_enc_path = os.path.join(model_dir, 'dwelling_encoder.pkl')

        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            self.region_enc = joblib.load(self.region_enc_path)
            self.dwelling_enc = joblib.load(self.dwelling_enc_path)
        else:
            self.model = None
            print(f"Warning: Model artifacts not found in {model_dir}")

    def preprocess(self, df):
        df = df.copy()
        
        # 1. Clean energy columns (Regex)
        for col in ['Expected_Energy(kwh)', 'Actual_Energy(kwh)']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace(r'[^0-9.]', '', regex=True)
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # 2. Encoding
        if 'Region_Code' in df.columns:
            # Handle unseen labels by mapping to a default if they aren't in classes_
            df['Region_Code'] = df['Region_Code'].astype(str).apply(
                lambda x: self.region_enc.transform([x])[0] if x in self.region_enc.classes_ else -1
            )
        if 'Dwelling_Type' in df.columns:
            df['Dwelling_Type'] = df['Dwelling_Type'].astype(str).apply(
                lambda x: self.dwelling_enc.transform([x])[0] if x in self.dwelling_enc.classes_ else -1
            )

        # 3. Feature Engineering
        df['Deviation_Abs'] = abs(df['Actual_Energy(kwh)'] - df['Expected_Energy(kwh)'])
        df['Usage_Ratio'] = np.where(df['Expected_Energy(kwh)'] != 0, 
                                    df['Actual_Energy(kwh)'] / df['Expected_Energy(kwh)'], 0)
        df['Load_Utilization'] = np.where(df['Connected_Load(kw)'] != 0, 
                                         df['Actual_Energy(kwh)'] / df['Connected_Load(kw)'], 0)

        df.replace([np.inf, -np.inf], 0, inplace=True)
        df.fillna(0, inplace=True)

        # 4. Feature Selection (Match Training)
        features = [
            'Region_Code', 'Dwelling_Type', 'Num_Occupants', 'House_Area (sqft)', 
            'Appliance_Score', 'Connected_Load(kw)', 'Temperature_C', 'Humidity (%)',
            'Deviation_Abs', 'Usage_Ratio', 'Load_Utilization'
        ]
        
        # Add missing columns as 0
        for f in features:
            if f not in df.columns:
                df[f] = 0
                
        # Scale
        return self.scaler.transform(df[features])

    def predict(self, input_data):
        if self.model is None:
            raise Exception("Model not loaded")

        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        else:
            df = pd.DataFrame(input_data)

        X = self.preprocess(df)
        predictions = self.model.predict(X)
        
        return predictions.tolist()
