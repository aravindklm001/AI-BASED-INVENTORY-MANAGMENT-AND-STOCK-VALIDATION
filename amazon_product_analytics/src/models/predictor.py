import joblib
import os
import pandas as pd
import numpy as np
import re

class NewProductPredictor:
    def __init__(self, models_dir='models'):
        try:
            self.reg_model = joblib.load(os.path.join(models_dir, 'demand_regressor.pkl'))
            self.clf_model = joblib.load(os.path.join(models_dir, 'success_classifier.pkl'))
            self.features = joblib.load(os.path.join(models_dir, 'model_features.pkl'))
        except Exception as e:
            self.reg_model = None
            self.clf_model = None
            self.features = None
            
    def predict_success(self, df_processed):
        """
        Takes a preprocessed single-row dataframe (via DataCleaner + FeatureBuilder)
        and predicts success probability and expected demand score.
        """
        if self.reg_model is None or self.clf_model is None:
            return {"error": "Models not loaded. Train models first."}
            
        # Ensure all required features are present, fill missing with 0
        X = pd.DataFrame(columns=self.features)
        for f in self.features:
            if f in df_processed.columns:
                X[f] = df_processed[f].values
            else:
                X[f] = 0
                
        # Fillnans to be safe
        X = X.fillna(0)
        
        expected_demand = float(self.reg_model.predict(X)[0])
        success_prob = float(self.clf_model.predict_proba(X)[0][1]) * 100.0 # class 1 probability
        
        # Risk assessment
        if success_prob > 75:
            risk = "Low Risk"
        elif success_prob >= 50:
            risk = "Medium Risk"
        else:
            risk = "High Risk"
            
        return {
            "expected_demand_score": round(expected_demand, 3),
            "success_probability": round(success_prob, 2),
            "risk_level": risk
        }

    def get_recommendations(self, df_raw, cleaner, builder):
        """
        Analyzes how changes in rating and price could improve success probability.
        Returns recommended targets.
        """
        if self.reg_model is None or self.clf_model is None:
            return {}
            
        # Get current result
        df_clean = cleaner.clean_data(df_raw)
        df_features = builder.build_features(df_clean, is_training=False)
        current_res = self.predict_success(df_features)
        
        if current_res.get("risk_level") == "Low Risk":
            return {"status": "already_optimal", "message": "Product is already in the Low Risk / High Success category!"}
            
        recommendations = {}
        curr_prob = float(current_res['success_probability'])
        curr_risk = current_res['risk_level']
        
        # 1. Target Rating Search
        temp_df = df_raw.copy()
        try:
            start_rating = float(df_raw['rating'].iloc[0])
            for r in np.arange(start_rating + 0.1, 5.1, 0.1):
                temp_df['rating'] = str(round(float(r), 1))
                res = self.predict_success(builder.build_features(cleaner.clean_data(temp_df), is_training=False))
                if res['risk_level'] != curr_risk or res['success_probability'] > curr_prob + 5:
                    recommendations['target_rating'] = round(float(r), 1)
                    break
            
            # Heuristic fallback if ML is too insensitive
            if 'target_rating' not in recommendations:
                recommendations['target_rating'] = 4.2 if start_rating < 4.2 else 4.5
        except Exception:
            pass
        
        # 2. Target Price Search
        temp_df = df_raw.copy()
        price_str = str(df_raw['actual_price'].iloc[0])
        try:
            p_val = re.sub(r'[^\d.]', '', price_str)
            start_price = float(p_val) if p_val else 1000.0
            for p in np.arange(start_price * 0.95, start_price * 0.5, -start_price * 0.05):
                temp_df['actual_price'] = f"₹{int(p)}"
                temp_df['discounted_price'] = f"₹{int(float(p) * 0.8)}"
                res = self.predict_success(builder.build_features(cleaner.clean_data(temp_df), is_training=False))
                if res['risk_level'] != curr_risk or res['success_probability'] > curr_prob + 5:
                    recommendations['target_price'] = int(p)
                    break
            
            # Heuristic fallback
            if 'target_price' not in recommendations:
                recommendations['target_price'] = int(start_price * 0.8) # Suggest 20% cut as benchmark
        except Exception:
            pass
            
        return recommendations
