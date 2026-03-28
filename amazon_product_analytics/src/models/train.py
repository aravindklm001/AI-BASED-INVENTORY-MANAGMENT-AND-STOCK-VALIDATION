import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score
import joblib
import os

class ModelTrainer:
    def __init__(self):
        self.reg_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        self.clf_model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
        self.features = [
            'discounted_price', 'actual_price', 'discount_percentage',
            'category_encoded', 'price_band'
        ]
        # Text features will be added dynamically if present
        
    def _prepare_data(self, df):
        # Add tfidf columns if they exist
        tfidf_cols = [c for c in df.columns if c.startswith('tfidf_about_')]
        use_features = self.features + tfidf_cols + ['sentiment_score', 'engagement_score']
        
        # Ensure only available columns exist
        available_features = [f for f in use_features if f in df.columns]
        
        X = df[available_features].fillna(0)
        y_reg = df['demand_score']
        
        # Success = demand_score > median
        median_demand = y_reg.median()
        y_clf = (y_reg > median_demand).astype(int)
        
        return X, y_reg, y_clf, available_features

    def train_and_evaluate(self, df):
        from sklearn.model_selection import RandomizedSearchCV

        if 'demand_score' not in df.columns:
            raise ValueError("demand_score must be computed before training.")
            
        X, y_reg, y_clf, self.trained_features = self._prepare_data(df)
        
        # Split Data
        X_train, X_test, yr_train, yr_test, yc_train, yc_test = train_test_split(
            X, y_reg, y_clf, test_size=0.2, random_state=42
        )
        
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 1.0]
        }

        # Train Regression Model with Tuning
        print("Tuning & Training Demand Estimation (Regression) Model...")
        reg_search = RandomizedSearchCV(self.reg_model, param_distributions=param_grid, n_iter=5, cv=3, random_state=42)
        reg_search.fit(X_train, yr_train)
        self.reg_model = reg_search.best_estimator_
        
        reg_preds = self.reg_model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(yr_test, reg_preds))
        
        # Train Classification Model with Tuning
        print("Tuning & Training Success Prediction (Classification) Model...")
        clf_search = RandomizedSearchCV(self.clf_model, param_distributions=param_grid, n_iter=5, cv=3, random_state=42)
        clf_search.fit(X_train, yc_train)
        self.clf_model = clf_search.best_estimator_
        
        clf_preds = self.clf_model.predict(X_test)
        acc = accuracy_score(yc_test, clf_preds)
        f1 = f1_score(yc_test, clf_preds)
        
        metrics = {
            'rmse': rmse,
            'accuracy': acc,
            'f1_score': f1,
            'best_reg_params': reg_search.best_params_,
            'best_clf_params': clf_search.best_params_
        }
        
        return metrics

    def save_models(self, save_dir='models'):
        os.makedirs(save_dir, exist_ok=True)
        joblib.dump(self.reg_model, os.path.join(save_dir, 'demand_regressor.pkl'))
        joblib.dump(self.clf_model, os.path.join(save_dir, 'success_classifier.pkl'))
        joblib.dump(self.trained_features, os.path.join(save_dir, 'model_features.pkl'))
        print(f"Models saved successfully to {save_dir}")
