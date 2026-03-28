import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class ProductInsights:
    def __init__(self):
        self.scaler = MinMaxScaler()
        
    def generate_insights(self, df):
        """Generates analytics insights and classification for products."""
        df = df.copy()
        
        # Initialize default columns to avoid KeyError later
        if 'demand_score' not in df.columns:
            df['demand_score'] = 0.0
        if 'product_classification' not in df.columns:
            df['product_classification'] = 'Unclassified'
        
        # Ensure minimum columns for scoring exist, otherwise return with defaults
        # For sales data, we check for rating_count (quantity) and rating
        required = ['rating_count', 'rating']
        if not all(col in df.columns for col in required):
            return df
            
        # We need to scale these to combine them
        df['norm_rc'] = self.scaler.fit_transform(df[['rating_count']].fillna(0))
        df['norm_r'] = self.scaler.fit_transform(df[['rating']].fillna(0))
        
        # sentiment is already -1 to 1, let's map it to 0-1, handle missing sentiment
        if 'sentiment_score' not in df.columns:
            df['sentiment_score'] = 0.0
        df['norm_s'] = (df['sentiment_score'] + 1.0) / 2.0
        
        # Ensure discount percentage exists
        if 'discount_percentage' not in df.columns:
            df['discount_percentage'] = 0.0
        
        # Demand estimation proxy
        # Rating count (45%), Rating (30%), Sentiment (15%), Discount (10%)
        df['demand_score'] = (df['norm_rc'] * 0.45) + (df['norm_r'] * 0.3) + (df['norm_s'] * 0.15) + (df['discount_percentage'] * 0.1)
        
        # Clean up temporary norm cols
        df = df.drop(columns=['norm_rc', 'norm_r', 'norm_s'], errors='ignore')
        
        # Classification logic based on medians
        rc_median = df['rating_count'].median()
        rating_median = df['rating'].median()
        
        def classify_product(row):
            if row['rating'] >= rating_median and row['rating_count'] >= rc_median:
                return 'High Demand'
            elif row['rating'] < rating_median and row['discount_percentage'] > 0.5:
                return 'Risky Product'
            elif row['rating_count'] < rc_median and row['rating'] < rating_median:
                return 'Low Demand'
            else:
                return 'Medium Demand'
                
        df['product_classification'] = df.apply(classify_product, axis=1)
        
        return df
