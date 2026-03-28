import pandas as pd
import numpy as np
from src.nlp.sentiment import SentimentAnalyzer

class DataCleaner:
    def __init__(self):
        self.sentiment_analyzer = SentimentAnalyzer()

    def _convert_price(self, val):
        if pd.isna(val):
            return np.nan
        # More aggressive: remove anything that is not a digit or a dot
        import re
        val_str = str(val)
        val_clean = re.sub(r'[^\d.]', '', val_str)
        try:
            return float(val_clean) if val_clean else np.nan
        except ValueError:
            return np.nan

    def _convert_percentage(self, val):
        if pd.isna(val):
            return np.nan
        import re
        val_str = str(val)
        val_clean = re.sub(r'[^\d.]', '', val_str)
        try:
            return float(val_clean) / 100.0 if val_clean else np.nan
        except ValueError:
            return np.nan

    def clean_data(self, df):
        df = df.copy()
        
        # 0. Automatic Mapping for Sales/Order Format
        mapping = {
            'ProductID': 'product_id',
            'ProductName': 'product_name',
            'Category': 'category',
            'UnitPrice': 'actual_price',
            'Quantity': 'quantity',
            'Brand': 'brand',
            'OrderDate': 'order_date',
            'OrderID': 'order_id'
        }
        # Only rename if the column name doesn't already exist in the preferred format
        rename_cols = {old: new for old, new in mapping.items() if old in df.columns and new not in df.columns}
        if rename_cols:
            df = df.rename(columns=rename_cols)
            
        # 1. Handle Duplicates (Only for product-based datasets, if not sales-based)
        if 'product_id' in df.columns and 'order_id' not in df.columns:
            df = df.drop_duplicates(subset=['product_id'], keep='first')
        
        # 2. Convert Price Columns
        if 'actual_price' in df.columns:
            df['actual_price'] = df['actual_price'].apply(self._convert_price)
        if 'discounted_price' in df.columns:
            df['discounted_price'] = df['discounted_price'].apply(self._convert_price)
            
        # 3. Convert Discount Percentage
        if 'discount_percentage' in df.columns:
            df['discount_percentage'] = df['discount_percentage'].apply(self._convert_percentage)
            
        # 4. Handle Rating
        if 'rating' in df.columns:
            # handle occurrences where rating is like '4.1 out of 5 stars', just keep the number if it's there
            df['rating'] = df['rating'].astype(str).str.extract(r'([0-9.]+)')[0]
            df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
        
        if 'rating_count' in df.columns:
            # Again, use regex to keep only numbers
            df['rating_count'] = df['rating_count'].astype(str).str.replace(r'[^\d]', '', regex=True)
            df['rating_count'] = pd.to_numeric(df['rating_count'], errors='coerce')

        # To keep as much data as possible, we don't drop everything.
        # We only drop if BOTH rating and price are missing, or if mandatory id/name is missing.
        critical_cols = ['rating', 'actual_price', 'product_id']
        existing_critical = [col for col in critical_cols if col in df.columns]
        # We can still drop rows that have NO usable information at all
        df = df.dropna(subset=['product_id'] if 'product_id' in df.columns else [], how='all').copy()
        
        # Fill critical NaNs with medians/sensible defaults so models don't crash but we keep the rows
        for col in ['rating', 'rating_count', 'actual_price', 'discounted_price', 'discount_percentage']:
            if col in df.columns:
                median_val = df[col].median()
                if pd.isna(median_val): median_val = 0
                df[col] = df[col].fillna(median_val)
        
        # 5. Derived Features
        if 'actual_price' in df.columns and 'discounted_price' in df.columns:
            df['price_difference'] = df['actual_price'] - df['discounted_price']
        
        if 'discount_percentage' in df.columns and 'rating' in df.columns:
            # Discount effectiveness score: higher discount % and higher rating means it's effective
            df['discount_effectiveness_score'] = df['discount_percentage'] * (df['rating'] / 5.0)
        
        if 'rating' in df.columns and 'rating_count' in df.columns:
            # Popularity score = rating * log(rating_count)
            df['popularity_score'] = df['rating'] * np.log1p(df['rating_count'])
        
        # Review length
        if 'review_content' in df.columns:
            df['review_length'] = df['review_content'].astype(str).apply(len)
        else:
            df['review_length'] = 0
            
        # Sentiment score
        df = self.sentiment_analyzer.compute_sentiment_scores(df, text_column='review_content')

        # Fill remaining NaNs for numerics with median
        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())

        return df
