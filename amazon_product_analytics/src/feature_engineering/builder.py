import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

class FeatureBuilder:
    def __init__(self):
        self.label_encoders = {}
        self.tfidf = TfidfVectorizer(max_features=50, stop_words='english')
        self.price_quantiles = {}
        self.cat_counts = {}
        
    def build_features(self, df, is_training=True):
        """Builds all engineered features"""
        df = df.copy()
        
        # 1. Categorical Encoding
        if 'category' in df.columns:
            # Extract main category (first part before |)
            df['main_category'] = df['category'].astype(str).apply(lambda x: x.split('|')[0] if '|' in x else x)
            
            if is_training:
                le = LabelEncoder()
                df['category_encoded'] = le.fit_transform(df['main_category'])
                self.label_encoders['main_category'] = le
            else:
                le = self.label_encoders.get('main_category')
                if le is not None:
                    known_classes = set(le.classes_)
                    encoded_cats = []
                    for c in df['main_category']:
                        if c in known_classes:
                            encoded_cats.append(le.transform([c])[0])
                        else:
                            encoded_cats.append(-1)
                    df['category_encoded'] = encoded_cats
                else:
                    df['category_encoded'] = -1

        # 2. Text Features (TF-IDF on about_product)
        if 'about_product' in df.columns:
            text_data = df['about_product'].fillna('')
            if is_training:
                tfidf_features = self.tfidf.fit_transform(text_data).toarray()
            else:
                tfidf_features = self.tfidf.transform(text_data).toarray()
                
            for i in range(tfidf_features.shape[1]):
                df[f'tfidf_about_{i}'] = tfidf_features[:, i]
                
        # 3. Derived Features
        # Price Band (Quantile based mostly)
        if 'actual_price' in df.columns:
            if is_training:
                self.price_quantiles = df['actual_price'].quantile([0.33, 0.66]).to_dict()
                
            def get_price_band(price):
                if not hasattr(self, 'price_quantiles'):
                    return 0 # Default low
                if price <= self.price_quantiles.get(0.33, 1):
                    return 0 # Low
                elif price <= self.price_quantiles.get(0.66, 1):
                    return 1 # Medium
                else:
                    return 2 # High
            df['price_band'] = df['actual_price'].apply(get_price_band)
            
            # Rating Density: ratio of ratings to price (popularity per rupee)
            df['rating_density'] = df['rating_count'] / (df['actual_price'] + 1.0)
        else:
            df['price_band'] = 0
            df['rating_density'] = 0
            
        # Discount Impact: interaction between rating and discount
        if 'rating' in df.columns and 'discount_percentage' in df.columns:
            df['discount_impact'] = df['rating'] * df['discount_percentage']
        else:
            df['discount_impact'] = 0
            
        # Category Popularity: How many products are in this category
        if 'main_category' in df.columns:
            if is_training:
                self.cat_counts = df['main_category'].value_counts().to_dict()
            
            df['category_popularity'] = df['main_category'].map(self.cat_counts).fillna(0)
        else:
            df['category_popularity'] = 0
            
        # Engagement Score = popularity_score + sentiment_score + (discount_effectiveness_score * 10)
        req_cols = ['popularity_score', 'sentiment_score', 'discount_effectiveness_score']
        if all(c in df.columns for c in req_cols):
            df['engagement_score'] = df['popularity_score'] + df['sentiment_score'] + (df['discount_effectiveness_score'] * 10)
        else:
            df['engagement_score'] = 0.0
            
        return df
