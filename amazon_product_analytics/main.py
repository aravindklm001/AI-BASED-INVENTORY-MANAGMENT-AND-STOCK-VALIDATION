import pandas as pd
import numpy as np
import os
import joblib

from src.preprocessing.cleaner import DataCleaner
from src.feature_engineering.builder import FeatureBuilder
from src.analytics.insights import ProductInsights
from src.recommendation.engine import RecommendationEngine
from src.models.train import ModelTrainer

def generate_dummy_data(output_path='data/amazon.csv'):
    """Generates dummy amazon data if actual dataset is missing."""
    print("Generating dummy dataset since 'data/amazon.csv' was not found.")
    os.makedirs('data', exist_ok=True)
    
    np.random.seed(42)
    n = 1000
    
    data = {
        'product_id': [f'B0{str(i).zfill(8)}' for i in range(n)],
        'product_name': [f'Sample Product {i}' for i in range(n)],
        'category': np.random.choice(['Electronics|TV|Smart', 'Computers|Laptops|Gaming', 'Home|Kitchen|Appliances', 'Clothing|Men|Shirts'], n),
        'discounted_price': [f'₹{np.random.randint(100, 5000)}' for _ in range(n)],
        'actual_price': [f'₹{np.random.randint(500, 10000)}' for _ in range(n)],
        'discount_percentage': [f'{np.random.randint(5, 80)}%' for _ in range(n)],
        'rating': [str(round(np.random.uniform(2.0, 5.0), 1)) for _ in range(n)],
        'rating_count': [f'{np.random.randint(1, 50000):,}' for _ in range(n)],
        'about_product': ['Great quality product with nice features.'] * n,
        'review_content': np.random.choice([
            "Amazing product, totally worth it!",
            "Terrible, broke after one day.",
            "It is okay, nothing special.",
            "Really good for the price. Fast shipping.",
            "Do not buy this waste of money."
        ], n)
    }
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    return output_path

def main():
    print("=== Amazon Product Analytics & Demand Prediction System ===")
    
    data_path = 'data/amazon.csv'
    if not os.path.exists(data_path):
        data_path = generate_dummy_data()
        
    print("\n[1] Loading dataset...")
    df_raw = pd.read_csv(data_path)
    
    print("\n[2] Cleaning & Preprocessing (This includes NLP sentiment analysis)...")
    cleaner = DataCleaner()
    df_cleaned = cleaner.clean_data(df_raw)
    
    print("\n[3] Engineering Features...")
    builder = FeatureBuilder()
    df_features = builder.build_features(df_cleaned, is_training=True)
    
    # Save the feature builder so we can use its encoders/tfidf for inference
    os.makedirs('models', exist_ok=True)
    joblib.dump(builder, 'models/feature_builder.pkl')
    
    print("\n[4] Generating Product Insights...")
    insights = ProductInsights()
    df_insights = insights.generate_insights(df_features)
    
    print("\n[5] Training ML Models...")
    trainer = ModelTrainer()
    metrics = trainer.train_and_evaluate(df_insights)
    print("Model Evaluation Metrics:")
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            print(f"  {k}: {round(v, 4)}")
        else:
            print(f"  {k}: {v}")
    trainer.save_models()
    
    print("\n[6] Generating Recommendations...")
    engine = RecommendationEngine()
    recs = engine.get_recommendations(df_insights)
    print(f"Products to Promote: {len(recs['promote'])}")
    print(f"Products to Discount: {len(recs['discount'])}")
    print(f"Products to Avoid: {len(recs['avoid'])}")
    
    # Save the final insights table
    os.makedirs('output', exist_ok=True)
    final_output_path = 'output/product_insights.csv'
    df_insights.to_csv(final_output_path, index=False)
    print(f"\nPipeline complete! Output saved to {final_output_path}")

if __name__ == '__main__':
    main()
