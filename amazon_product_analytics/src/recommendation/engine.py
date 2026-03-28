import pandas as pd

class RecommendationEngine:
    def __init__(self):
        pass
        
    def get_recommendations(self, df):
        """Generates actionable recommendations for existing products."""
        recommendations = {
            'promote': [],
            'discount': [],
            'avoid': []
        }
        
        # Products to promote: High rating (> 4.3), positive sentiment (> 0.2), low visibility (low rating_count)
        promote_mask = (df['rating'] > 4.3) & (df['sentiment_score'] > 0.2) & (df['product_classification'].isin(['Low Demand', 'Medium Demand']))
        
        # Products to discount: Low demand + high price
        # High price means price_band == 2 (High)
        discount_mask = (df['product_classification'] == 'Low Demand') & (df['price_band'] == 2)
        
        # Products to avoid: Low rating (< 3.5) + negative sentiment (< 0)
        avoid_mask = (df['rating'] < 3.5) & (df['sentiment_score'] < 0)
        
        # We output product_id or names for recommendations
        if 'product_name' in df.columns:
            recommendations['promote'] = df[promote_mask][['product_name', 'rating', 'actual_price']].head(10).to_dict('records')
            recommendations['discount'] = df[discount_mask][['product_name', 'actual_price']].head(10).to_dict('records')
            recommendations['avoid'] = df[avoid_mask][['product_name', 'rating', 'sentiment_score']].head(10).to_dict('records')
            
        return recommendations

    def get_pricing_suggestions(self, df):
        """Provides heuristic pricing suggestions"""
        pricing_sugg = []
        for _, row in df.iterrows():
            if row.get('product_classification') == 'High Demand' and row.get('discount_percentage', 0) > 0.3:
                pricing_sugg.append({'product_name': row.get('product_name'), 'suggestion': 'Reduce discount to maximize margin'})
            elif row.get('product_classification') == 'Low Demand' and row.get('discount_percentage', 0) < 0.1:
                pricing_sugg.append({'product_name': row.get('product_name'), 'suggestion': 'Increase discount to stimulate demand'})
        
        return pricing_sugg[:10]
