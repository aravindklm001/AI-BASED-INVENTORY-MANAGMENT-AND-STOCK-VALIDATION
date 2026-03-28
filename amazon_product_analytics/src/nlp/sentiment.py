from textblob import TextBlob
import pandas as pd

class SentimentAnalyzer:
    """Analyze sentiment of reviews and product descriptions."""
    
    def __init__(self):
        pass
        
    def analyze(self, text):
        """Returns polarity score from -1.0 to 1.0"""
        if pd.isna(text) or not isinstance(text, str):
            return 0.0
        return TextBlob(text).sentiment.polarity

    def compute_sentiment_scores(self, df, text_column='review_content'):
        """Computes sentiment score for the entire column."""
        if text_column not in df.columns:
            df['sentiment_score'] = 0.0
            return df
        # Apply sentiment
        df['sentiment_score'] = df[text_column].apply(self.analyze)
        return df
