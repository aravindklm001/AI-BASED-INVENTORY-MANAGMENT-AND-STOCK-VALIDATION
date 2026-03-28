# Amazon Product Analytics & Demand Prediction System

A complete standalone system that analyzes product performance, estimates proxy demand, and predicts the success of new products using Amazon product data.

## System Features
- **Data Preprocessing Pipeline**: Handles messy prices, discount strings, and missing ratings.
- **NLP Sentiment Analysis**: Uses TextBlob to understand review sentiment to factor into demand.
- **Feature Engineering**: Extracts `price_band`, `discount_effectiveness_score`, TF-IDF vectors, and categorical encoding.
- **Product Insights Engine**: Classifies products into Demand tiers (High, Medium, Low, Risky).
- **ML Demand Estimation & Success Prediction**: Uses XGBoost to proxy-estimate demand and predict whether a new product will be successful.
- **Recommendation Engine**: Outlines actionable heuristic rules for pricing.
- **Streamlit Dashboard**: Interactive UI to view analytics, charts, and test new product probabilities.

## Setup Instructions

1. **Install Requirements**:
   Ensure you have Python 3.8+ installed. Run:
   ```bash
   pip install -r requirements.txt
   ```

2. **Supply Data (Optional)**:
   By default, the script generates a robust dummy Amazon dataset if one is not found. If you have the real `amazon.csv` (containing columns like `product_id`, `actual_price`, `discounted_price`, `rating`, `review_content`, etc.), place it in:
   ```
   data/amazon.csv
   ```

3. **Run the Backend ML Pipeline**:
   This orchestrates cleaning, feature engineering, and model training.
   ```bash
   python main.py
   ```
   This will create trained models in the `models/` directory, and the insights table in the `output/` directory.

4. **Launch the Dashboard**:
   Once the pipeline finishes and models are saved, run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Directory Structure
- `src/preprocessing/`: Data cleaners
- `src/nlp/`: TextBlob sentiment parsing
- `src/feature_engineering/`: Building complex model features
- `src/analytics/`: Calculating demand proxies and insights
- `src/models/`: XGBoost Regression & Classification
- `src/recommendation/`: Actionable recommendation heuristics
