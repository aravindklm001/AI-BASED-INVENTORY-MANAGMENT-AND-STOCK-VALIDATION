import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

from src.models.predictor import NewProductPredictor
from src.preprocessing.cleaner import DataCleaner
from src.analytics.insights import ProductInsights
from src.feature_engineering.builder import FeatureBuilder

def load_data():
    if os.path.exists('output/product_insights.csv'):
        return pd.read_csv('output/product_insights.csv')
    return None

st.set_page_config(page_title="Amazon Product Analytics", layout="wide")

st.title("Amazon Product Analytics & Demand Prediction")

st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Attach a CSV file to analyze", type=['csv'])

df = None
if uploaded_file is not None:
    try:
        raw_user_df = pd.read_csv(uploaded_file)
        st.sidebar.success("File uploaded!")
        
        st.header("Uploaded File Details")
        colA, colB = st.columns(2)
        colA.metric("Total Rows", raw_user_df.shape[0])
        colB.metric("Total Columns", raw_user_df.shape[1])
        with st.expander("Preview Raw Data Attached"):
            st.dataframe(raw_user_df.head(25))
            
        # 0. Detect and Aggregate if it's an Order List
        if 'OrderID' in raw_user_df.columns or 'order_id' in raw_user_df.columns:
            st.info("Detected Sales/Order format. Aggregating data by Product...")
            
            # Map columns for aggregation if not already mapped
            agg_map = {
                'ProductID': 'product_id', 'ProductName': 'product_name', 
                'Category': 'category', 'UnitPrice': 'actual_price', 
                'Quantity': 'quantity', 'Brand': 'brand'
            }
            work_df = raw_user_df.rename(columns={k: v for k, v in agg_map.items() if k in raw_user_df.columns})
            
            # Clean numeric columns before aggregation
            cleaner = DataCleaner()
            if 'actual_price' in work_df.columns:
                work_df['actual_price'] = work_df['actual_price'].apply(cleaner._convert_price)
            if 'quantity' in work_df.columns:
                work_df['quantity'] = pd.to_numeric(work_df['quantity'], errors='coerce').fillna(0)
            
            # Aggregate: One row per product
            agg_logic = {
                'product_name': 'first',
                'category': 'first',
                'actual_price': 'mean',
                'quantity': 'sum'
            }
            if 'brand' in work_df.columns: agg_logic['brand'] = 'first'
            
            raw_user_df = work_df.groupby('product_id').agg(agg_logic).reset_index()
            # Map back to 'rating_count' for pipeline compatibility if needed, 
            # but cleaner will handle it if we just pass quantity
            raw_user_df['rating_count'] = raw_user_df['quantity']
            raw_user_df['rating'] = 4.0 # Default rating for sales data
            
        required_cols = ['actual_price', 'product_id']
        
        if all(c in raw_user_df.columns for c in required_cols):
            with st.spinner("Running full ML Analytics Pipeline on uploaded data..."):
                cleaner = DataCleaner()
                df_clean = cleaner.clean_data(raw_user_df)
                
                try:
                    builder = joblib.load('models/feature_builder.pkl')
                    df_features = builder.build_features(df_clean, is_training=False)
                    
                    insights = ProductInsights()
                    df = insights.generate_insights(df_features)
                    st.success("Successfully analyzed your uploaded dataset! All KPIs and Visualizations below have been updated.")
                except Exception as e:
                    st.error(f"Analyzed base data, but failed to load ML models: {str(e)}")
                    df = df_clean # Fallback
        else:
            st.warning("Your CSV does not match the expected Amazon product schema (missing `actual_price` or `rating`). Showing generic file details instead of Product Insights.")
    except Exception as e:
        st.sidebar.error(f"Error reading CSV: {str(e)}")
else:
    df = load_data()

if df is not None:
    st.header("Product Insights Engine")
    
    # Quick KPIs
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Total Products", len(df))
    
    if 'product_classification' in df.columns:
        kpi2.metric("High Demand Products", len(df[df['product_classification'] == 'High Demand']))
        kpi3.metric("Risky Products", len(df[df['product_classification'] == 'Risky Product']))
    else:
        kpi2.metric("High Demand Products", "N/A")
        kpi3.metric("Risky Products", "N/A")
        
    avg_sent = round(df['sentiment_score'].mean(), 2) if 'sentiment_score' in df.columns else 0.0
    kpi4.metric("Avg Sentiment Score", avg_sent)
    
    st.subheader("Detailed Data Catalog (Full Dataset)")
    display_cols = ['product_name', 'category', 'actual_price', 'discount_percentage', 'rating', 'demand_score', 'product_classification']
    if 'brand' in df.columns: display_cols.insert(2, 'brand')
    if 'quantity' in df.columns: display_cols.insert(3, 'quantity')
    
    st.dataframe(df[display_cols])
    
    st.write("---")
    st.header("Category-wise Summary Report")
    
    # Calculate category summary
    agg_dict = {
        'product_name': 'count',
        'actual_price': 'mean',
        'rating': 'mean'
    }
    if 'demand_score' in df.columns: agg_dict['demand_score'] = 'mean'
    if 'sentiment_score' in df.columns: agg_dict['sentiment_score'] = 'mean'
    
    cat_summary = df.groupby('category').agg(agg_dict).rename(columns={
        'product_name': 'Total Products', 
        'actual_price': 'Avg Price', 
        'rating': 'Avg Rating', 
        'demand_score': 'Avg Demand Score', 
        'sentiment_score': 'Avg Sentiment Score'
    })
    
    st.dataframe(cat_summary.sort_values(by='Total Products', ascending=False))
    
    st.write("---")
    st.header("Visualizations")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Rating vs Demand Score")
        fig, ax = plt.subplots(figsize=(8,6))
        sns.scatterplot(data=df, x='rating', y='demand_score', hue='product_classification', alpha=0.7, ax=ax)
        st.pyplot(fig)
        
        st.subheader("Category Performance")
        fig, ax = plt.subplots(figsize=(8,6))
        # Category selection for performance analysis
        if 'category' in df.columns:
            all_categories = sorted(df['category'].dropna().unique().tolist())
            top_default = df['category'].value_counts().head(5).index.tolist()
            
            selected_cats = st.multiselect("Select Categories to Compare", options=all_categories, default=top_default)
            
            if selected_cats:
                sns.boxplot(data=df[df['category'].isin(selected_cats)], y='category', x='demand_score', ax=ax)
                st.pyplot(fig)
            else:
                st.info("Select one or more categories to visualize performance.")
        
    with col2:
        st.subheader("Sentiment Distribution")
        fig, ax = plt.subplots(figsize=(8,6))
        sns.histplot(df['sentiment_score'], kde=True, bins=30, ax=ax)
        st.pyplot(fig)
        
        st.subheader("Price vs Popularity Score")
        fig, ax = plt.subplots(figsize=(8,6))
        if 'actual_price' in df.columns and 'popularity_score' in df.columns:
            sns.scatterplot(data=df, x='actual_price', y='popularity_score', hue='popularity_score', alpha=0.5, ax=ax)
            ax.set_xscale('log')
        st.pyplot(fig)
else:
    st.warning("Data not found. Please run `python main.py` first to generate the analytics output and models.")
    
st.write("---")
st.header("New Product Success Prediction")

with st.form("new_product_form"):
    col_a, col_b = st.columns(2)
    with col_a:
        product_name = st.text_input("Product Name", value="New Smart TV")
        
        # Populate dropdown from active analyzed dataset categories
        if df is not None:
            # Try finding the best category column (full string or main category)
            cat_col = 'category' if 'category' in df.columns else 'main_category' if 'main_category' in df.columns else None
            
            if cat_col:
                available_categories = sorted([str(cat) for cat in df[cat_col].dropna().unique().tolist() if str(cat).strip()])
            else:
                available_categories = ["Electronics|TV|Smart", "Computers|Laptops|Gaming"]
        else:
            available_categories = ["Electronics|TV|Smart", "Computers|Laptops|Gaming"]
        
        if not available_categories:
            available_categories = ["Electronics|TV|Smart", "Computers|Laptops|Gaming"]
        
        category = st.selectbox("Category", options=available_categories)
        
        actual_price = st.number_input("Actual Price (₹)", min_value=1.0, value=25000.0)
    with col_b:
        expected_rating = st.slider("Expected Target Rating", 1.0, 5.0, 4.2)
        discount_percentage = st.slider("Expected Discount %", 0, 100, 15)
        about_product = st.text_area("Product Description", value="High quality smart TV with 4k display.")
    
    submit = st.form_submit_button("Predict Success")

if submit:
    if not os.path.exists('models/feature_builder.pkl'):
        st.error("Models not found. You need to run `python main.py` first to train models.")
    else:
        predictor = NewProductPredictor(models_dir='models')
        
        discounted_price = actual_price * (1 - (discount_percentage / 100.0))
        
        raw_data = {
            'product_id': 'NEW_001',
            'product_name': product_name,
            'category': category,
            'actual_price': f'₹{actual_price}',
            'discounted_price': f'₹{discounted_price}',
            'discount_percentage': f'{discount_percentage}%',
            'about_product': about_product,
            'rating': str(expected_rating),
            'rating_count': '100', # Nominal baseline for new product
            'review_content': '' 
        }
        
        df_new = pd.DataFrame([raw_data])
        
        cleaner = DataCleaner()
        df_clean = cleaner.clean_data(df_new)
        
        try:
            builder = joblib.load('models/feature_builder.pkl')
            df_features = builder.build_features(df_clean, is_training=False)
            
            result = predictor.predict_success(df_features)
            
            if "error" in result:
                st.error(result["error"])
            else:
                st.success("Prediction Generated!")
                res_col1, res_col2, res_col3 = st.columns(3)
                res_col1.metric("Expected Demand Score", result['expected_demand_score'])
                res_col2.metric("Success Probability", f"{result['success_probability']}%")
                
                risk_color = "green" if result['risk_level'] == "Low Risk" else "orange" if result['risk_level'] == "Medium Risk" else "red"
                res_col3.markdown(f"### Risk Level:<br><span style='color:{risk_color}'>{result['risk_level']}</span>", unsafe_allow_html=True)
                
                # --- NEW Recommendations Section ---
                st.write("---")
                st.subheader("Recommended Minimum Targets to Lower Risk")
                
                # Get recommendations
                recs = predictor.get_recommendations(df_new, cleaner, builder)
                
                if not recs:
                    st.info("No substantial improvements found within reasonable ranges.")
                elif recs.get("status") == "already_optimal":
                    st.success(recs["message"])
                else:
                    rec_col1, rec_col2 = st.columns(2)
                    if 'target_rating' in recs:
                        rec_col1.info(f"📈 **Target Rating**: {recs['target_rating']}+")
                        rec_col1.write("An increase in quality/rating significantly boosts success probability.")
                    
                    if 'target_price' in recs:
                        rec_col2.info(f"💰 **Optimal Price**: ₹{recs['target_price']}")
                        rec_col2.write("Reducing the price to this level makes the product more competitive.")
                        
                    if len(about_product.split()) < 10:
                        st.warning("📝 **Tip**: Broaden your Product Description with more features to improve NLP-based demand score.")
                
        except Exception as e:
            st.error(f"Failed to run predictions: {str(e)}")
