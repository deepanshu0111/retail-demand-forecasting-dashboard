"""
==============================================
RETAIL DEMAND FORECASTING - STREAMLIT DASHBOARD
==============================================
Interactive web application for demand forecasting

To run this dashboard:
    streamlit run streamlit_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Retail Demand Forecasting",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    h1 {
        color: #1f77b4;
        padding-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("📊 Retail Demand Forecasting Dashboard")
st.markdown("### Predict future product demand with machine learning")
st.markdown("---")

# Sidebar
st.sidebar.header("⚙️ Configuration")
st.sidebar.markdown("---")

# Initialize or load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('retail_demand_data.csv')
        df['date'] = pd.to_datetime(df['date'])
        return df
    except FileNotFoundError:
        st.error("Dataset not found! Please run the main project first.")
        return None

@st.cache_resource
def train_model(df):
    """Train the forecasting model"""
    # Feature engineering
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek
    df['week_of_year'] = df['date'].dt.isocalendar().week
    df['quarter'] = df['date'].dt.quarter
    
    # Encode season
    le_season = LabelEncoder()
    df['season_encoded'] = le_season.fit_transform(df['season'])
    
    # Features
    feature_columns = ['store_id', 'item_id', 'price', 'promotion', 'holiday',
                      'year', 'month', 'day', 'day_of_week', 'week_of_year',
                      'quarter', 'season_encoded']
    
    X = df[feature_columns]
    y = df['sales']
    
    # Train model
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)
    
    return model, le_season, feature_columns

# Load data
df = load_data()

if df is not None:
    # Train model
    with st.spinner("Training model... This may take a moment."):
        model, le_season, feature_columns = train_model(df)
    
    st.sidebar.success("✅ Model trained successfully!")
    
    # Sidebar inputs
    st.sidebar.subheader("📍 Forecast Parameters")
    
    # Store selection
    available_stores = sorted(df['store_id'].unique())
    selected_store = st.sidebar.selectbox(
        "Select Store ID",
        options=available_stores,
        index=0
    )
    
    # Item selection
    available_items = sorted(df['item_id'].unique())
    selected_item = st.sidebar.selectbox(
        "Select Item ID",
        options=available_items,
        index=0
    )
    
    # Date range
    st.sidebar.subheader("📅 Forecast Period")
    forecast_days = st.sidebar.slider(
        "Number of days to forecast",
        min_value=7,
        max_value=90,
        value=30,
        step=7
    )
    
    # Additional parameters
    st.sidebar.subheader("🎯 Scenario Parameters")
    price_input = st.sidebar.number_input(
        "Average Price ($)",
        min_value=10.0,
        max_value=200.0,
        value=float(df['price'].mean()),
        step=5.0
    )
    
    promotion_input = st.sidebar.checkbox("Apply Promotion", value=False)
    holiday_input = st.sidebar.checkbox("Include Holidays", value=False)
    
    # Main content
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🏪 Store</h3>
            <h2>{}</h2>
        </div>
        """.format(selected_store), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>📦 Item</h3>
            <h2>{}</h2>
        </div>
        """.format(selected_item), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>📅 Days</h3>
            <h2>{}</h2>
        </div>
        """.format(forecast_days), unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3>💰 Price</h3>
            <h2>${:.2f}</h2>
        </div>
        """.format(price_input), unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Generate forecast button
    if st.button("🔮 Generate Forecast", type="primary", use_container_width=True):
        with st.spinner("Generating forecast..."):
            # Create future dates
            last_date = df['date'].max()
            future_dates = pd.date_range(
                start=last_date + timedelta(days=1),
                periods=forecast_days,
                freq='D'
            )
            
            # Create future dataframe
            future_data = []
            for date in future_dates:
                season = ('Winter' if date.month in [12, 1, 2] 
                         else 'Spring' if date.month in [3, 4, 5]
                         else 'Summer' if date.month in [6, 7, 8]
                         else 'Fall')
                
                future_data.append({
                    'date': date,
                    'store_id': selected_store,
                    'item_id': selected_item,
                    'price': price_input,
                    'promotion': int(promotion_input),
                    'holiday': int(holiday_input),
                    'year': date.year,
                    'month': date.month,
                    'day': date.day,
                    'day_of_week': date.dayofweek,
                    'week_of_year': date.isocalendar().week,
                    'quarter': (date.month - 1) // 3 + 1,
                    'season': season,
                    'season_encoded': le_season.transform([season])[0]
                })
            
            future_df = pd.DataFrame(future_data)
            
            # Make predictions
            X_future = future_df[feature_columns]
            predictions = model.predict(X_future)
            future_df['predicted_sales'] = np.maximum(predictions, 0)
            
            # Display results
            st.success("✅ Forecast generated successfully!")
            
            # Metrics
            st.subheader("📈 Forecast Summary")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Total Predicted Sales",
                    f"{future_df['predicted_sales'].sum():.0f} units"
                )
            
            with col2:
                st.metric(
                    "Average Daily Sales",
                    f"{future_df['predicted_sales'].mean():.1f} units"
                )
            
            with col3:
                st.metric(
                    "Peak Day Sales",
                    f"{future_df['predicted_sales'].max():.0f} units"
                )
            
            with col4:
                st.metric(
                    "Minimum Day Sales",
                    f"{future_df['predicted_sales'].min():.0f} units"
                )
            
            st.markdown("---")
            
            # Visualization tabs
            tab1, tab2, tab3 = st.tabs(["📊 Forecast Chart", "📅 Daily Breakdown", "📉 Statistics"])
            
            with tab1:
                st.subheader("Demand Forecast Over Time")
                
                fig, ax = plt.subplots(figsize=(14, 6))
                ax.plot(future_df['date'], future_df['predicted_sales'],
                       marker='o', linewidth=2, markersize=6,
                       color='#1f77b4', label='Predicted Sales')
                
                # Add trend line
                z = np.polyfit(range(len(future_df)), future_df['predicted_sales'], 1)
                p = np.poly1d(z)
                ax.plot(future_df['date'], p(range(len(future_df))),
                       linestyle='--', color='red', linewidth=2,
                       alpha=0.7, label='Trend')
                
                # Confidence band
                ax.fill_between(future_df['date'],
                               future_df['predicted_sales'] * 0.9,
                               future_df['predicted_sales'] * 1.1,
                               alpha=0.2, color='#1f77b4',
                               label='90% Confidence Band')
                
                ax.set_xlabel('Date', fontsize=12, fontweight='bold')
                ax.set_ylabel('Predicted Sales (units)', fontsize=12, fontweight='bold')
                ax.set_title(f'Demand Forecast - Store {selected_store}, Item {selected_item}',
                           fontsize=14, fontweight='bold')
                ax.legend(loc='best')
                ax.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
                
                # Additional insights
                st.info(f"""
                **Forecast Insights:**
                - Trend: {'📈 Increasing' if z[0] > 0 else '📉 Decreasing' if z[0] < 0 else '➡️ Stable'}
                - Volatility: {(future_df['predicted_sales'].std() / future_df['predicted_sales'].mean() * 100):.1f}% (Coefficient of Variation)
                - Best day: {future_df.loc[future_df['predicted_sales'].idxmax(), 'date'].strftime('%Y-%m-%d')} ({future_df['predicted_sales'].max():.0f} units)
                - Worst day: {future_df.loc[future_df['predicted_sales'].idxmin(), 'date'].strftime('%Y-%m-%d')} ({future_df['predicted_sales'].min():.0f} units)
                """)
            
            with tab2:
                st.subheader("Daily Sales Predictions")
                
                # Format dataframe for display
                display_df = future_df[['date', 'predicted_sales']].copy()
                display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
                display_df['day_of_week'] = future_df['date'].dt.day_name()
                display_df['predicted_sales'] = display_df['predicted_sales'].round(1)
                display_df.columns = ['Date', 'Predicted Sales (units)', 'Day of Week']
                
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    height=400
                )
                
                # Download button
                csv = display_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Forecast as CSV",
                    data=csv,
                    file_name=f'forecast_store{selected_store}_item{selected_item}.csv',
                    mime='text/csv'
                )
            
            with tab3:
                st.subheader("Statistical Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Distribution plot
                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.hist(future_df['predicted_sales'], bins=20,
                           color='skyblue', edgecolor='black', alpha=0.7)
                    ax.axvline(future_df['predicted_sales'].mean(),
                              color='red', linestyle='--', linewidth=2,
                              label=f'Mean: {future_df["predicted_sales"].mean():.1f}')
                    ax.axvline(future_df['predicted_sales'].median(),
                              color='green', linestyle='--', linewidth=2,
                              label=f'Median: {future_df["predicted_sales"].median():.1f}')
                    ax.set_xlabel('Predicted Sales', fontweight='bold')
                    ax.set_ylabel('Frequency', fontweight='bold')
                    ax.set_title('Sales Distribution', fontweight='bold')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                
                with col2:
                    # Day of week analysis
                    dow_avg = future_df.groupby('day_of_week')['predicted_sales'].mean()
                    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
                    
                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.bar(range(7), [dow_avg.get(i, 0) for i in range(7)],
                          color='coral', edgecolor='black', alpha=0.7)
                    ax.set_xticks(range(7))
                    ax.set_xticklabels(days)
                    ax.set_xlabel('Day of Week', fontweight='bold')
                    ax.set_ylabel('Average Predicted Sales', fontweight='bold')
                    ax.set_title('Average Sales by Day of Week', fontweight='bold')
                    ax.grid(True, alpha=0.3, axis='y')
                    st.pyplot(fig)
                
                # Summary statistics
                st.markdown("### 📊 Summary Statistics")
                stats_df = pd.DataFrame({
                    'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Range', 'Q1', 'Q3', 'IQR'],
                    'Value': [
                        f"{future_df['predicted_sales'].mean():.2f}",
                        f"{future_df['predicted_sales'].median():.2f}",
                        f"{future_df['predicted_sales'].std():.2f}",
                        f"{future_df['predicted_sales'].min():.2f}",
                        f"{future_df['predicted_sales'].max():.2f}",
                        f"{future_df['predicted_sales'].max() - future_df['predicted_sales'].min():.2f}",
                        f"{future_df['predicted_sales'].quantile(0.25):.2f}",
                        f"{future_df['predicted_sales'].quantile(0.75):.2f}",
                        f"{future_df['predicted_sales'].quantile(0.75) - future_df['predicted_sales'].quantile(0.25):.2f}"
                    ]
                })
                st.dataframe(stats_df, use_container_width=True, hide_index=True)
    
    # Historical data section
    st.markdown("---")
    st.subheader("📚 Historical Data Analysis")
    
    with st.expander("View Historical Sales Data"):
        # Filter historical data
        hist_df = df[
            (df['store_id'] == selected_store) &
            (df['item_id'] == selected_item)
        ].sort_values('date', ascending=False)
        
        if len(hist_df) > 0:
            st.write(f"Showing {len(hist_df)} historical records")
            
            # Plot historical trend
            fig, ax = plt.subplots(figsize=(14, 5))
            ax.plot(hist_df['date'], hist_df['sales'],
                   marker='o', linewidth=1.5, markersize=4, alpha=0.7)
            ax.set_xlabel('Date', fontweight='bold')
            ax.set_ylabel('Sales', fontweight='bold')
            ax.set_title(f'Historical Sales - Store {selected_store}, Item {selected_item}',
                       fontweight='bold')
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            
            # Display data table
            display_hist = hist_df[['date', 'sales', 'price', 'promotion', 'holiday']].head(50)
            st.dataframe(display_hist, use_container_width=True)
        else:
            st.warning("No historical data available for this store-item combination.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>📊 Retail Demand Forecasting Dashboard | Built with Streamlit & Scikit-learn</p>
        <p>Model: Random Forest Regressor | Last Updated: {}</p>
    </div>
    """.format(datetime.now().strftime('%Y-%m-%d %H:%M')), unsafe_allow_html=True)

else:
    st.error("❌ Unable to load data. Please ensure 'retail_demand_data.csv' exists.")
    st.info("💡 Run the main project script first to generate the dataset.")