import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
import os
warnings.filterwarnings('ignore')

from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import io

# Page configuration
st.set_page_config(
    page_title="Qatar Museums Forecasting",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        text-align: center;
        transition: transform 0.3s ease;
        height: 100%;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.2);
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1e3c72;
        margin: 0.5rem 0;
    }
    .metric-label {
        font-size: 1rem;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }
    .insight-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .success-box {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .warning-box {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: #333;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        font-weight: 600;
    }
    .section-title {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        margin: 2rem 0 1rem 0;
        font-size: 1.5rem;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .recommendation-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    .recommendation-card h4 {
        color: #1e3c72;
        margin-top: 0;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
GDRIVE_FILE_ID = "1xenR2QPW6P6qe3SzUDUlLsfqqBcyiGTT"
DATA_URL = f"https://drive.google.com/uc?export=download&id={GDRIVE_FILE_ID}"

@st.cache_data(ttl=3600)
def load_data():
    """Load and parse the dataset from Google Drive"""
    try:
        df = pd.read_excel(DATA_URL)

        # Convert date column - try multiple formats
        try:
            df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
        except Exception:
            try:
                df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
            except Exception:
                df['date'] = pd.to_datetime(df['date'])

        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def engineer_features(df):
    """Add time-based features"""
    df = df.copy()
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['week'] = df['date'].dt.isocalendar().week
    df['quarter'] = df['date'].dt.quarter
    df['day_of_year'] = df['date'].dt.dayofyear
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    le_museum = LabelEncoder()
    le_dow = LabelEncoder()
    df['museum_encoded'] = le_museum.fit_transform(df['museum'])
    df['dow_encoded'] = le_dow.fit_transform(df['day_of_week'])
    
    return df, le_museum, le_dow

@st.cache_data
def create_forecasts(df, forecast_days):
    """Create forecasts using Prophet and ML"""
    df_featured, le_museum, le_dow = engineer_features(df)
    
    # Prophet forecasts
    prophet_results = {}
    for museum in df['museum'].unique():
        museum_df = df[df['museum'] == museum].copy()
        prophet_df = pd.DataFrame({
            'ds': museum_df['date'],
            'y': museum_df['visitor_count']
        })
        
        prophet_df['is_weekend'] = museum_df['is_weekend_qatar'].values
        prophet_df['temp'] = museum_df['avg_temp_c'].values
        
        model = Prophet(
            yearly_seasonality=True, 
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.05
        )
        model.add_regressor('is_weekend')
        model.add_regressor('temp')
        
        with st.spinner(f'Training Prophet model for {museum}...'):
            model.fit(prophet_df)
        
        future = model.make_future_dataframe(periods=forecast_days)
        future['is_weekend'] = future['ds'].dt.dayofweek.isin([4, 5]).astype(int)
        future['temp'] = prophet_df['temp'].mean()
        
        forecast = model.predict(future)
        prophet_results[museum] = forecast
    
    # ML Model
    feature_cols = ['museum_encoded', 'dow_encoded', 'is_weekend_qatar', 'month',
                    'quarter', 'day_of_year', 'month_sin', 'month_cos', 'avg_temp_c']
    
    X = df_featured[feature_cols]
    y = df_featured['visitor_count']
    
    with st.spinner('Training Random Forest model...'):
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf_model.fit(X, y)
    
    # Create future predictions
    last_date = df['date'].max()
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_days)
    
    predictions = []
    for museum in df['museum'].unique():
        prophet_pred = prophet_results[museum]
        prophet_future = prophet_pred[prophet_pred['ds'] > last_date]['yhat'].values[:forecast_days]
        
        for i, date in enumerate(future_dates):
            future_row = {
                'museum_encoded': le_museum.transform([museum])[0],
                'dow_encoded': date.weekday(),
                'is_weekend_qatar': 1 if date.weekday() in [4, 5] else 0,
                'month': date.month,
                'quarter': (date.month - 1) // 3 + 1,
                'day_of_year': date.dayofyear,
                'month_sin': np.sin(2 * np.pi * date.month / 12),
                'month_cos': np.cos(2 * np.pi * date.month / 12),
                'avg_temp_c': df['avg_temp_c'].mean()
            }
            
            ml_pred = rf_model.predict(pd.DataFrame([future_row]))[0]
            ensemble_pred = 0.5 * prophet_future[i] + 0.5 * ml_pred
            
            predictions.append({
                'date': date,
                'museum': museum,
                'predicted_visitors': max(0, ensemble_pred),
                'prophet_prediction': prophet_future[i],
                'ml_prediction': ml_pred,
                'day_of_week': date.strftime('%A'),
                'is_weekend': 1 if date.weekday() in [4, 5] else 0
            })
    
    return pd.DataFrame(predictions), rf_model

def calculate_sales(predictions_df, historical_df):
    """Calculate predicted sales based on visitor predictions"""
    historical_df['sales_per_visitor'] = historical_df['total_sales_qar'] / historical_df['visitor_count']
    historical_df['ticket_per_visitor'] = historical_df['ticket_sales_qar'] / historical_df['visitor_count']
    historical_df['giftshop_per_visitor'] = historical_df['giftshop_sales_qar'] / historical_df['visitor_count']
    historical_df['cafe_per_visitor'] = historical_df['cafe_sales_qar'] / historical_df['visitor_count']
    
    ratios = historical_df.groupby('museum').agg({
        'sales_per_visitor': 'mean',
        'ticket_per_visitor': 'mean',
        'giftshop_per_visitor': 'mean',
        'cafe_per_visitor': 'mean'
    }).reset_index()
    
    predictions_df = predictions_df.merge(ratios, on='museum', how='left')
    predictions_df['predicted_total_sales'] = predictions_df['predicted_visitors'] * predictions_df['sales_per_visitor']
    predictions_df['predicted_ticket_sales'] = predictions_df['predicted_visitors'] * predictions_df['ticket_per_visitor']
    predictions_df['predicted_giftshop_sales'] = predictions_df['predicted_visitors'] * predictions_df['giftshop_per_visitor']
    predictions_df['predicted_cafe_sales'] = predictions_df['predicted_visitors'] * predictions_df['cafe_per_visitor']
    
    return predictions_df

def create_excel_report(historical_df, predictions_df):
    """Create Excel report"""
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Daily predictions
        daily_export = predictions_df[[
            'date', 'museum', 'day_of_week', 'is_weekend', 'predicted_visitors',
            'predicted_total_sales', 'predicted_ticket_sales',
            'predicted_giftshop_sales', 'predicted_cafe_sales'
        ]].copy()
        daily_export.columns = ['Date', 'Museum', 'Day', 'Weekend', 'Visitors',
                                'Total Sales (QAR)', 'Ticket Sales', 'Gift Shop', 'Cafe']
        daily_export.to_excel(writer, sheet_name='Daily Forecast', index=False)
        
        # Monthly summary
        predictions_df['month'] = predictions_df['date'].dt.strftime('%B %Y')
        monthly = predictions_df.groupby(['month', 'museum']).agg({
            'predicted_visitors': 'sum',
            'predicted_total_sales': 'sum'
        }).reset_index()
        monthly.columns = ['Month', 'Museum', 'Predicted Visitors', 'Predicted Sales (QAR)']
        monthly.to_excel(writer, sheet_name='Monthly Summary', index=False)
        
        # Museum summary
        museum_summary = predictions_df.groupby('museum').agg({
            'predicted_visitors': 'sum',
            'predicted_total_sales': 'sum'
        }).reset_index()
        museum_summary.columns = ['Museum', 'Total Visitors', 'Total Sales (QAR)']
        museum_summary.to_excel(writer, sheet_name='Museum Summary', index=False)
        
        # Summary statistics
        total_visitors = predictions_df['predicted_visitors'].sum()
        total_sales = predictions_df['predicted_total_sales'].sum()
        avg_daily = total_visitors / len(predictions_df['date'].unique())
        
        summary_data = {
            'Metric': [
                'Total Predicted Visitors',
                'Total Predicted Sales (QAR)',
                'Average Daily Visitors',
                'Number of Museums',
                'Forecast Period (days)'
            ],
            'Value': [
                f'{total_visitors:,.0f}',
                f'{total_sales:,.2f}',
                f'{avg_daily:,.0f}',
                predictions_df['museum'].nunique(),
                len(predictions_df['date'].unique())
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
    
    return output.getvalue()

# Main App
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏛️ Qatar Museums Forecasting Dashboard</h1>
        <p style="font-size: 1.2rem; margin-top: 0.5rem;">
            AI-Powered Visitor & Revenue Predictions for Strategic Planning
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📊 Dashboard Controls")
        st.markdown("---")

        forecast_days = st.slider(
            "📅 Forecast Period (Days)",
            min_value=30,
            max_value=180,
            value=90,
            step=30,
            help="How many days to predict"
        )

        st.markdown("---")
        st.markdown("### 🎯 What You'll Get")
        st.info("""
        **This dashboard provides:**
        - 📊 Visitor predictions
        - 💰 Revenue forecasts
        - 📈 Trend analysis
        - 📅 Peak period identification
        - 💡 Actionable recommendations
        """)

        st.markdown("---")
        st.markdown("### 🤖 AI Models Used")
        st.success("""
        - **Prophet**: Time series forecasting
        - **Random Forest**: Machine learning
        - **Ensemble**: Combined predictions
        """)
    
    # Main content
    df = load_data()

    if df is not None:
        st.success(f"✅ Loaded {len(df):,} records from {df['date'].min().strftime('%b %d, %Y')} to {df['date'].max().strftime('%b %d, %Y')}")
        
        # Show basic info
        with st.expander("📊 Data Preview"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Records", f"{len(df):,}")
            with col2:
                st.metric("Museums", df['museum'].nunique())
            with col3:
                st.metric("Date Range", f"{(df['date'].max() - df['date'].min()).days} days")
            
            st.dataframe(df.head(10), use_container_width=True)
        
        # Create tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Overview",
            "🔮 Predictions",
            "💰 Revenue Forecast",
            "📅 Calendar View",
            "📥 Download Report"
        ])
        
        # TAB 1: Overview
        with tab1:
            st.markdown('<div class="section-title">📊 Historical Performance</div>', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_visitors = df['visitor_count'].sum()
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Total Visitors</div>
                    <div class="metric-value">{total_visitors:,.0f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                total_revenue = df['total_sales_qar'].sum()
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Total Revenue</div>
                    <div class="metric-value">{total_revenue/1000000:.1f}M</div>
                    <div style="color: #666; font-size: 0.9rem;">QAR</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                avg_daily = df.groupby('date')['visitor_count'].sum().mean()
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Daily Average</div>
                    <div class="metric-value">{avg_daily:,.0f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                revenue_per_visitor = total_revenue / total_visitors
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Revenue/Visitor</div>
                    <div class="metric-value">{revenue_per_visitor:.0f}</div>
                    <div style="color: #666; font-size: 0.9rem;">QAR</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Trend chart
            st.markdown("#### 📈 Visitor Trends")
            fig = go.Figure()
            
            colors = ['#667eea', '#764ba2', '#f093fb', '#fa709a', '#fee140', '#4facfe']
            for i, museum in enumerate(df['museum'].unique()):
                museum_data = df[df['museum'] == museum]
                fig.add_trace(go.Scatter(
                    x=museum_data['date'],
                    y=museum_data['visitor_count'],
                    name=museum,
                    mode='lines',
                    line=dict(width=2, color=colors[i % len(colors)])
                ))
            
            fig.update_layout(
                height=450,
                hovermode='x unified',
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(size=13),
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Museum comparison
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🏛️ Total Visitors by Museum")
                museum_totals = df.groupby('museum')['visitor_count'].sum().sort_values(ascending=False)
                
                fig_museum = go.Figure(data=[
                    go.Bar(
                        y=museum_totals.index,
                        x=museum_totals.values,
                        orientation='h',
                        marker=dict(
                            color=colors[:len(museum_totals)],
                            line=dict(color='white', width=2)
                        ),
                        text=[f'{v:,.0f}' for v in museum_totals.values],
                        textposition='outside'
                    )
                ])
                
                fig_museum.update_layout(
                    height=350,
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font=dict(size=12),
                    xaxis=dict(title="Visitors", gridcolor='#f0f0f0'),
                    yaxis=dict(title="")
                )
                
                st.plotly_chart(fig_museum, use_container_width=True)
            
            with col2:
                st.markdown("#### 📅 Weekend Performance")
                weekend_avg = df[df['is_weekend_qatar'] == 1]['visitor_count'].mean()
                weekday_avg = df[df['is_weekend_qatar'] == 0]['visitor_count'].mean()
                uplift = ((weekend_avg / weekday_avg) - 1) * 100
                
                fig_weekend = go.Figure(data=[
                    go.Bar(
                        x=['Weekday', 'Weekend'],
                        y=[weekday_avg, weekend_avg],
                        marker=dict(color=['#667eea', '#f093fb']),
                        text=[f'{weekday_avg:,.0f}', f'{weekend_avg:,.0f}'],
                        textposition='outside'
                    )
                ])
                
                fig_weekend.update_layout(
                    height=350,
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font=dict(size=12),
                    yaxis=dict(title="Avg Visitors", gridcolor='#f0f0f0')
                )
                
                st.plotly_chart(fig_weekend, use_container_width=True)
            
            st.markdown(f"""
            <div class="insight-box">
                <h3 style="margin-top: 0;">💡 Key Insight</h3>
                <p style="font-size: 1.2rem; margin-bottom: 0;">
                    Weekend visitors are <strong>{uplift:.1f}% higher</strong> than weekdays.
                    Plan staffing and inventory accordingly!
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # TAB 2: Predictions
        with tab2:
            st.markdown('<div class="section-title">🔮 Future Predictions</div>', unsafe_allow_html=True)
            
            with st.spinner("🤖 AI is analyzing patterns and creating predictions..."):
                predictions_df, rf_model = create_forecasts(df, forecast_days)
                predictions_df = calculate_sales(predictions_df, df)
            
            st.success(f"✅ Generated {forecast_days}-day forecast successfully!")
            
            # Summary metrics
            total_pred_visitors = predictions_df['predicted_visitors'].sum()
            total_pred_sales = predictions_df['predicted_total_sales'].sum()
            avg_daily_pred = total_pred_visitors / forecast_days
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card" style="background: linear-gradient(135deg, #667eea, #764ba2); color: white;">
                    <div class="metric-label" style="color: white;">Predicted Visitors</div>
                    <div class="metric-value" style="color: white;">{total_pred_visitors:,.0f}</div>
                    <div style="opacity: 0.9;">Next {forecast_days} days</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card" style="background: linear-gradient(135deg, #4facfe, #00f2fe); color: white;">
                    <div class="metric-label" style="color: white;">Predicted Revenue</div>
                    <div class="metric-value" style="color: white;">{total_pred_sales/1000000:.2f}M</div>
                    <div style="opacity: 0.9;">QAR</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                growth = ((avg_daily_pred / avg_daily) - 1) * 100
                st.markdown(f"""
                <div class="metric-card" style="background: linear-gradient(135deg, #f093fb, #f5576c); color: white;">
                    <div class="metric-label" style="color: white;">Daily Average</div>
                    <div class="metric-value" style="color: white;">{avg_daily_pred:,.0f}</div>
                    <div style="opacity: 0.9;">{growth:+.1f}% vs historical</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                pred_rpv = total_pred_sales / total_pred_visitors
                st.markdown(f"""
                <div class="metric-card" style="background: linear-gradient(135deg, #fa709a, #fee140); color: white;">
                    <div class="metric-label" style="color: white;">Revenue/Visitor</div>
                    <div class="metric-value" style="color: white;">{pred_rpv:.0f}</div>
                    <div style="opacity: 0.9;">QAR</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Forecast chart
            st.markdown("#### 📊 Historical vs Predicted Visitors")
            
            fig_forecast = go.Figure()
            
            colors = ['#667eea', '#764ba2', '#f093fb', '#fa709a', '#fee140', '#4facfe']
            for i, museum in enumerate(df['museum'].unique()):
                # Historical (last 90 days)
                hist_data = df[df['museum'] == museum].tail(90)
                fig_forecast.add_trace(go.Scatter(
                    x=hist_data['date'],
                    y=hist_data['visitor_count'],
                    name=f'{museum} (Historical)',
                    mode='lines',
                    line=dict(width=2, color=colors[i % len(colors)])
                ))
                
                # Forecast
                pred_data = predictions_df[predictions_df['museum'] == museum]
                fig_forecast.add_trace(go.Scatter(
                    x=pred_data['date'],
                    y=pred_data['predicted_visitors'],
                    name=f'{museum} (Forecast)',
                    mode='lines',
                    line=dict(width=2, color=colors[i % len(colors)], dash='dash')
                ))
            
            fig_forecast.update_layout(
                height=500,
                hovermode='x unified',
                plot_bgcolor='white',
                paper_bgcolor='white',
                legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
            )
            
            st.plotly_chart(fig_forecast, use_container_width=True)
            
            # Monthly breakdown
            st.markdown("#### 📅 Monthly Predictions")
            
            predictions_df['month_name'] = predictions_df['date'].dt.strftime('%B %Y')
            monthly_pred = predictions_df.groupby(['month_name', 'museum'])['predicted_visitors'].sum().reset_index()
            
            fig_monthly = px.bar(
                monthly_pred,
                x='month_name',
                y='predicted_visitors',
                color='museum',
                barmode='group',
                color_discrete_sequence=colors
            )
            
            fig_monthly.update_layout(
                height=400,
                plot_bgcolor='white',
                paper_bgcolor='white',
                xaxis=dict(title="Month"),
                yaxis=dict(title="Predicted Visitors")
            )
            
            st.plotly_chart(fig_monthly, use_container_width=True)
            
            # Museum breakdown
            st.markdown("#### 🏛️ Predictions by Museum")
            
            museums = predictions_df['museum'].unique()
            cols = st.columns(min(len(museums), 3))
            
            for idx, museum in enumerate(museums):
                museum_pred = predictions_df[predictions_df['museum'] == museum]
                total_visitors = museum_pred['predicted_visitors'].sum()
                total_sales = museum_pred['predicted_total_sales'].sum()
                
                with cols[idx % 3]:
                    st.markdown(f"""
                    <div class="recommendation-card">
                        <h4>{museum}</h4>
                        <p style="font-size: 1.5rem; color: #667eea; font-weight: bold; margin: 0.5rem 0;">
                            {total_visitors:,.0f} visitors
                        </p>
                        <p style="font-size: 1.2rem; color: #764ba2; margin: 0;">
                            {total_sales/1000:.0f}K QAR revenue
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
        
        # TAB 3: Revenue Forecast
        with tab3:
            st.markdown('<div class="section-title">💰 Revenue Forecasting</div>', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            
            total_ticket = predictions_df['predicted_ticket_sales'].sum()
            total_giftshop = predictions_df['predicted_giftshop_sales'].sum()
            total_cafe = predictions_df['predicted_cafe_sales'].sum()
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Total Revenue</div>
                    <div class="metric-value">{total_pred_sales/1000:.0f}K</div>
                    <div style="color: #666;">QAR</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Ticket Sales</div>
                    <div class="metric-value">{total_ticket/1000:.0f}K</div>
                    <div style="color: #666;">QAR ({total_ticket/total_pred_sales*100:.1f}%)</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Gift Shop</div>
                    <div class="metric-value">{total_giftshop/1000:.0f}K</div>
                    <div style="color: #666;">QAR ({total_giftshop/total_pred_sales*100:.1f}%)</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Cafe Sales</div>
                    <div class="metric-value">{total_cafe/1000:.0f}K</div>
                    <div style="color: #666;">QAR ({total_cafe/total_pred_sales*100:.1f}%)</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Revenue pie chart
            st.markdown("#### 🥧 Revenue Sources")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                revenue_breakdown = pd.DataFrame({
                    'Category': ['Tickets', 'Gift Shop', 'Cafe'],
                    'Amount': [total_ticket, total_giftshop, total_cafe]
                })
                
                fig_pie = go.Figure(data=[go.Pie(
                    labels=revenue_breakdown['Category'],
                    values=revenue_breakdown['Amount'],
                    hole=0.4,
                    marker=dict(colors=['#667eea', '#764ba2', '#f093fb']),
                    textinfo='label+percent',
                    textfont=dict(size=14)
                )])
                
                fig_pie.update_layout(
                    height=400,
                    paper_bgcolor='white',
                    showlegend=True
                )
                
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                st.markdown("#### 💡 Revenue Insights")
                st.markdown(f"""
                <div class="success-box">
                    <h4 style="margin-top: 0;">Top Revenue Generator</h4>
                    <p style="font-size: 1.1rem;">
                        <strong>Tickets</strong> account for the largest revenue share at
                        <strong>{total_ticket/total_pred_sales*100:.1f}%</strong>
                    </p>
                </div>
                
                <div class="warning-box">
                    <h4 style="margin-top: 0;">Growth Opportunity</h4>
                    <p style="font-size: 1rem;">
                        Increase gift shop marketing during peak visiting hours to boost
                        secondary revenue streams.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            # Daily revenue trend
            st.markdown("#### 📈 Daily Revenue Forecast")
            
            daily_revenue = predictions_df.groupby('date')['predicted_total_sales'].sum().reset_index()
            
            fig_revenue = go.Figure()
            
            fig_revenue.add_trace(go.Scatter(
                x=daily_revenue['date'],
                y=daily_revenue['predicted_total_sales'],
                name='Total Revenue',
                mode='lines',
                line=dict(width=3, color='#667eea'),
                fill='tozeroy',
                fillcolor='rgba(102, 126, 234, 0.1)'
            ))
            
            fig_revenue.update_layout(
                height=400,
                plot_bgcolor='white',
                paper_bgcolor='white',
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_revenue, use_container_width=True)
        
        # TAB 4: Calendar View
        with tab4:
            st.markdown('<div class="section-title">📅 Calendar View & Peak Periods</div>', unsafe_allow_html=True)
            
            # Peak days
            st.markdown("#### 🔥 Peak Visitor Days")
            
            top_days = predictions_df.groupby('date').agg({
                'predicted_visitors': 'sum',
                'predicted_total_sales': 'sum'
            }).reset_index().nlargest(10, 'predicted_visitors')
            
            top_days['day_name'] = top_days['date'].dt.strftime('%A, %B %d, %Y')
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig_top = go.Figure(data=[
                    go.Bar(
                        x=top_days['predicted_visitors'],
                        y=top_days['day_name'],
                        orientation='h',
                        marker=dict(
                            color=top_days['predicted_visitors'],
                            colorscale='Reds',
                            showscale=False
                        ),
                        text=[f'{v:,.0f}' for v in top_days['predicted_visitors']],
                        textposition='outside'
                    )
                ])
                
                fig_top.update_layout(
                    height=500,
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    xaxis=dict(title="Predicted Visitors"),
                    yaxis=dict(title="", autorange="reversed")
                )
                
                st.plotly_chart(fig_top, use_container_width=True)
            
            with col2:
                st.markdown("#### 📋 Top 10 Days")
                for idx, row in top_days.iterrows():
                    st.markdown(f"""
                    <div style="background: white; padding: 1rem; border-radius: 8px;
                                margin-bottom: 0.5rem; border-left: 4px solid #667eea;">
                        <strong>{row['date'].strftime('%b %d')}</strong><br>
                        <span style="color: #667eea; font-size: 1.3rem; font-weight: bold;">
                            {row['predicted_visitors']:,.0f}
                        </span> visitors<br>
                        <span style="color: #666; font-size: 0.9rem;">
                            {row['predicted_total_sales']:,.0f} QAR
                        </span>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Day of week pattern
            st.markdown("#### 📊 Day of Week Patterns")
            
            dow_pattern = predictions_df.groupby('day_of_week')['predicted_visitors'].mean().reindex([
                'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
            ])
            
            fig_dow = go.Figure(data=[
                go.Bar(
                    x=dow_pattern.index,
                    y=dow_pattern.values,
                    marker=dict(
                        color=dow_pattern.values,
                        colorscale='Viridis',
                        showscale=False
                    ),
                    text=[f'{v:,.0f}' for v in dow_pattern.values],
                    textposition='outside'
                )
            ])
            
            fig_dow.update_layout(
                height=400,
                plot_bgcolor='white',
                paper_bgcolor='white',
                xaxis=dict(title="Day of Week"),
                yaxis=dict(title="Average Visitors")
            )
            
            st.plotly_chart(fig_dow, use_container_width=True)
            
            # Recommendations
            st.markdown("#### 💡 Strategic Recommendations")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                <div class="recommendation-card">
                    <h4>👥 Staffing</h4>
                    <p>Increase staff by 30-40% on peak days</p>
                    <p style="color: #667eea; font-weight: bold;">
                        ✓ Review top 10 peak days and schedule accordingly
                    </p>
                </div>
                
                <div class="recommendation-card">
                    <h4>📦 Inventory</h4>
                    <p>Stock gift shops 2-3 days before predicted peak periods</p>
                    <p style="color: #667eea; font-weight: bold;">
                        ✓ Focus on high-margin items during busy periods
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class="recommendation-card">
                    <h4>🎫 Pricing Strategy</h4>
                    <p>Consider dynamic pricing for peak vs off-peak periods</p>
                    <p style="color: #667eea; font-weight: bold;">
                        ✓ Offer weekday discounts to balance demand
                    </p>
                </div>
                
                <div class="recommendation-card">
                    <h4>📱 Marketing</h4>
                    <p>Launch campaigns 7-10 days before low-traffic periods</p>
                    <p style="color: #667eea; font-weight: bold;">
                        ✓ Target local visitors during slower weekdays
                    </p>
                </div>
                """, unsafe_allow_html=True)
        
        # TAB 5: Download Report
        with tab5:
            st.markdown('<div class="section-title">📥 Download Comprehensive Report</div>', unsafe_allow_html=True)
            
            st.markdown("""
            <div class="success-box">
                <h3 style="margin-top: 0;">📊 Your Report Includes:</h3>
                <ul style="font-size: 1.1rem; line-height: 2;">
                    <li>✅ Daily predictions for all museums</li>
                    <li>✅ Monthly summaries with totals</li>
                    <li>✅ Museum-by-museum breakdown</li>
                    <li>✅ Revenue forecasts by category</li>
                    <li>✅ Summary statistics</li>
                    <li>✅ Excel format for easy analysis</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # Generate report
            excel_data = create_excel_report(df, predictions_df)
            
            # Download button
            st.download_button(
                label="📥 Download Excel Report",
                data=excel_data,
                file_name=f"qatar_museums_forecast_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Preview data
            st.markdown("#### 👀 Report Preview")
            
            preview_tab1, preview_tab2, preview_tab3 = st.tabs([
                "Daily Forecast",
                "Monthly Summary",
                "Museum Summary"
            ])
            
            with preview_tab1:
                st.dataframe(
                    predictions_df[[
                        'date', 'museum', 'day_of_week', 'predicted_visitors',
                        'predicted_total_sales'
                    ]].head(20),
                    use_container_width=True,
                    hide_index=True
                )
            
            with preview_tab2:
                monthly_summary = predictions_df.groupby(
                    [predictions_df['date'].dt.strftime('%B %Y'), 'museum']
                ).agg({
                    'predicted_visitors': 'sum',
                    'predicted_total_sales': 'sum'
                }).reset_index()
                monthly_summary.columns = ['Month', 'Museum', 'Visitors', 'Sales (QAR)']
                
                st.dataframe(
                    monthly_summary,
                    use_container_width=True,
                    hide_index=True
                )
            
            with preview_tab3:
                museum_summary_df = predictions_df.groupby('museum').agg({
                    'predicted_visitors': 'sum',
                    'predicted_total_sales': 'sum',
                    'predicted_ticket_sales': 'sum',
                    'predicted_giftshop_sales': 'sum',
                    'predicted_cafe_sales': 'sum'
                }).reset_index()
                
                museum_summary_df.columns = [
                    'Museum', 'Total Visitors', 'Total Sales',
                    'Ticket Sales', 'Gift Shop Sales', 'Cafe Sales'
                ]
                
                st.dataframe(
                    museum_summary_df,
                    use_container_width=True,
                    hide_index=True
                )
            
            # Final summary
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"""
            <div class="insight-box">
                <h3 style="margin-top: 0;">📈 Forecast Summary</h3>
                <p style="font-size: 1.2rem; line-height: 1.8;">
                    <strong>Period:</strong> {predictions_df['date'].min().strftime('%B %d, %Y')}
                    to {predictions_df['date'].max().strftime('%B %d, %Y')}<br>
                    <strong>Total Predicted Visitors:</strong> {total_pred_visitors:,.0f}<br>
                    <strong>Total Predicted Revenue:</strong> {total_pred_sales:,.2f} QAR<br>
                    <strong>Average Daily Visitors:</strong> {avg_daily_pred:,.0f}<br>
                    <strong>Models Used:</strong> Prophet (AI) + Random Forest (ML) + Ensemble
                </p>
            </div>
            """, unsafe_allow_html=True)

    # Footer
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; color: white; padding: 2rem;
                background: rgba(255,255,255,0.1); border-radius: 10px;">
        <p style="margin: 0; font-size: 1.1rem;">
            🏛️ Qatar Museums Forecasting Dashboard |
            Powered by AI & Machine Learning |
            © 2025
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()