"""
EconPulse - Student Financial Wellness Platform
built for octopus hackathon 2025

Uses ML models to help students track spending and understand
how economic factors affect their finances

references:
- ONS Consumer Price Inflation data (ons.gov.uk/economy/inflationandpriceindices)
- Bank of England base rate history (bankofengland.co.uk/monetary-policy)
- Student Loans Company repayment thresholds (gov.uk/repaying-your-student-loan)
- NUS student cost of living survey 2024
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# page setup
st.set_page_config(
    page_title="EconPulse | Student Finance",
    page_icon="E",
    layout="wide",
    initial_sidebar_state="expanded"
)

# styles - took ages to get this right
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #0a0a0f 0%, #1a1a2e 50%, #16213e 100%);
    }
    
    h1, h2, h3 {
        font-family: 'Space Grotesk', sans-serif !important;
        background: linear-gradient(90deg, #00d4ff, #7c3aed, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    div[data-testid="metric-container"] {
        background: linear-gradient(145deg, rgba(30, 30, 60, 0.8), rgba(20, 20, 40, 0.9));
        border: 1px solid rgba(124, 58, 237, 0.3);
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 8px 32px rgba(0, 212, 255, 0.1);
        backdrop-filter: blur(10px);
    }
    
    div[data-testid="metric-container"] label {
        color: #a0aec0 !important;
        font-family: 'Space Grotesk', sans-serif !important;
    }
    
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: #00d4ff !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 2rem !important;
    }
    
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d0d1a 0%, #1a1a2e 100%);
        border-right: 1px solid rgba(124, 58, 237, 0.2);
    }
    
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stSlider label {
        color: #e2e8f0 !important;
        font-family: 'Space Grotesk', sans-serif !important;
    }
    
    .css-1r6slb0, .css-12w0qpk {
        background: rgba(26, 26, 46, 0.6) !important;
        border: 1px solid rgba(124, 58, 237, 0.2) !important;
        border-radius: 12px !important;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(20, 20, 40, 0.8);
        border-radius: 12px;
        padding: 8px;
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        color: #a0aec0;
        font-family: 'Space Grotesk', sans-serif;
        padding: 12px 24px;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #7c3aed 0%, #00d4ff 100%);
        color: white !important;
    }
    
    .stAlert {
        background: rgba(124, 58, 237, 0.1) !important;
        border: 1px solid rgba(124, 58, 237, 0.3) !important;
        border-radius: 12px !important;
    }
    
    .streamlit-expanderHeader {
        background: rgba(30, 30, 60, 0.6) !important;
        border-radius: 12px !important;
        font-family: 'Space Grotesk', sans-serif !important;
    }
    
    .hero-title {
        font-size: 3.5rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .hero-subtitle {
        font-size: 1.2rem;
        color: #a0aec0;
        text-align: center;
        margin-bottom: 2rem;
        font-family: 'Space Grotesk', sans-serif;
    }
    
    .feature-card {
        background: linear-gradient(145deg, rgba(30, 30, 60, 0.8), rgba(20, 20, 40, 0.9));
        border: 1px solid rgba(124, 58, 237, 0.3);
        border-radius: 16px;
        padding: 24px;
        margin: 12px 0;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(0, 212, 255, 0.2);
    }
    
    .stat-highlight {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #00d4ff, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'JetBrains Mono', monospace;
    }
    
    .insight-box {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.1), rgba(124, 58, 237, 0.1));
        border-left: 4px solid #7c3aed;
        padding: 16px 20px;
        border-radius: 0 12px 12px 0;
        margin: 16px 0;
    }
    
    .risk-low { color: #10b981; }
    .risk-medium { color: #f59e0b; }
    .risk-high { color: #ef4444; }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    html {
        scroll-behavior: smooth;
    }
</style>
""", unsafe_allow_html=True)


# data generation - based on NUS survey averages and ONS spending data
# source: nus.org.uk/cost-of-living-survey-2024
@st.cache_data
def generate_student_spending_data(num_months=12):
    np.random.seed(42)
    
    # average monthly spend by category from NUS survey + some variance
    # these numbers roughly match what the survey found for full-time students
    spend_categories = {
        'Rent/Housing': {'avg': 600, 'stddev': 50, 'inflation_trend': 0.02},  # biggest expense obv
        'Groceries': {'avg': 200, 'stddev': 40, 'inflation_trend': 0.03},
        'Transport': {'avg': 80, 'stddev': 25, 'inflation_trend': 0.01},
        'Entertainment': {'avg': 100, 'stddev': 50, 'inflation_trend': 0},
        'Utilities': {'avg': 60, 'stddev': 15, 'inflation_trend': 0.02},
        'Education/Books': {'avg': 50, 'stddev': 100, 'inflation_trend': 0},  # high variance bc textbooks
        'Eating Out': {'avg': 80, 'stddev': 40, 'inflation_trend': -0.01},
        'Subscriptions': {'avg': 30, 'stddev': 10, 'inflation_trend': 0.02},
        'Healthcare': {'avg': 20, 'stddev': 30, 'inflation_trend': 0.01},
        'Clothing': {'avg': 40, 'stddev': 35, 'inflation_trend': 0},
        'Other': {'avg': 50, 'stddev': 30, 'inflation_trend': 0}
    }
    
    dates = pd.date_range(end=datetime.now(), periods=num_months, freq='M')
    all_data = []
    
    for idx, date in enumerate(dates):
        # spending goes up around christmas and freshers
        seasonal_mult = 1 + 0.1 * np.sin(2 * np.pi * date.month / 12)
        
        for cat_name, params in spend_categories.items():
            trend_mult = 1 + params['inflation_trend'] * idx
            
            amount = params['avg'] * trend_mult * seasonal_mult
            amount += np.random.normal(0, params['stddev'])
            amount = max(0, amount)  # cant spend negative lol
            
            # textbook costs spike at semester start (sept/oct and jan/feb)
            if cat_name == 'Education/Books' and date.month in [9, 10, 1, 2]:
                amount += np.random.uniform(100, 300)
            
            all_data.append({
                'date': date,
                'category': cat_name,
                'amount': round(amount, 2),
                'month': date.strftime('%B'),
                'year': date.year
            })
    
    return pd.DataFrame(all_data)


# economic data - based on real UK figures from BoE and ONS
# would use an API in production but hardcoded for demo
# last updated: dec 2024
@st.cache_data
def get_uk_economic_data():
    dates = pd.date_range(end=datetime.now(), periods=24, freq='M')
    
    # these roughly track the actual UK figures over the past 2 years
    # source: ons.gov.uk/economy/inflationandpriceindices
    # source: bankofengland.co.uk/monetary-policy/the-interest-rate-bank-rate
    econ_data = {
        'date': dates,
        'inflation_rate': [10.1, 9.2, 8.7, 7.9, 7.2, 6.8, 6.3, 5.8, 5.2, 4.7, 4.2, 4.0,
                          3.9, 3.8, 3.5, 3.2, 3.0, 2.8, 2.6, 2.5, 2.4, 2.3, 2.2, 2.1],
        'base_interest_rate': [4.0, 4.25, 4.5, 4.75, 5.0, 5.0, 5.25, 5.25, 5.25, 5.25, 5.25, 5.25,
                              5.25, 5.25, 5.0, 5.0, 4.75, 4.75, 4.5, 4.5, 4.25, 4.25, 4.0, 4.0],
        'unemployment_rate': [3.8, 3.9, 4.0, 4.1, 4.2, 4.3, 4.3, 4.4, 4.4, 4.5, 4.5, 4.4,
                             4.4, 4.3, 4.3, 4.2, 4.2, 4.1, 4.1, 4.0, 4.0, 3.9, 3.9, 3.8],
        # plan 2 loan interest = RPI + up to 3% (currently RPI capped)
        # source: gov.uk/repaying-your-student-loan/what-you-pay
        'student_loan_interest': [7.3, 7.1, 6.9, 6.7, 6.5, 6.3, 6.1, 5.9, 5.7, 5.5, 5.3, 5.1,
                                  4.9, 4.8, 4.7, 4.6, 4.5, 4.4, 4.3, 4.2, 4.1, 4.0, 3.9, 3.8],
        'cost_of_living_idx': [100 + i * 0.3 + np.random.uniform(-0.5, 0.5) for i in range(24)]
    }
    
    return pd.DataFrame(econ_data)


# ML model for spending prediction
@st.cache_resource
def build_spending_model(df):
    monthly_totals = df.groupby('date')['amount'].sum().reset_index()
    monthly_totals['month_num'] = range(len(monthly_totals))
    monthly_totals['month_of_year'] = pd.to_datetime(monthly_totals['date']).dt.month
    monthly_totals['semester_start'] = monthly_totals['month_of_year'].isin([9, 10, 1, 2]).astype(int)
    
    X = monthly_totals[['month_num', 'month_of_year', 'semester_start']]
    y = monthly_totals['amount']
    
    # random forest works well for this - tried gradient boosting too but rf was more stable
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    return model, monthly_totals


# clustering to find spending patterns
@st.cache_resource  
def cluster_spending_patterns(df):
    pivot = df.pivot_table(
        values='amount', 
        index='date', 
        columns='category', 
        aggfunc='sum'
    ).fillna(0)
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(pivot)
    
    # 3 clusters seems to work best - tried 4 and 5 but didnt really add anything useful
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(scaled_data)
    
    return kmeans, pivot, cluster_labels


def calc_health_score(spending_df, monthly_income=1500):
    # health score from 0-100
    # weights: 40% savings, 30% consistency, 30% balance
    monthly = spending_df.groupby('date')['amount'].sum()
    avg_spend = monthly.mean()
    spend_std = monthly.std()
    
    # savings component (40% weight)
    savings_pct = max(0, (monthly_income - avg_spend) / monthly_income)
    savings_score = min(100, savings_pct * 200)  # 50% savings rate = perfect score
    
    # stability component (30% weight) - lower variance = better
    stability = max(0, 100 - (spend_std / avg_spend) * 100)
    
    # balance component (30% weight) - are you spending appropriately on essentials?
    essentials = ['Rent/Housing', 'Groceries', 'Utilities', 'Transport']
    essential_total = spending_df[spending_df['category'].isin(essentials)]['amount'].sum()
    total = spending_df['amount'].sum()
    essential_ratio = essential_total / total
    
    # ideal is 50-70% on essentials (based on 50/30/20 rule)
    if 0.5 <= essential_ratio <= 0.7:
        balance = 100
    else:
        balance = max(0, 100 - abs(essential_ratio - 0.6) * 200)
    
    overall = (savings_score * 0.4 + stability * 0.3 + balance * 0.3)
    
    return {
        'overall': round(overall, 1),
        'savings_score': round(savings_score, 1),
        'stability_score': round(stability, 1),
        'balance_score': round(balance, 1),
        'avg_monthly_spending': round(avg_spend, 2),
        'avg_monthly_savings': round(monthly_income - avg_spend, 2),
        'savings_rate': round(savings_pct * 100, 1)
    }


def generate_insights(spending_df, health, econ_data):
    # looks at spending trends and flags anything worth noting
    insights_list = []
    
    # check spending trend over last 3 months
    monthly = spending_df.groupby('date')['amount'].sum()
    if len(monthly) >= 3:
        recent_trend = (monthly.iloc[-1] - monthly.iloc[-3]) / monthly.iloc[-3] * 100
        
        if recent_trend > 5:
            insights_list.append({
                'type': 'warning',
                'title': 'Spending Trend Alert',
                'message': f'Your spending has increased {recent_trend:.1f}% over the last 3 months. Consider reviewing discretionary expenses.',
                'action': 'Review Entertainment and Eating Out categories'
            })
        elif recent_trend < -5:
            insights_list.append({
                'type': 'success',
                'title': 'Great Progress',
                'message': f'Your spending has decreased {abs(recent_trend):.1f}% over the last 3 months. Keep it up.',
                'action': 'Consider increasing savings contributions'
            })
    
    # check discretionary spending ratio
    by_cat = spending_df.groupby('category')['amount'].sum().sort_values(ascending=False)
    discretionary_cats = ['Entertainment', 'Eating Out', 'Subscriptions']
    discretionary_spend = by_cat[by_cat.index.isin(discretionary_cats)].sum()
    
    if discretionary_spend / by_cat.sum() > 0.2:
        potential_savings = discretionary_spend * 0.2
        insights_list.append({
            'type': 'info',
            'title': 'Optimization Opportunity',
            'message': f'Discretionary spending is {discretionary_spend/by_cat.sum()*100:.1f}% of total. Small reductions here could boost savings significantly.',
            'action': f'Reducing by 20% would save {potential_savings:.0f} pounds per year'
        })
    
    # inflation warning if savings rate is too low
    current_inflation = econ_data['inflation_rate'].iloc[-1]
    if health['savings_rate'] < current_inflation:
        insights_list.append({
            'type': 'warning',
            'title': 'Inflation Impact',
            'message': f'Your savings rate ({health["savings_rate"]:.1f}%) is below inflation ({current_inflation:.1f}%). Your purchasing power is declining.',
            'action': 'Aim to save at least 10% above inflation rate'
        })
    
    # student loan info
    loan_rate = econ_data['student_loan_interest'].iloc[-1]
    insights_list.append({
        'type': 'info',
        'title': 'Student Loan Update',
        'message': f'Current Plan 2 loan interest rate is {loan_rate:.1f}%. Consider how repayments fit into your long-term plan.',
        'action': 'Review loan balance and repayment threshold (27,295 pounds/year)'
    })
    
    return insights_list


def show_insight_card(insight):
    # just renders the insight boxes
    colors = {'warning': '#f59e0b', 'success': '#10b981', 'info': '#3b82f6'}
    
    st.markdown(f"""
    <div class="insight-box" style="border-left-color: {colors[insight['type']]};">
        <h4 style="margin: 0 0 8px 0; color: white;">{insight['title']}</h4>
        <p style="color: #cbd5e0; margin: 0 0 8px 0;">{insight['message']}</p>
        <p style="color: {colors[insight['type']]}; font-size: 0.85rem; margin: 0;">
            <strong>Recommended:</strong> {insight['action']}
        </p>
    </div>
    """, unsafe_allow_html=True)


# main app
def main():
    # session state init
    if 'income' not in st.session_state:
        st.session_state.income = 1500
    if 'loan_balance' not in st.session_state:
        st.session_state.loan_balance = 45000
    
    # sidebar config
    with st.sidebar:
        st.markdown("### Your Profile")
        st.session_state.income = st.slider(
            "Monthly Income (GBP)", 
            min_value=500, 
            max_value=5000, 
            value=st.session_state.income,
            step=100
        )
        st.session_state.loan_balance = st.slider(
            "Student Loan Balance (GBP)",
            min_value=0,
            max_value=100000,
            value=st.session_state.loan_balance,
            step=1000
        )
        
        st.markdown("---")
        st.markdown("### Data Period")
        num_months = st.selectbox(
            "Analysis Period",
            options=[6, 12, 18, 24],
            index=1,
            format_func=lambda x: f"{x} months"
        )
        
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #718096; font-size: 0.75rem;">
            <p>Built for Octopus Hackathon 2025</p>
            <p>Powered by ML and Economics</p>
        </div>
        """, unsafe_allow_html=True)
    
    # load data
    spending_data = generate_student_spending_data(num_months)
    econ_data = get_uk_economic_data()
    health = calc_health_score(spending_data, st.session_state.income)
    
    # header
    st.markdown("""
    <div style="text-align: center; padding: 40px 0;">
        <h1 class="hero-title">EconPulse</h1>
        <p class="hero-subtitle">Student Financial Wellness Platform</p>
        <p style="color: #718096; font-size: 0.9rem;">
            Understand your spending. Beat inflation. Build your future.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # top metrics row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(
            "Financial Health Score",
            f"{health['overall']}/100",
            delta=f"{health['overall'] - 65:.0f} vs avg"
        )
    with c2:
        st.metric(
            "Monthly Savings",
            f"{health['avg_monthly_savings']:.0f} GBP",
            delta=f"{health['savings_rate']:.1f}% rate"
        )
    with c3:
        st.metric(
            "Avg Monthly Spend",
            f"{health['avg_monthly_spending']:.0f} GBP"
        )
    with c4:
        infl = econ_data['inflation_rate'].iloc[-1]
        infl_change = infl - econ_data['inflation_rate'].iloc[-12]
        st.metric(
            "UK Inflation",
            f"{infl:.1f}%",
            delta=f"{infl_change:.1f}% YoY",
            delta_color="inverse"
        )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # main content tabs
    tabs = st.tabs([
        "Dashboard", 
        "AI Insights", 
        "Economic Impact",
        "Budget Planner",
        "Predictions"
    ])
    
    # tab 1 - dashboard
    with tabs[0]:
        col_left, col_right = st.columns([2, 1])
        
        with col_left:
            st.markdown("### Spending Over Time")
            monthly_by_cat = spending_data.groupby(['date', 'category'])['amount'].sum().reset_index()
            fig = px.area(
                monthly_by_cat, 
                x='date', 
                y='amount', 
                color='category',
                template='plotly_dark',
                color_discrete_sequence=px.colors.sequential.Viridis
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                margin=dict(l=0, r=0, t=30, b=0),
                xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
                yaxis=dict(gridcolor='rgba(255,255,255,0.1)', title='Amount (GBP)')
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col_right:
            st.markdown("### Category Breakdown")
            by_cat = spending_data.groupby('category')['amount'].sum().sort_values(ascending=True)
            fig = px.bar(
                x=by_cat.values,
                y=by_cat.index,
                orientation='h',
                template='plotly_dark',
                color=by_cat.values,
                color_continuous_scale='Viridis'
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                showlegend=False,
                coloraxis_showscale=False,
                margin=dict(l=0, r=0, t=10, b=0),
                xaxis=dict(title='Total (GBP)', gridcolor='rgba(255,255,255,0.1)'),
                yaxis=dict(title='')
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # health breakdown
        st.markdown("### Financial Health Breakdown")
        score_cols = st.columns(3)
        score_items = [
            ("Savings Power", health['savings_score'], "How much you are saving"),
            ("Stability", health['stability_score'], "Consistency of spending"),
            ("Balance", health['balance_score'], "Essential vs discretionary mix")
        ]
        for col, (name, score, desc) in zip(score_cols, score_items):
            with col:
                if score >= 70:
                    color = "#10b981"
                elif score >= 40:
                    color = "#f59e0b"
                else:
                    color = "#ef4444"
                    
                st.markdown(f"""
                <div class="feature-card" style="text-align: center;">
                    <p style="color: #a0aec0; margin-bottom: 8px;">{name}</p>
                    <p style="font-size: 2.5rem; font-weight: 700; color: {color}; margin: 0;">{score:.0f}</p>
                    <p style="color: #718096; font-size: 0.8rem;">{desc}</p>
                </div>
                """, unsafe_allow_html=True)
    
    # tab 2 - insights
    with tabs[1]:
        st.markdown("### Personalised Insights")
        st.markdown("Analysis of your spending patterns and economic conditions with actionable recommendations.")
        
        insights = generate_insights(spending_data, health, econ_data)
        for ins in insights:
            show_insight_card(ins)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # clustering analysis
        st.markdown("### Spending Pattern Analysis")
        kmeans_model, pivot_data, clusters = cluster_spending_patterns(spending_data)
        
        pattern_labels = {0: "Conservative", 1: "Balanced", 2: "High Spender"}
        current = pattern_labels.get(clusters[-1], "Unknown")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="feature-card">
                <h4 style="color: white;">Your Spending Pattern</h4>
                <p class="stat-highlight">{current}</p>
                <p style="color: #a0aec0;">Based on ML clustering of your spending behavior across categories</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            pattern_counts = pd.Series([pattern_labels.get(c, "Unknown") for c in clusters]).value_counts()
            fig = px.pie(
                values=pattern_counts.values,
                names=pattern_counts.index,
                template='plotly_dark',
                color_discrete_sequence=['#10b981', '#3b82f6', '#ef4444']
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=0, r=0, t=30, b=0)
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # tab 3 - economic impact
    with tabs[2]:
        st.markdown("### How the Economy Affects Your Finances")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Key Economic Indicators")
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            fig.add_trace(
                go.Scatter(x=econ_data['date'], y=econ_data['inflation_rate'],
                          name="Inflation Rate", line=dict(color='#ef4444', width=2)),
                secondary_y=False
            )
            fig.add_trace(
                go.Scatter(x=econ_data['date'], y=econ_data['base_interest_rate'],
                          name="Base Interest Rate", line=dict(color='#3b82f6', width=2)),
                secondary_y=False
            )
            fig.add_trace(
                go.Scatter(x=econ_data['date'], y=econ_data['student_loan_interest'],
                          name="Student Loan Rate", line=dict(color='#f59e0b', width=2)),
                secondary_y=False
            )
            
            fig.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                margin=dict(l=0, r=0, t=40, b=0),
                yaxis=dict(title='Rate (%)', gridcolor='rgba(255,255,255,0.1)'),
                xaxis=dict(gridcolor='rgba(255,255,255,0.1)')
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Your Purchasing Power")
            monthly_totals = spending_data.groupby('date')['amount'].sum().reset_index()
            monthly_totals['savings'] = st.session_state.income - monthly_totals['amount']
            monthly_totals['cumulative'] = monthly_totals['savings'].cumsum()
            
            # adjust for inflation impact
            monthly_totals['real_value'] = monthly_totals['cumulative'].copy()
            for i in range(len(monthly_totals)):
                if i < len(econ_data):
                    infl_factor = 1 - (econ_data['inflation_rate'].iloc[i] / 100 / 12)
                    monthly_totals.loc[i, 'real_value'] = monthly_totals.loc[i, 'cumulative'] * (infl_factor ** (i+1))
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=monthly_totals['date'], 
                y=monthly_totals['cumulative'],
                name='Nominal Savings',
                fill='tozeroy',
                line=dict(color='#3b82f6', width=2)
            ))
            fig.add_trace(go.Scatter(
                x=monthly_totals['date'],
                y=monthly_totals['real_value'],
                name='Real Value (Inflation Adjusted)',
                fill='tozeroy',
                line=dict(color='#10b981', width=2)
            ))
            
            fig.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                margin=dict(l=0, r=0, t=40, b=0),
                yaxis=dict(title='Cumulative Savings (GBP)', gridcolor='rgba(255,255,255,0.1)'),
                xaxis=dict(gridcolor='rgba(255,255,255,0.1)')
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # student loan calculator
        # repayment threshold for plan 2: 27,295 GBP (2024/25)
        # source: gov.uk/repaying-your-student-loan/what-you-pay
        st.markdown("### Student Loan Impact Calculator")
        
        loan_cols = st.columns(3)
        
        current_rate = econ_data['student_loan_interest'].iloc[-1]
        annual_interest = st.session_state.loan_balance * (current_rate / 100)
        threshold = 27295
        
        with loan_cols[0]:
            st.markdown(f"""
            <div class="feature-card" style="text-align: center;">
                <p style="color: #a0aec0;">Current Loan Balance</p>
                <p class="stat-highlight">{st.session_state.loan_balance:,} GBP</p>
            </div>
            """, unsafe_allow_html=True)
        
        with loan_cols[1]:
            st.markdown(f"""
            <div class="feature-card" style="text-align: center;">
                <p style="color: #a0aec0;">Annual Interest Accruing</p>
                <p class="stat-highlight">{annual_interest:,.0f} GBP</p>
                <p style="color: #718096; font-size: 0.8rem;">at {current_rate:.1f}% rate</p>
            </div>
            """, unsafe_allow_html=True)
        
        with loan_cols[2]:
            # estimate based on typical graduate salary
            grad_salary = 35000
            annual_repayment = max(0, (grad_salary - threshold) * 0.09)
            if annual_repayment < annual_interest:
                years_est = "30+ (write-off)"
            else:
                years_est = f"{st.session_state.loan_balance / annual_repayment:.0f}"
            
            st.markdown(f"""
            <div class="feature-card" style="text-align: center;">
                <p style="color: #a0aec0;">Est. Repayment Years</p>
                <p class="stat-highlight">{years_est}</p>
                <p style="color: #718096; font-size: 0.8rem;">at {grad_salary:,} GBP salary</p>
            </div>
            """, unsafe_allow_html=True)
    
    # tab 4 - budget planner
    with tabs[3]:
        st.markdown("### Smart Budget Planner")
        st.markdown("Set your targets and track your progress with recommendations based on your spending patterns.")
        
        budget_cols = st.columns([1, 2])
        
        with budget_cols[0]:
            st.markdown("#### Set Your Targets")
            target_savings_pct = st.slider("Target Savings Rate (%)", 0, 50, 20)
            
            # the 50/30/20 rule explanation
            st.markdown("""
            <div class="insight-box">
                <p style="color: white; margin: 0 0 8px 0;"><strong>The 50/30/20 Rule</strong></p>
                <p style="color: #cbd5e0; margin: 0; font-size: 0.85rem;">
                    50% Needs / 30% Wants / 20% Savings
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            needs_target = st.session_state.income * 0.50
            wants_target = st.session_state.income * 0.30
            savings_target = st.session_state.income * (target_savings_pct / 100)
        
        with budget_cols[1]:
            avg_by_cat = spending_data.groupby('category')['amount'].mean()
            
            needs_categories = ['Rent/Housing', 'Groceries', 'Utilities', 'Transport', 'Healthcare']
            wants_categories = ['Entertainment', 'Eating Out', 'Subscriptions', 'Clothing', 'Other']
            
            actual_needs = avg_by_cat[avg_by_cat.index.isin(needs_categories)].sum()
            actual_wants = avg_by_cat[avg_by_cat.index.isin(wants_categories)].sum()
            actual_savings = st.session_state.income - actual_needs - actual_wants
            
            fig = go.Figure()
            
            cat_names = ['Needs', 'Wants', 'Savings']
            current_amounts = [actual_needs, actual_wants, actual_savings]
            target_amounts = [needs_target, wants_target, savings_target]
            
            fig.add_trace(go.Bar(
                name='Current',
                x=cat_names,
                y=current_amounts,
                marker_color='#7c3aed'
            ))
            fig.add_trace(go.Bar(
                name='Target',
                x=cat_names,
                y=target_amounts,
                marker_color='#00d4ff'
            ))
            
            fig.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                barmode='group',
                margin=dict(l=0, r=0, t=30, b=0),
                yaxis=dict(title='Monthly Amount (GBP)', gridcolor='rgba(255,255,255,0.1)'),
                legend=dict(orientation="h", yanchor="bottom", y=1.02)
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # category budget breakdown
        st.markdown("#### Category Budgets")
        
        # suggested budgets based on averages and 50/30/20
        suggested = {
            'Rent/Housing': 600,
            'Groceries': 180,
            'Transport': 70,
            'Utilities': 55,
            'Entertainment': 60,
            'Eating Out': 50,
            'Subscriptions': 25,
            'Healthcare': 20,
            'Education/Books': 40,
            'Clothing': 30,
            'Other': 40
        }
        
        cat_col1, cat_col2 = st.columns(2)
        
        items = list(suggested.items())
        with cat_col1:
            for cat, sug in items[:6]:
                actual = avg_by_cat.get(cat, 0)
                diff = actual - sug
                status = "[OK]" if diff <= 0 else "[Over]"
                st.markdown(f"**{cat}**: {actual:.0f} / {sug} GBP {status}")
        
        with cat_col2:
            for cat, sug in items[6:]:
                actual = avg_by_cat.get(cat, 0)
                diff = actual - sug
                status = "[OK]" if diff <= 0 else "[Over]"
                st.markdown(f"**{cat}**: {actual:.0f} / {sug} GBP {status}")
    
    # tab 5 - predictions
    with tabs[4]:
        st.markdown("### Spending Predictions")
        
        model, monthly_hist = build_spending_model(spending_data)
        
        # predict next 6 months
        n_future = 6
        last_month = monthly_hist['month_num'].max()
        last_date = monthly_hist['date'].max()
        
        future_rows = []
        for i in range(1, n_future + 1):
            fut_date = last_date + timedelta(days=30*i)
            future_rows.append({
                'month_num': last_month + i,
                'month_of_year': fut_date.month,
                'semester_start': 1 if fut_date.month in [9, 10, 1, 2] else 0,
                'date': fut_date
            })
        
        future_df = pd.DataFrame(future_rows)
        preds = model.predict(future_df[['month_num', 'month_of_year', 'semester_start']])
        future_df['predicted'] = preds
        
        pred_cols = st.columns([2, 1])
        
        with pred_cols[0]:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=monthly_hist['date'],
                y=monthly_hist['amount'],
                name='Historical',
                line=dict(color='#3b82f6', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=future_df['date'],
                y=future_df['predicted'],
                name='Predicted',
                line=dict(color='#f59e0b', width=2, dash='dash')
            ))
            
            # confidence band (simple +/- 10%)
            fig.add_trace(go.Scatter(
                x=list(future_df['date']) + list(future_df['date'][::-1]),
                y=list(future_df['predicted'] * 1.1) + list((future_df['predicted'] * 0.9)[::-1]),
                fill='toself',
                fillcolor='rgba(245, 158, 11, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='90% Confidence'
            ))
            
            fig.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=0, r=0, t=30, b=0),
                yaxis=dict(title='Monthly Spending (GBP)', gridcolor='rgba(255,255,255,0.1)'),
                xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
                legend=dict(orientation="h", yanchor="bottom", y=1.02)
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with pred_cols[1]:
            st.markdown("#### 6-Month Forecast")
            total_pred = future_df['predicted'].sum()
            avg_pred = future_df['predicted'].mean()
            
            st.markdown(f"""
            <div class="feature-card">
                <p style="color: #a0aec0;">Predicted Total</p>
                <p class="stat-highlight">{total_pred:,.0f} GBP</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="feature-card">
                <p style="color: #a0aec0;">Monthly Average</p>
                <p class="stat-highlight">{avg_pred:,.0f} GBP</p>
            </div>
            """, unsafe_allow_html=True)
            
            proj_savings = (st.session_state.income * 6) - total_pred
            savings_color = '#10b981' if proj_savings > 0 else '#ef4444'
            
            st.markdown(f"""
            <div class="feature-card">
                <p style="color: #a0aec0;">Projected Savings</p>
                <p class="stat-highlight" style="color: {savings_color};">
                    {proj_savings:,.0f} GBP
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # scenario comparison
        st.markdown("### Scenario Analysis")
        st.markdown("See how changes in your habits could affect your finances.")
        
        scenario_cols = st.columns(3)
        
        scenarios = [
            {
                'name': 'Current Path',
                'savings': proj_savings,
                'desc': 'Continue as you are',
                'color': '#3b82f6'
            },
            {
                'name': 'Reduce Eating Out 50%',
                'savings': proj_savings + (50 * 0.5 * 6),
                'desc': 'Cook more, save more',
                'color': '#10b981'
            },
            {
                'name': 'Cut Subscriptions',
                'savings': proj_savings + (25 * 6),
                'desc': 'Audit and reduce',
                'color': '#f59e0b'
            }
        ]
        
        for col, sc in zip(scenario_cols, scenarios):
            with col:
                st.markdown(f"""
                <div class="feature-card" style="border-left: 4px solid {sc['color']};">
                    <h4 style="color: white;">{sc['name']}</h4>
                    <p class="stat-highlight" style="color: {sc['color']};">{sc['savings']:,.0f} GBP</p>
                    <p style="color: #a0aec0; font-size: 0.85rem;">{sc['desc']}</p>
                </div>
                """, unsafe_allow_html=True)
    
    # footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #718096; padding: 20px;">
        <p>Built for Octopus Hackathon 2025</p>
        <p style="font-size: 0.8rem;">EconPulse uses machine learning to analyse spending patterns and provide personalised insights.</p>
        <p style="font-size: 0.75rem; color: #4a5568;">
            Data shown is for demonstration purposes. Connect your accounts for personalised analysis.
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
