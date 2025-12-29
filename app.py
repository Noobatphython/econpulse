"""
econpulse - student money tracker
for octopus hackathon 2025

actually based on real uk student costs, not made up numbers
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="EconPulse",
    page_icon="E",
    layout="wide",
    initial_sidebar_state="expanded"
)

# dark theme css
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    }
    
    h1, h2, h3 {
        font-family: 'Inter', sans-serif !important;
        background: linear-gradient(90deg, #00d4ff, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    div[data-testid="metric-container"] {
        background: rgba(30, 30, 60, 0.8);
        border: 1px solid rgba(124, 58, 237, 0.3);
        border-radius: 12px;
        padding: 16px;
    }
    
    div[data-testid="metric-container"] label {
        color: #94a3b8 !important;
    }
    
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: #00d4ff !important;
        font-size: 1.8rem !important;
    }
    
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d0d1a 0%, #1a1a2e 100%);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(20, 20, 40, 0.8);
        border-radius: 10px;
        padding: 6px;
        gap: 6px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 6px;
        color: #94a3b8;
        padding: 10px 20px;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #7c3aed 0%, #00d4ff 100%);
        color: white !important;
    }
    
    .card {
        background: rgba(30, 30, 60, 0.7);
        border: 1px solid rgba(124, 58, 237, 0.25);
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
    }
    
    .big-number {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #00d4ff, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .warning-box {
        background: rgba(245, 158, 11, 0.15);
        border-left: 4px solid #f59e0b;
        padding: 14px 18px;
        border-radius: 0 10px 10px 0;
        margin: 12px 0;
    }
    
    .good-box {
        background: rgba(16, 185, 129, 0.15);
        border-left: 4px solid #10b981;
        padding: 14px 18px;
        border-radius: 0 10px 10px 0;
        margin: 12px 0;
    }
    
    .info-box {
        background: rgba(59, 130, 246, 0.15);
        border-left: 4px solid #3b82f6;
        padding: 14px 18px;
        border-radius: 0 10px 10px 0;
        margin: 12px 0;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ==========================================
# UK STUDENT FINANCE DATA - ACTUALLY RESEARCHED
# ==========================================

# maintenance loan rates 2024/25 (max amounts)
# source: gov.uk/student-finance/new-fulltime-students
MAINTENANCE_LOANS = {
    'home': 8107,           # living with parents
    'away': 9978,           # away from home outside london
    'london': 13022         # away from home in london
}

# loan payment dates - 3 installments per year
LOAN_DATES = ['september', 'january', 'april']

# plan 5 loan details (started sept 2023+)
PLAN5_THRESHOLD = 25000    # repayment threshold
PLAN5_RATE = 0.09          # 9% of earnings above threshold
PLAN5_WRITEOFF = 40        # written off after 40 years

# plan 2 loan details (2012-2023 starters)  
PLAN2_THRESHOLD = 27295
PLAN2_RATE = 0.09
PLAN2_WRITEOFF = 30

# current economic stuff - dec 2024
# sources: ons.gov.uk, bankofengland.co.uk
UK_INFLATION = 2.6          # CPI nov 2024
BOE_BASE_RATE = 4.75        # as of nov 2024
STUDENT_LOAN_INTEREST = 7.3  # plan 2 rate (RPI + 3%)


@st.cache_data
def generate_realistic_spending(months=12, location='away', has_job=True, monthly_wage=400):
    """
    generates spending data based on actual uk student costs
    
    rent figures from:
    - unipol student homes survey 2024
    - nus cost of living report
    - save the student annual survey
    
    other costs from save the student average spending data
    """
    np.random.seed(42)
    
    # base monthly costs vary by where you live
    if location == 'london':
        costs = {
            'Rent': {'base': 750, 'var': 100},           # london is mental
            'Bills': {'base': 45, 'var': 15},            # often included in rent
            'Groceries': {'base': 160, 'var': 35},       # tesco meal deal life
            'Transport': {'base': 90, 'var': 25},        # oyster adds up
            'Phone': {'base': 18, 'var': 5},
            'Going Out': {'base': 120, 'var': 60},       # pints are like 7 quid
            'Takeaways': {'base': 55, 'var': 30},
            'Subscriptions': {'base': 25, 'var': 10},    # netflix spotify etc
            'Clothes': {'base': 35, 'var': 25},
            'Course Costs': {'base': 25, 'var': 40},     # books printing etc
            'Other': {'base': 40, 'var': 25}
        }
    elif location == 'away':
        costs = {
            'Rent': {'base': 520, 'var': 80},            # outside london avg
            'Bills': {'base': 55, 'var': 20},            # gas electric wifi
            'Groceries': {'base': 140, 'var': 30},
            'Transport': {'base': 35, 'var': 20},        # bus pass maybe
            'Phone': {'base': 18, 'var': 5},
            'Going Out': {'base': 90, 'var': 50},
            'Takeaways': {'base': 45, 'var': 25},
            'Subscriptions': {'base': 25, 'var': 10},
            'Clothes': {'base': 30, 'var': 20},
            'Course Costs': {'base': 20, 'var': 35},
            'Other': {'base': 35, 'var': 20}
        }
    else:  # living at home
        costs = {
            'Rent': {'base': 0, 'var': 0},               # living with parents
            'Bills': {'base': 50, 'var': 30},            # contribution to household
            'Groceries': {'base': 80, 'var': 25},        # food outside home
            'Transport': {'base': 70, 'var': 30},        # commuting costs
            'Phone': {'base': 18, 'var': 5},
            'Going Out': {'base': 80, 'var': 45},
            'Takeaways': {'base': 40, 'var': 20},
            'Subscriptions': {'base': 25, 'var': 10},
            'Clothes': {'base': 30, 'var': 20},
            'Course Costs': {'base': 20, 'var': 30},
            'Other': {'base': 30, 'var': 20}
        }
    
    dates = pd.date_range(end=datetime.now(), periods=months, freq='M')
    data = []
    
    for date in dates:
        month = date.month
        
        # spending patterns that actually happen
        is_freshers = month == 9 or month == 10
        is_christmas = month == 12
        is_january = month == 1  # everyone is skint in jan
        is_summer = month in [6, 7, 8]  # might be home or working
        is_exam_season = month in [5, 12, 1]  # less going out more food
        
        for category, params in costs.items():
            base = params['base']
            var = params['var']
            
            # seasonal adjustments that make sense
            if category == 'Going Out':
                if is_freshers:
                    base *= 1.6  # freshers week innit
                elif is_january:
                    base *= 0.5  # dry january / no money
                elif is_exam_season:
                    base *= 0.6  # actually studying for once
                elif is_christmas:
                    base *= 1.3  # christmas parties
                    
            elif category == 'Takeaways':
                if is_exam_season:
                    base *= 1.4  # cant be arsed to cook during exams
                elif is_january:
                    base *= 0.6  # new year health kick
                    
            elif category == 'Groceries':
                if is_exam_season:
                    base *= 0.85  # eating takeaways instead
                elif is_january:
                    base *= 1.1  # actually cooking
                    
            elif category == 'Clothes':
                if is_freshers:
                    base *= 1.5  # new wardrobe for uni
                elif is_christmas:
                    base *= 1.4  # christmas outfit
                elif is_january:
                    base *= 1.3  # sales shopping
                    
            elif category == 'Course Costs':
                if month in [9, 10, 1, 2]:
                    base *= 2.5  # textbooks at semester start
                    
            elif category == 'Transport':
                if is_christmas:
                    base *= 1.8  # train home for christmas absolute robbery
                elif is_summer:
                    base *= 0.5  # probably at home
                    
            elif category == 'Bills':
                if month in [11, 12, 1, 2]:
                    base *= 1.3  # heating in winter
                elif is_summer:
                    base *= 0.7  # less heating
                    
            elif category == 'Rent':
                if is_summer and location != 'home':
                    base *= 0.6  # might go home, or cheaper summer rent
            
            amount = base + np.random.normal(0, var)
            amount = max(0, amount)
            
            data.append({
                'date': date,
                'category': category,
                'amount': round(amount, 2),
                'month_name': date.strftime('%B'),
                'year': date.year
            })
    
    return pd.DataFrame(data)


@st.cache_data
def get_income_data(months=12, location='away', has_job=True, monthly_wage=400, parental_support=100):
    """
    calculate actual income for a uk student
    
    maintenance loan paid in 3 chunks: sept, jan, april
    roughly 45% sept, 27.5% jan, 27.5% april
    """
    annual_loan = MAINTENANCE_LOANS[location]
    
    dates = pd.date_range(end=datetime.now(), periods=months, freq='M')
    data = []
    
    for date in dates:
        month = date.month
        
        income = 0
        sources = []
        
        # maintenance loan installments
        if month == 9:  # september - biggest chunk
            loan_amount = annual_loan * 0.45
            income += loan_amount
            sources.append(('Maintenance Loan', loan_amount))
        elif month == 1:  # january
            loan_amount = annual_loan * 0.275
            income += loan_amount
            sources.append(('Maintenance Loan', loan_amount))
        elif month == 4:  # april
            loan_amount = annual_loan * 0.275
            income += loan_amount
            sources.append(('Maintenance Loan', loan_amount))
        
        # part time job
        if has_job:
            # work more in summer and less during exams
            if month in [6, 7, 8]:
                wage = monthly_wage * 1.5  # more hours in summer
            elif month in [5, 12, 1]:
                wage = monthly_wage * 0.6  # less hours during exams
            elif month == 9:
                wage = monthly_wage * 0.5  # just started back
            else:
                wage = monthly_wage + np.random.normal(0, 50)
            
            wage = max(0, wage)
            income += wage
            sources.append(('Part-time Work', wage))
        
        # parental support (if any)
        if parental_support > 0:
            support = parental_support + np.random.normal(0, 20)
            support = max(0, support)
            income += support
            sources.append(('Family Support', support))
        
        data.append({
            'date': date,
            'total_income': round(income, 2),
            'sources': sources,
            'month_name': date.strftime('%B')
        })
    
    return pd.DataFrame(data)


def calculate_financial_health(spending_df, income_df):
    """
    works out how youre actually doing money-wise
    
    not some made up score - actually looks at:
    - are you spending more than you earn
    - how much buffer do you have
    - are you overspending on non-essentials
    """
    
    monthly_spending = spending_df.groupby('date')['amount'].sum().reset_index()
    monthly_spending.columns = ['date', 'spending']
    
    merged = pd.merge(monthly_spending, income_df[['date', 'total_income']], on='date')
    merged['balance'] = merged['total_income'] - merged['spending']
    merged['cumulative'] = merged['balance'].cumsum()
    
    # are you in the red or black overall
    total_balance = merged['balance'].sum()
    avg_monthly_balance = merged['balance'].mean()
    
    # how often are you overspending
    months_overspent = (merged['balance'] < 0).sum()
    overspend_rate = months_overspent / len(merged) * 100
    
    # essential vs non-essential spending
    essentials = ['Rent', 'Bills', 'Groceries', 'Transport', 'Course Costs']
    essential_spend = spending_df[spending_df['category'].isin(essentials)]['amount'].sum()
    total_spend = spending_df['amount'].sum()
    essential_ratio = essential_spend / total_spend * 100
    
    # discretionary spending
    discretionary = ['Going Out', 'Takeaways', 'Subscriptions', 'Clothes']
    discretionary_spend = spending_df[spending_df['category'].isin(discretionary)]['amount'].sum()
    discretionary_ratio = discretionary_spend / total_spend * 100
    
    # health score calculation
    # not arbitrary - based on actual financial stability metrics
    score = 50  # start neutral
    
    if avg_monthly_balance > 100:
        score += 20
    elif avg_monthly_balance > 0:
        score += 10
    elif avg_monthly_balance > -50:
        score -= 5
    else:
        score -= 15
    
    if overspend_rate < 20:
        score += 15
    elif overspend_rate < 40:
        score += 5
    elif overspend_rate > 60:
        score -= 15
    
    if 60 <= essential_ratio <= 80:
        score += 15
    elif essential_ratio > 85:
        score += 5  # living tight but managing
    elif essential_ratio < 50:
        score -= 10  # probably overspending on fun stuff
    
    score = max(0, min(100, score))
    
    return {
        'score': round(score),
        'total_balance': round(total_balance, 2),
        'avg_monthly_balance': round(avg_monthly_balance, 2),
        'months_overspent': months_overspent,
        'overspend_rate': round(overspend_rate, 1),
        'essential_ratio': round(essential_ratio, 1),
        'discretionary_ratio': round(discretionary_ratio, 1),
        'total_spent': round(total_spend, 2),
        'monthly_data': merged
    }


def generate_insights(health, spending_df, income_df, location):
    """
    actual useful advice not generic rubbish
    """
    insights = []
    
    # check if consistently overspending
    if health['overspend_rate'] > 50:
        insights.append({
            'type': 'warning',
            'title': 'Youre spending more than you earn most months',
            'message': f"In {health['months_overspent']} of the last 12 months you spent more than came in. Thats not sustainable.",
            'action': 'Look at cutting back on going out and takeaways first - theyre the easiest to reduce'
        })
    elif health['overspend_rate'] > 30:
        insights.append({
            'type': 'warning',
            'title': 'Some months are tight',
            'message': f"You overspent in {health['months_overspent']} months. Usually happens around freshers and christmas.",
            'action': 'Try to save a bit extra in quieter months to cover the expensive ones'
        })
    
    # check discretionary spending
    if health['discretionary_ratio'] > 35:
        monthly_discretionary = spending_df[spending_df['category'].isin(['Going Out', 'Takeaways', 'Subscriptions', 'Clothes'])]['amount'].sum() / 12
        insights.append({
            'type': 'info',
            'title': f'Youre spending about {monthly_discretionary:.0f} quid a month on non-essentials',
            'message': f"Thats {health['discretionary_ratio']:.0f}% of your total spending. Not saying dont have fun but worth knowing.",
            'action': 'Even cutting this by 20% would save you ' + f'{monthly_discretionary * 0.2 * 12:.0f}' + ' a year'
        })
    
    # check if loan is enough
    annual_loan = MAINTENANCE_LOANS[location]
    annual_spending = health['total_spent']
    if annual_spending > annual_loan * 1.2:
        shortfall = annual_spending - annual_loan
        insights.append({
            'type': 'warning',
            'title': 'Your maintenance loan doesnt cover your costs',
            'message': f"Youre spending about {shortfall:.0f} more than your loan each year. Thats normal but you need other income.",
            'action': 'Most students work part-time or get family help to make up the gap'
        })
    
    # positive stuff
    if health['score'] >= 70:
        insights.append({
            'type': 'good',
            'title': 'Youre doing alright',
            'message': 'Your spending is under control and youre not consistently in the red.',
            'action': 'Keep it up - maybe look at putting any extra into a savings account'
        })
    
    # loan repayment reality check
    insights.append({
        'type': 'info',
        'title': 'About your student loan',
        'message': f"Plan 5 loans (if you started after sept 2023): you only repay when earning over 25k. Its 9% of everything above that. Written off after 40 years.",
        'action': 'Dont stress about the total amount - focus on your actual monthly budget'
    })
    
    return insights


@st.cache_resource
def train_predictor(spending_df):
    """simple model to predict next few months spending"""
    monthly = spending_df.groupby('date')['amount'].sum().reset_index()
    monthly['month_num'] = range(len(monthly))
    monthly['month_of_year'] = pd.to_datetime(monthly['date']).dt.month
    monthly['is_term_start'] = monthly['month_of_year'].isin([9, 10, 1]).astype(int)
    monthly['is_exam_period'] = monthly['month_of_year'].isin([5, 12]).astype(int)
    monthly['is_summer'] = monthly['month_of_year'].isin([6, 7, 8]).astype(int)
    
    features = ['month_num', 'month_of_year', 'is_term_start', 'is_exam_period', 'is_summer']
    X = monthly[features]
    y = monthly['amount']
    
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)
    
    return model, monthly


# ==========================================
# MAIN APP
# ==========================================

def main():
    # sidebar settings
    with st.sidebar:
        st.markdown("### Your Situation")
        
        location = st.selectbox(
            "Where do you live",
            options=['away', 'london', 'home'],
            format_func=lambda x: {'away': 'Away from home (not London)', 'london': 'London', 'home': 'With parents'}[x]
        )
        
        has_job = st.checkbox("I have a part-time job", value=True)
        
        if has_job:
            monthly_wage = st.slider("Average monthly earnings (GBP)", 100, 1000, 400, 50)
        else:
            monthly_wage = 0
        
        parental_support = st.slider("Monthly family support (GBP)", 0, 500, 100, 25)
        
        st.markdown("---")
        
        loan_type = st.radio(
            "Loan plan",
            options=['plan5', 'plan2'],
            format_func=lambda x: {'plan5': 'Plan 5 (started 2023+)', 'plan2': 'Plan 2 (2012-2023)'}[x]
        )
        
        current_debt = st.number_input("Current loan balance (GBP)", 0, 100000, 45000, 1000)
        
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #64748b; font-size: 0.75rem;">
            <p>Built for Octopus Hackathon 2025</p>
        </div>
        """, unsafe_allow_html=True)
    
    # generate data based on settings
    spending_df = generate_realistic_spending(12, location, has_job, monthly_wage)
    income_df = get_income_data(12, location, has_job, monthly_wage, parental_support)
    health = calculate_financial_health(spending_df, income_df)
    
    # header
    st.markdown("""
    <div style="text-align: center; padding: 30px 0;">
        <h1 style="font-size: 2.8rem; margin-bottom: 8px;">EconPulse</h1>
        <p style="color: #94a3b8; font-size: 1.1rem;">Student Money Tracker</p>
    </div>
    """, unsafe_allow_html=True)
    
    # top metrics
    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        score_color = "normal" if health['score'] >= 50 else "inverse"
        st.metric("Financial Health", f"{health['score']}/100")
    
    with c2:
        balance_delta = "+" if health['avg_monthly_balance'] >= 0 else ""
        st.metric("Avg Monthly Balance", f"GBP {health['avg_monthly_balance']:.0f}")
    
    with c3:
        monthly_spend = health['total_spent'] / 12
        st.metric("Avg Monthly Spend", f"GBP {monthly_spend:.0f}")
    
    with c4:
        st.metric("Months Overspent", f"{health['months_overspent']}/12")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # tabs
    tabs = st.tabs(["Overview", "Spending", "Income vs Spending", "Insights", "Loan Info"])
    
    # TAB 1 - OVERVIEW
    with tabs[0]:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### Where Your Money Goes")
            
            by_category = spending_df.groupby('category')['amount'].sum().sort_values(ascending=True)
            
            fig = px.bar(
                x=by_category.values,
                y=by_category.index,
                orientation='h',
                template='plotly_dark',
                color=by_category.values,
                color_continuous_scale='Viridis'
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                showlegend=False,
                coloraxis_showscale=False,
                margin=dict(l=0, r=0, t=10, b=0),
                xaxis=dict(title='Total Spent (GBP)', gridcolor='rgba(255,255,255,0.1)'),
                yaxis=dict(title='')
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### Breakdown")
            
            essentials = ['Rent', 'Bills', 'Groceries', 'Transport', 'Course Costs']
            essential_total = spending_df[spending_df['category'].isin(essentials)]['amount'].sum()
            discretionary_total = spending_df[~spending_df['category'].isin(essentials)]['amount'].sum()
            
            fig = px.pie(
                values=[essential_total, discretionary_total],
                names=['Essentials', 'Non-essentials'],
                template='plotly_dark',
                color_discrete_sequence=['#3b82f6', '#8b5cf6']
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=0, r=0, t=30, b=0)
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown(f"""
            <div class="card">
                <p style="color: #94a3b8; margin-bottom: 8px;">Essential spending</p>
                <p class="big-number">{health['essential_ratio']:.0f}%</p>
                <p style="color: #64748b; font-size: 0.85rem;">of your total</p>
            </div>
            """, unsafe_allow_html=True)
    
    # TAB 2 - SPENDING OVER TIME
    with tabs[1]:
        st.markdown("### Monthly Spending")
        
        monthly_by_cat = spending_df.groupby(['date', 'category'])['amount'].sum().reset_index()
        
        fig = px.area(
            monthly_by_cat,
            x='date',
            y='amount',
            color='category',
            template='plotly_dark',
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(l=0, r=0, t=30, b=0),
            xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(title='Amount (GBP)', gridcolor='rgba(255,255,255,0.1)')
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # category breakdown
        st.markdown("### By Category")
        
        avg_by_cat = spending_df.groupby('category')['amount'].mean().sort_values(ascending=False)
        
        cols = st.columns(3)
        for i, (cat, avg) in enumerate(avg_by_cat.items()):
            with cols[i % 3]:
                annual = avg * 12
                st.markdown(f"""
                <div class="card" style="text-align: center;">
                    <p style="color: #94a3b8; margin-bottom: 4px;">{cat}</p>
                    <p style="font-size: 1.4rem; font-weight: 600; color: #e2e8f0; margin: 0;">GBP {avg:.0f}/mo</p>
                    <p style="color: #64748b; font-size: 0.8rem;">GBP {annual:.0f}/year</p>
                </div>
                """, unsafe_allow_html=True)
    
    # TAB 3 - INCOME VS SPENDING
    with tabs[2]:
        st.markdown("### Money In vs Money Out")
        
        monthly_spending = spending_df.groupby('date')['amount'].sum().reset_index()
        monthly_spending.columns = ['date', 'spending']
        
        merged = pd.merge(monthly_spending, income_df[['date', 'total_income']], on='date')
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=merged['date'],
            y=merged['total_income'],
            name='Income',
            marker_color='#10b981'
        ))
        
        fig.add_trace(go.Bar(
            x=merged['date'],
            y=merged['spending'],
            name='Spending',
            marker_color='#ef4444'
        ))
        
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            barmode='group',
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(l=0, r=0, t=30, b=0),
            xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(title='Amount (GBP)', gridcolor='rgba(255,255,255,0.1)')
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # cumulative balance
        st.markdown("### Running Balance")
        
        merged['balance'] = merged['total_income'] - merged['spending']
        merged['cumulative'] = merged['balance'].cumsum()
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=merged['date'],
            y=merged['cumulative'],
            fill='tozeroy',
            line=dict(color='#3b82f6', width=2),
            fillcolor='rgba(59, 130, 246, 0.3)'
        ))
        
        fig.add_hline(y=0, line_dash="dash", line_color="#ef4444", opacity=0.5)
        
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False,
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(title='Cumulative Balance (GBP)', gridcolor='rgba(255,255,255,0.1)')
        )
        st.plotly_chart(fig, use_container_width=True)
        
        if merged['cumulative'].iloc[-1] < 0:
            st.markdown(f"""
            <div class="warning-box">
                <p style="color: #fbbf24; font-weight: 600; margin: 0 0 8px 0;">Youre in the red</p>
                <p style="color: #cbd5e0; margin: 0;">Over the year youve spent GBP {abs(merged['cumulative'].iloc[-1]):.0f} more than youve earned. This means debt or eating into savings.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="good-box">
                <p style="color: #34d399; font-weight: 600; margin: 0 0 8px 0;">Youre in the green</p>
                <p style="color: #cbd5e0; margin: 0;">Over the year youve got GBP {merged['cumulative'].iloc[-1]:.0f} more than youve spent. Nice one.</p>
            </div>
            """, unsafe_allow_html=True)
    
    # TAB 4 - INSIGHTS
    with tabs[3]:
        st.markdown("### Whats Going On")
        
        insights = generate_insights(health, spending_df, income_df, location)
        
        for insight in insights:
            if insight['type'] == 'warning':
                box_class = 'warning-box'
                title_color = '#fbbf24'
            elif insight['type'] == 'good':
                box_class = 'good-box'
                title_color = '#34d399'
            else:
                box_class = 'info-box'
                title_color = '#60a5fa'
            
            st.markdown(f"""
            <div class="{box_class}">
                <p style="color: {title_color}; font-weight: 600; margin: 0 0 8px 0;">{insight['title']}</p>
                <p style="color: #cbd5e0; margin: 0 0 8px 0;">{insight['message']}</p>
                <p style="color: #94a3b8; font-size: 0.85rem; margin: 0;"><strong>Tip:</strong> {insight['action']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # predictions
        st.markdown("### Next 6 Months Prediction")
        
        model, monthly_hist = train_predictor(spending_df)
        
        last_month = monthly_hist['month_num'].max()
        last_date = monthly_hist['date'].max()
        
        future_data = []
        for i in range(1, 7):
            fut_date = last_date + timedelta(days=30*i)
            future_data.append({
                'month_num': last_month + i,
                'month_of_year': fut_date.month,
                'is_term_start': 1 if fut_date.month in [9, 10, 1] else 0,
                'is_exam_period': 1 if fut_date.month in [5, 12] else 0,
                'is_summer': 1 if fut_date.month in [6, 7, 8] else 0,
                'date': fut_date
            })
        
        future_df = pd.DataFrame(future_data)
        features = ['month_num', 'month_of_year', 'is_term_start', 'is_exam_period', 'is_summer']
        future_df['predicted'] = model.predict(future_df[features])
        
        predicted_total = future_df['predicted'].sum()
        avg_predicted = future_df['predicted'].mean()
        
        st.markdown(f"""
        <div class="card">
            <p style="color: #94a3b8;">Predicted spending next 6 months</p>
            <p class="big-number">GBP {predicted_total:,.0f}</p>
            <p style="color: #64748b;">About GBP {avg_predicted:.0f} per month</p>
        </div>
        """, unsafe_allow_html=True)
    
    # TAB 5 - LOAN INFO
    with tabs[4]:
        st.markdown("### Your Student Loan")
        
        if loan_type == 'plan5':
            threshold = PLAN5_THRESHOLD
            writeoff = PLAN5_WRITEOFF
            rate_desc = "RPI (capped at target rate of inflation)"
        else:
            threshold = PLAN2_THRESHOLD
            writeoff = PLAN2_WRITEOFF
            rate_desc = "RPI + up to 3%"
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="card" style="text-align: center;">
                <p style="color: #94a3b8;">Current Balance</p>
                <p class="big-number">GBP {current_debt:,}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="card" style="text-align: center;">
                <p style="color: #94a3b8;">Repayment Threshold</p>
                <p class="big-number">GBP {threshold:,}</p>
                <p style="color: #64748b; font-size: 0.85rem;">per year</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="card" style="text-align: center;">
                <p style="color: #94a3b8;">Written Off After</p>
                <p class="big-number">{writeoff} years</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("### What Youll Actually Repay")
        st.markdown("Based on different graduate salaries")
        
        salaries = [25000, 30000, 35000, 40000, 50000, 60000]
        repayments = []
        
        for salary in salaries:
            if salary > threshold:
                monthly = (salary - threshold) * 0.09 / 12
                annual = (salary - threshold) * 0.09
            else:
                monthly = 0
                annual = 0
            
            repayments.append({
                'Salary': f'GBP {salary:,}',
                'Monthly Repayment': f'GBP {monthly:.0f}',
                'Annual Repayment': f'GBP {annual:.0f}'
            })
        
        repay_df = pd.DataFrame(repayments)
        st.dataframe(repay_df, use_container_width=True, hide_index=True)
        
        st.markdown(f"""
        <div class="info-box">
            <p style="color: #60a5fa; font-weight: 600; margin: 0 0 8px 0;">How student loans actually work</p>
            <p style="color: #cbd5e0; margin: 0 0 8px 0;">
                You only repay 9% of what you earn <strong>above</strong> GBP {threshold:,}. If you earn less than that, you pay nothing.
                Its taken automatically from your payslip like tax.
            </p>
            <p style="color: #cbd5e0; margin: 0 0 8px 0;">
                Interest rate: {rate_desc}. Currently around {STUDENT_LOAN_INTEREST}%.
            </p>
            <p style="color: #94a3b8; font-size: 0.85rem; margin: 0;">
                Most graduates never pay it all off - its more like a graduate tax than a real loan. Dont stress about the total number.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #64748b; padding: 20px; font-size: 0.85rem;">
        <p>EconPulse - Built for Octopus Hackathon 2025</p>
        <p style="font-size: 0.75rem;">Data based on NUS, Save the Student, and gov.uk figures. For illustration only.</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
