"""
econpulse - student money tracker
for octopus hackathon 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
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


# maintenance loan rates 2024/25 from gov.uk
MAINTENANCE_LOANS = {
    'home': 8107,
    'away': 9978,
    'london': 13022
}

# plan 5 (started sept 2023+)
PLAN5_THRESHOLD = 25000
PLAN5_WRITEOFF = 40

# plan 2 (2012-2023)  
PLAN2_THRESHOLD = 27295
PLAN2_WRITEOFF = 30

# current rates dec 2024
UK_INFLATION = 2.6
STUDENT_LOAN_INTEREST = 7.3


def generate_data(months, location, has_job, monthly_wage, parental_support):
    """generate both spending and income with matching dates"""
    
    np.random.seed(42)
    
    # base monthly costs by location
    if location == 'london':
        costs = {
            'Rent': 750, 'Bills': 45, 'Groceries': 160, 'Transport': 90,
            'Phone': 18, 'Going Out': 120, 'Takeaways': 55,
            'Subscriptions': 25, 'Clothes': 35, 'Course Costs': 25, 'Other': 40
        }
    elif location == 'away':
        costs = {
            'Rent': 520, 'Bills': 55, 'Groceries': 140, 'Transport': 35,
            'Phone': 18, 'Going Out': 90, 'Takeaways': 45,
            'Subscriptions': 25, 'Clothes': 30, 'Course Costs': 20, 'Other': 35
        }
    else:  # home
        costs = {
            'Rent': 0, 'Bills': 50, 'Groceries': 80, 'Transport': 70,
            'Phone': 18, 'Going Out': 80, 'Takeaways': 40,
            'Subscriptions': 25, 'Clothes': 30, 'Course Costs': 20, 'Other': 30
        }
    
    # generate month list
    today = datetime.now()
    month_list = []
    for i in range(months - 1, -1, -1):
        d = today - timedelta(days=30 * i)
        month_list.append(datetime(d.year, d.month, 1))
    
    spending_data = []
    income_data = []
    annual_loan = MAINTENANCE_LOANS[location]
    
    for date in month_list:
        month = date.month
        
        # seasonal flags
        is_freshers = month in [9, 10]
        is_christmas = month == 12
        is_january = month == 1
        is_exam = month in [5, 12, 1]
        is_summer = month in [6, 7, 8]
        
        # generate spending for each category
        for cat, base in costs.items():
            amount = base
            
            # adjustments
            if cat == 'Going Out':
                if is_freshers: amount *= 1.5
                elif is_january: amount *= 0.5
                elif is_exam: amount *= 0.6
            elif cat == 'Takeaways':
                if is_exam: amount *= 1.4
            elif cat == 'Course Costs':
                if month in [9, 10, 1, 2]: amount *= 2.5
            elif cat == 'Transport':
                if is_christmas: amount *= 1.8
            elif cat == 'Bills':
                if month in [11, 12, 1, 2]: amount *= 1.3
            elif cat == 'Rent':
                if is_summer and location != 'home': amount *= 0.6
            
            # add some randomness
            amount = amount * (1 + np.random.uniform(-0.15, 0.15))
            amount = max(0, amount)
            
            spending_data.append({
                'date': date,
                'category': cat,
                'amount': round(amount, 2)
            })
        
        # generate income
        income = 0
        
        # loan installments
        if month == 9:
            income += annual_loan * 0.45
        elif month == 1:
            income += annual_loan * 0.275
        elif month == 4:
            income += annual_loan * 0.275
        
        # job income
        if has_job:
            wage = monthly_wage
            if is_summer: wage *= 1.5
            elif is_exam: wage *= 0.6
            wage = wage * (1 + np.random.uniform(-0.1, 0.1))
            income += max(0, wage)
        
        # family support
        if parental_support > 0:
            support = parental_support * (1 + np.random.uniform(-0.1, 0.1))
            income += max(0, support)
        
        income_data.append({
            'date': date,
            'income': round(income, 2)
        })
    
    spending_df = pd.DataFrame(spending_data)
    income_df = pd.DataFrame(income_data)
    
    return spending_df, income_df


def calculate_health(spending_df, income_df):
    """work out financial health score"""
    
    monthly_spend = spending_df.groupby('date')['amount'].sum().reset_index()
    monthly_spend.columns = ['date', 'spending']
    
    merged = pd.merge(monthly_spend, income_df, on='date')
    merged['balance'] = merged['income'] - merged['spending']
    merged['cumulative'] = merged['balance'].cumsum()
    
    if len(merged) == 0:
        return {
            'score': 50,
            'avg_balance': 0,
            'months_overspent': 0,
            'total_spent': spending_df['amount'].sum(),
            'essential_pct': 50,
            'merged': merged
        }
    
    avg_balance = merged['balance'].mean()
    months_overspent = (merged['balance'] < 0).sum()
    overspend_rate = months_overspent / len(merged) * 100
    
    # essential ratio
    essentials = ['Rent', 'Bills', 'Groceries', 'Transport', 'Course Costs', 'Phone']
    essential_spend = spending_df[spending_df['category'].isin(essentials)]['amount'].sum()
    total_spend = spending_df['amount'].sum()
    essential_pct = essential_spend / total_spend * 100 if total_spend > 0 else 0
    
    # calculate score
    score = 50
    
    if avg_balance > 100:
        score += 20
    elif avg_balance > 0:
        score += 10
    elif avg_balance > -50:
        score -= 5
    else:
        score -= 15
    
    if overspend_rate < 20:
        score += 15
    elif overspend_rate < 40:
        score += 5
    elif overspend_rate > 60:
        score -= 15
    
    if 55 <= essential_pct <= 80:
        score += 15
    
    score = max(0, min(100, score))
    
    return {
        'score': round(score),
        'avg_balance': round(avg_balance, 2),
        'months_overspent': int(months_overspent),
        'total_spent': round(total_spend, 2),
        'essential_pct': round(essential_pct, 1),
        'merged': merged
    }


def get_insights(health, spending_df, location):
    """generate useful insights"""
    insights = []
    
    if health['months_overspent'] >= 6:
        insights.append({
            'type': 'warning',
            'title': 'Spending more than you earn most months',
            'msg': f"Youve overspent in {health['months_overspent']} of the last 12 months. Time to look at where you can cut back."
        })
    elif health['months_overspent'] >= 3:
        insights.append({
            'type': 'warning',
            'title': 'Some months are tight',
            'msg': f"You overspent in {health['months_overspent']} months - usually around freshers or christmas. Try to save extra in quieter months."
        })
    
    # discretionary spending
    discretionary = ['Going Out', 'Takeaways', 'Subscriptions', 'Clothes']
    disc_spend = spending_df[spending_df['category'].isin(discretionary)]['amount'].sum()
    monthly_disc = disc_spend / 12
    
    if monthly_disc > 200:
        insights.append({
            'type': 'info',
            'title': f'About {monthly_disc:.0f} quid a month on non-essentials',
            'msg': 'Thats going out, takeaways, subscriptions and clothes. Not saying stop having fun but worth tracking.'
        })
    
    # loan vs spending
    annual_loan = MAINTENANCE_LOANS[location]
    if health['total_spent'] > annual_loan * 1.2:
        shortfall = health['total_spent'] - annual_loan
        insights.append({
            'type': 'warning',
            'title': 'Maintenance loan doesnt cover your costs',
            'msg': f"Youre spending about {shortfall:.0f} more than your loan. You need a job or family help to cover the gap."
        })
    
    if health['score'] >= 65:
        insights.append({
            'type': 'good',
            'title': 'Youre doing alright',
            'msg': 'Spending is under control. Keep it up.'
        })
    
    # always add loan info
    insights.append({
        'type': 'info',
        'title': 'About student loans',
        'msg': 'Plan 5 (started 2023+): repay when earning over 25k, 9% of everything above that. Written off after 40 years. Its more like a tax than a real loan.'
    })
    
    return insights


def main():
    # sidebar
    with st.sidebar:
        st.markdown("### Your Situation")
        
        location = st.selectbox(
            "Where do you live",
            options=['away', 'london', 'home'],
            format_func=lambda x: {'away': 'Away from home (not London)', 'london': 'London', 'home': 'With parents'}[x]
        )
        
        has_job = st.checkbox("I have a part-time job", value=True)
        
        if has_job:
            monthly_wage = st.slider("Monthly earnings (GBP)", 100, 1000, 400, 50)
        else:
            monthly_wage = 0
        
        parental_support = st.slider("Family support per month (GBP)", 0, 500, 100, 25)
        
        st.markdown("---")
        
        loan_type = st.radio(
            "Loan plan",
            options=['plan5', 'plan2'],
            format_func=lambda x: {'plan5': 'Plan 5 (started 2023+)', 'plan2': 'Plan 2 (2012-2023)'}[x]
        )
        
        current_debt = st.number_input("Current loan balance (GBP)", 0, 100000, 45000, 1000)
        
        st.markdown("---")
        st.markdown("<p style='text-align:center; color:#64748b; font-size:0.75rem;'>Octopus Hackathon 2025</p>", unsafe_allow_html=True)
    
    # generate data
    spending_df, income_df = generate_data(12, location, has_job, monthly_wage, parental_support)
    health = calculate_health(spending_df, income_df)
    merged = health['merged']
    
    # header
    st.markdown("""
    <div style="text-align: center; padding: 20px 0;">
        <h1 style="font-size: 2.5rem; margin-bottom: 5px;">EconPulse</h1>
        <p style="color: #94a3b8;">Student Money Tracker</p>
    </div>
    """, unsafe_allow_html=True)
    
    # top metrics
    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        st.metric("Health Score", f"{health['score']}/100")
    with c2:
        st.metric("Avg Monthly Balance", f"GBP {health['avg_balance']:.0f}")
    with c3:
        monthly_spend = health['total_spent'] / 12
        st.metric("Avg Monthly Spend", f"GBP {monthly_spend:.0f}")
    with c4:
        st.metric("Months Overspent", f"{health['months_overspent']}/12")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # tabs
    tabs = st.tabs(["Overview", "Spending", "Cash Flow", "Insights", "Loan Info"])
    
    # TAB 1 - OVERVIEW
    with tabs[0]:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### Where Your Money Goes")
            
            by_cat = spending_df.groupby('category')['amount'].sum().sort_values(ascending=True)
            
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
                xaxis=dict(title='Total Spent (GBP)', gridcolor='rgba(255,255,255,0.1)'),
                yaxis=dict(title='')
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### Split")
            
            essentials = ['Rent', 'Bills', 'Groceries', 'Transport', 'Course Costs', 'Phone']
            ess_total = spending_df[spending_df['category'].isin(essentials)]['amount'].sum()
            disc_total = spending_df[~spending_df['category'].isin(essentials)]['amount'].sum()
            
            fig = px.pie(
                values=[ess_total, disc_total],
                names=['Essentials', 'Fun stuff'],
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
                <p style="color: #94a3b8; margin-bottom: 8px;">Essentials</p>
                <p class="big-number">{health['essential_pct']:.0f}%</p>
                <p style="color: #64748b; font-size: 0.85rem;">of total spending</p>
            </div>
            """, unsafe_allow_html=True)
    
    # TAB 2 - SPENDING
    with tabs[1]:
        st.markdown("### Monthly Spending")
        
        monthly_cat = spending_df.groupby(['date', 'category'])['amount'].sum().reset_index()
        
        fig = px.area(
            monthly_cat,
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
        
        st.markdown("### Category Averages")
        
        avg_cat = spending_df.groupby('category')['amount'].mean().sort_values(ascending=False)
        
        cols = st.columns(3)
        for i, (cat, avg) in enumerate(avg_cat.items()):
            with cols[i % 3]:
                st.markdown(f"""
                <div class="card" style="text-align: center;">
                    <p style="color: #94a3b8; margin-bottom: 4px;">{cat}</p>
                    <p style="font-size: 1.3rem; font-weight: 600; color: #e2e8f0; margin: 0;">GBP {avg:.0f}/mo</p>
                </div>
                """, unsafe_allow_html=True)
    
    # TAB 3 - CASH FLOW
    with tabs[2]:
        st.markdown("### Money In vs Out")
        
        if len(merged) > 0:
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=merged['date'],
                y=merged['income'],
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
            
            st.markdown("### Running Balance")
            
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
                yaxis=dict(title='Cumulative (GBP)', gridcolor='rgba(255,255,255,0.1)')
            )
            st.plotly_chart(fig, use_container_width=True)
            
            final_balance = merged['cumulative'].iloc[-1]
            if final_balance < 0:
                st.markdown(f"""
                <div class="warning-box">
                    <p style="color: #fbbf24; font-weight: 600; margin: 0 0 8px 0;">Youre in the red</p>
                    <p style="color: #cbd5e0; margin: 0;">Over the year youve spent GBP {abs(final_balance):.0f} more than came in.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="good-box">
                    <p style="color: #34d399; font-weight: 600; margin: 0 0 8px 0;">Youre in the green</p>
                    <p style="color: #cbd5e0; margin: 0;">Over the year youve saved GBP {final_balance:.0f}. Nice one.</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("No data to display")
    
    # TAB 4 - INSIGHTS
    with tabs[3]:
        st.markdown("### Whats Going On")
        
        insights = get_insights(health, spending_df, location)
        
        for ins in insights:
            if ins['type'] == 'warning':
                box = 'warning-box'
                col = '#fbbf24'
            elif ins['type'] == 'good':
                box = 'good-box'
                col = '#34d399'
            else:
                box = 'info-box'
                col = '#60a5fa'
            
            st.markdown(f"""
            <div class="{box}">
                <p style="color: {col}; font-weight: 600; margin: 0 0 8px 0;">{ins['title']}</p>
                <p style="color: #cbd5e0; margin: 0;">{ins['msg']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # TAB 5 - LOAN INFO
    with tabs[4]:
        st.markdown("### Student Loan")
        
        if loan_type == 'plan5':
            threshold = PLAN5_THRESHOLD
            writeoff = PLAN5_WRITEOFF
        else:
            threshold = PLAN2_THRESHOLD
            writeoff = PLAN2_WRITEOFF
        
        c1, c2, c3 = st.columns(3)
        
        with c1:
            st.markdown(f"""
            <div class="card" style="text-align: center;">
                <p style="color: #94a3b8;">Balance</p>
                <p class="big-number">GBP {current_debt:,}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with c2:
            st.markdown(f"""
            <div class="card" style="text-align: center;">
                <p style="color: #94a3b8;">Repay Threshold</p>
                <p class="big-number">GBP {threshold:,}</p>
                <p style="color: #64748b; font-size: 0.8rem;">per year</p>
            </div>
            """, unsafe_allow_html=True)
        
        with c3:
            st.markdown(f"""
            <div class="card" style="text-align: center;">
                <p style="color: #94a3b8;">Written Off</p>
                <p class="big-number">{writeoff} yrs</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("### Monthly Repayments by Salary")
        
        salaries = [25000, 30000, 35000, 40000, 50000, 60000]
        repay_data = []
        
        for sal in salaries:
            if sal > threshold:
                monthly = (sal - threshold) * 0.09 / 12
            else:
                monthly = 0
            
            repay_data.append({
                'Salary': f'GBP {sal:,}',
                'Monthly Payment': f'GBP {monthly:.0f}'
            })
        
        st.dataframe(pd.DataFrame(repay_data), use_container_width=True, hide_index=True)
        
        st.markdown(f"""
        <div class="info-box">
            <p style="color: #60a5fa; font-weight: 600; margin: 0 0 8px 0;">How it works</p>
            <p style="color: #cbd5e0; margin: 0;">
                You pay 9% of everything you earn above GBP {threshold:,}. If you earn less, you pay nothing.
                Its taken from your payslip automatically. Most people never pay it all off - think of it more like a graduate tax.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # footer
    st.markdown("---")
    st.markdown("""
    <p style="text-align: center; color: #64748b; font-size: 0.8rem;">
        EconPulse - Octopus Hackathon 2025<br>
        Data based on NUS and Save the Student surveys
    </p>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
