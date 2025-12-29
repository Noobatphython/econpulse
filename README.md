# EconPulse

Student Financial Wellness Platform - built for Octopus Hackathon 2025

## The Problem

72% of UK students experience financial stress (NUS Cost of Living Survey 2024). Most budgeting apps treat students like everyone else, ignoring the unique challenges we face: irregular income from part-time work, student loans hanging over us, semester-based expenses, and lets be honest - not much financial literacy coming out of school.

## What It Does

EconPulse helps students understand their spending in the context of real economic conditions. It uses ML to spot patterns, predict future spending, and give recommendations that actually make sense for students.

## Features

- **Dashboard** - spending breakdown by category with a health score
- **AI Insights** - pattern analysis using K-means clustering, trend detection
- **Economic Impact** - shows how inflation and interest rates affect your money and student loans
- **Budget Planner** - 50/30/20 rule with category tracking
- **Predictions** - 6-month forecasts using Random Forest

## Tech

- Python 3.9+
- Streamlit
- Pandas / NumPy
- Scikit-learn
- Plotly

## Data Sources

- ONS Consumer Price Inflation (ons.gov.uk)
- Bank of England base rate (bankofengland.co.uk)
- Student Loans Company thresholds (gov.uk)
- NUS student spending survey

## Running It

```bash
git clone https://github.com/yourusername/econpulse.git
cd econpulse
pip install -r requirements.txt
streamlit run app.py
```

## Structure

```
econpulse/
  app.py              # main app
  requirements.txt    
  README.md          
  DEPLOYMENT.md      
```

## How It Works

1. Spending data based on NUS survey averages with realistic variance
2. Health score combines savings rate, stability, and needs/wants balance
3. K-means clustering finds spending patterns
4. Random Forest predicts future months
5. Economic data shows inflation impact on purchasing power

## Whats Next

- Open Banking integration for real data
- Push notifications
- Benchmarking against other students
- Mobile apps

## Licence

MIT
