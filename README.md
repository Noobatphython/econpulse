# EconPulse

money tracker for uk students - octopus hackathon 2025

## what it does

tracks spending and income for uk students based on actual costs and maintenance loan amounts. shows you where your money goes and whether youre overspending.

## features

- spending breakdown by category
- income vs spending comparison (including loan installments)
- financial health score based on real metrics
- student loan repayment info (plan 2 and plan 5)
- 6 month spending predictions

## data sources

spending figures based on:
- save the student annual survey
- nus cost of living report
- unipol student homes survey

loan info from gov.uk

## running it

```
pip install -r requirements.txt
streamlit run app.py
```

## tech

- streamlit
- pandas
- plotly
- scikit-learn
