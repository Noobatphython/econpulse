# EconPulse Deployment Guide

## Option 1: Streamlit Cloud (Recommended)

Easiest option - free hosting, no server setup needed.

### Steps

1. Push your code to GitHub
```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/yourusername/econpulse.git
git push -u origin main
```

2. Go to share.streamlit.io and sign in with GitHub

3. Click "New app" and select:
   - Repository: your repo
   - Branch: main
   - Main file path: app.py

4. Click Deploy

Your app will be live at: https://yourusername-econpulse-app-xxxxx.streamlit.app

### Troubleshooting

- If deployment fails, check the logs for missing dependencies
- Make sure requirements.txt includes all packages
- The free tier has memory limits - if you hit them, reduce data size


## Option 2: Google Colab

Good for quick demos without permanent hosting.

### Steps

1. Open Google Colab (colab.research.google.com)

2. Create a new notebook

3. Run these cells:

```python
# Cell 1: Install dependencies
!pip install -q streamlit pandas numpy plotly scikit-learn pyngrok
```

```python
# Cell 2: Create the app file
%%writefile app.py
# [paste entire app.py content here]
```

```python
# Cell 3: Run with ngrok tunnel
from pyngrok import ngrok
import subprocess
import time

# Get free ngrok token from ngrok.com/signup
!ngrok authtoken YOUR_TOKEN_HERE

# Start streamlit
process = subprocess.Popen(['streamlit', 'run', 'app.py', '--server.port=8501'])
time.sleep(5)

# Create public URL
public_url = ngrok.connect(8501)
print(f"App live at: {public_url}")
```

Note: Keep the notebook running to keep your app live.


## Option 3: Local Development

For testing and development.

### Steps

1. Clone the repo
```bash
git clone https://github.com/yourusername/econpulse.git
cd econpulse
```

2. Create virtual environment (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Run the app
```bash
streamlit run app.py
```

5. Open http://localhost:8501 in your browser


## Screenshots for Submission

Take screenshots of:

1. Dashboard tab - showing the spending chart and health score
2. AI Insights tab - showing the spending pattern analysis
3. Economic Impact tab - showing the indicators chart and loan calculator
4. Budget Planner tab - showing the needs/wants/savings comparison
5. Predictions tab - showing the forecast chart and scenarios

Tips:
- Use a clean browser window (no bookmarks bar)
- Make sure all charts are fully loaded
- Consider using the full-screen mode for cleaner screenshots


## Environment Variables

No environment variables required for the demo.

For production with real data, you would need:
- `PLAID_CLIENT_ID` - for Open Banking integration
- `PLAID_SECRET` - Plaid API secret
- `DATABASE_URL` - if using persistent storage
