# Moneylytics
AI-powered personal financial analyst built with Streamlit
# 💰 Moneylytics – AI Personal Financial Analyst

> **Your all-in-one financial intelligence dashboard.**  
> Upload your Excel transactions, visualize spending, track savings, detect anomalies, and get AI-powered insights — all in seconds.

---

## 🚀 Live App

👉 **Try it now:** [https://moneylytics.streamlit.app](#)  
*(Replace this with your actual Streamlit Cloud URL after publishing)*

---

## 🧠 Overview

**Moneylytics** transforms raw financial data into **interactive dashboards** and **AI-driven recommendations**.

- 📊 Interactive insights on income, expenses, and net balance  
- 💡 AI-generated advice (powered by Gemini API)  
- 🎯 Category-level budgeting with live progress  
- 🔁 Detection of recurring subscriptions  
- ⚠️ Smart anomaly & overspend alerts  
- 💹 Savings rate and trend forecasts  
- 📤 One-click export to Excel, PDF, or ZIP bundles  
- 🌗 Automatic dark/light mode sync with your system  

---

## 🧾 Features Breakdown

### 1️⃣ Dashboard
- KPIs: Total income, expenses, net balance, savings rate  
- Best / worst / average month tracking  
- Expense trend & forecast (linear regression)  
- Income vs Expenses, Savings Rate, Monthly Comparison charts  
- Cashflow heatmap by day & month  

### 2️⃣ Category Budget Tracker
- Set and visualize monthly budgets per expense category  
- Track progress with color-coded bars (green = within budget, red = overspend)  
- Compare “Spent vs Budget” across categories  

### 3️⃣ Financial Health
- Compute **Financial Health Index (0–100)**  
- Expense-to-Income ratio tracking  
- Category correlation matrix  

### 4️⃣ Smart Alerts & Insights
- Overspending & anomaly detection  
- Recurring subscription identification  
- AI Budget Coach for personalized guidance  

### 5️⃣ Exports
- 🧾 Monthly Summary (Excel)  
- 📘 Generative Executive Summary (PDF)  
- 📈 Visual Insights Report (Charts as PDF)  
- 📦 Complete ZIP Export (all reports + CSV summaries)

---

## 📸 Screenshots

| Dashboard | AI Insights | Category Budgets |
|------------|-------------|------------------|
| ![Dashboard](assets/dashboard.png) | ![AI](assets/ai_insights.png) | ![Budgets](assets/budget_tracker.png) |

*(Replace with your actual screenshots once deployed.)*

---

## ⚙️ Installation (Local)

```bash
# Clone repository
git clone https://github.com/yourusername/moneylytics.git
cd moneylytics

# Create virtual environment (optional)
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Create .env file with your Gemini API key
echo 'GEMINI_API_KEY=your_google_genai_key_here' > .env

# Run the app
streamlit run app.py
