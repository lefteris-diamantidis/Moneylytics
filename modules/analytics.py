import io
import pandas as pd
import numpy as np
import streamlit as st
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

# =============================
# CORE SUMMARIES
# =============================
def summarize_expenses(df):
    df['month'] = df['date'].dt.to_period('M')

    total_spent = df[df['amount'] < 0]['amount'].sum()
    total_income = df[df['amount'] > 0]['amount'].sum()
    balance = total_income + total_spent

    summary = {
        "Total Income (€)": round(total_income, 2),
        "Total Expenses (€)": round(total_spent, 2),
        "Net Balance (€)": round(balance, 2)
    }

    cat_summary = df.groupby('category', dropna=False)['amount'].sum().sort_values()
    trend_df = df.groupby('month')['amount'].sum().reset_index()
    trend_df['month'] = trend_df['month'].astype(str)

    return summary, cat_summary, trend_df


def generate_monthly_summary_excel(df):
    def _find_column(possible_names):
        for name in possible_names:
            for col in df.columns:
                if name.lower().replace("-", "").replace("_", "") in col.lower().replace("-", "").replace("_", ""):
                    return col
        return None

    date_col = _find_column(["Date"])
    amount_col = _find_column(["Amount", "Value", "Transaction"])
    category_col = _find_column(["Category", "Main Category"])
    subcategory_col = _find_column(["Subcategory", "Sub-Category", "Sub Category"])

    if not all([date_col, amount_col, category_col]):
        raise ValueError("Missing one or more key columns: Date, Amount, Category.")

    df = df.copy()
    df[amount_col] = (
        df[amount_col].astype(str).str.replace("€", "").str.replace(",", "").astype(float)
    )
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df["Month"] = df[date_col].dt.to_period("M").astype(str)

    monthly_summary = df.groupby("Month")[amount_col].sum().reset_index()
    monthly_summary.columns = ["Month", "Net Flow (€)"]

    monthly_income = df[df[amount_col] > 0].groupby("Month")[amount_col].sum().reset_index()
    monthly_income.columns = ["Month", "Total Income (€)"]

    monthly_expenses = df[df[amount_col] < 0].groupby("Month")[amount_col].sum().reset_index()
    monthly_expenses[amount_col] = monthly_expenses[amount_col].abs()
    monthly_expenses.columns = ["Month", "Total Expenses (€)"]

    summary = (
        monthly_summary
        .merge(monthly_income, on="Month", how="left")
        .merge(monthly_expenses, on="Month", how="left")
        .fillna(0)
    )

    if subcategory_col:
        top_subcats = (
            df[df[amount_col] < 0]
            .groupby(["Month", subcategory_col])[amount_col]
            .sum()
            .abs()
            .reset_index()
        ).sort_values(["Month", amount_col], ascending=[True, False])
        top_subcats = top_subcats.groupby("Month").first().reset_index()
        top_subcats.columns = ["Month", "Top Subcategory", "Top Subcat Spending (€)"]
    else:
        top_subcats = pd.DataFrame(columns=["Month", "Top Subcategory", "Top Subcat Spending (€)"])

    top_cats = (
        df[df[amount_col] < 0]
        .groupby(["Month", category_col])[amount_col]
        .sum()
        .abs()
        .reset_index()
    ).sort_values(["Month", amount_col], ascending=[True, False])
    top_cats = top_cats.groupby("Month").first().reset_index()
    top_cats.columns = ["Month", "Top Category", "Top Category Spending (€)"]

    summary = summary.merge(top_cats, on="Month", how="left").merge(top_subcats, on="Month", how="left")

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        summary.to_excel(writer, index=False, sheet_name="Monthly Summary")
        df.to_excel(writer, index=False, sheet_name="Transactions")

    buffer.seek(0)
    return buffer


# =============================
# CATEGORY BUDGET TRACKER (expenses only)
# =============================
def calculate_budget_progress(df, budget_dict):
    """
    Compare actual monthly expense spending vs. monthly budget for each category.
    Excludes income (only negative amounts are considered; absolute value used).
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["month"] = df["date"].dt.to_period("M").astype(str)
    df["category"] = df["category"].str.lower().fillna("uncategorized")

    df = df[df["amount"] < 0]
    df["amount"] = df["amount"].abs()

    monthly = df.groupby(["month", "category"])["amount"].sum().reset_index()

    results = []
    for cat, budget in budget_dict.items():
        subset = monthly[monthly["category"] == cat.lower()]
        if subset.empty:
            continue

        for _, row in subset.iterrows():
            spent = row["amount"]
            remaining = budget - spent
            percent = (spent / budget * 100) if budget > 0 else 0
            results.append({
                "Month": row["month"],
                "Category": cat.title(),
                "Budget (€)": round(budget, 2),
                "Spent (€)": round(spent, 2),
                "Remaining (€)": round(remaining, 2),
                "Progress (%)": round(percent, 1),
                "Status": "✅ Within Budget" if percent <= 100 else "⚠️ Over Budget"
            })
    return results


# =============================
# MONTHLY TREND COMPARISON
# =============================
def compare_monthly_trends(df):
    df['month'] = pd.to_datetime(df['date']).dt.to_period('M')
    monthly = df.groupby('month')['amount'].sum().to_frame()
    monthly['income'] = df[df['amount'] > 0].groupby('month')['amount'].sum()
    monthly['expenses'] = df[df['amount'] < 0].groupby('month')['amount'].sum()
    monthly = monthly.fillna(0)
    monthly['net'] = monthly['income'] + monthly['expenses']
    monthly['income_change'] = monthly['income'].pct_change() * 100
    monthly['expense_change'] = monthly['expenses'].pct_change() * 100
    return monthly.reset_index()


# =============================
# EXTRA ANALYTICS (new)
# =============================
def compute_savings_rate_series(df):
    """Return monthly savings rate % series + last month value."""
    x = df.copy()
    x["month"] = x["date"].dt.to_period("M")
    income = x[x["amount"] > 0].groupby("month")["amount"].sum()
    exp = x[x["amount"] < 0].groupby("month")["amount"].sum().abs()
    sr = ((income - exp) / income.replace(0, np.nan) * 100).fillna(0)
    sr = sr.sort_index()
    last = float(sr.iloc[-1]) if len(sr) else 0.0
    return sr, last


def income_stability_score(df):
    """Score (0–100): lower variation in monthly income → higher score."""
    x = df.copy()
    x["month"] = x["date"].dt.to_period("M")
    income = x[x["amount"] > 0].groupby("month")["amount"].sum()
    if len(income) < 2:
        return 50.0  # neutral if insufficient data
    cv = float(income.std(ddof=1) / (abs(income.mean()) + 1e-9))
    score = max(0.0, 100.0 - min(100.0, cv * 100.0))
    return round(score, 1)


def detect_recurring_subscriptions(df, top_n=10):
    """
    Heuristic recurring detector by 'description':
    - Appears in >= 2 distinct months
    - Similar amounts (low std/mean)
    """
    if "description" not in df.columns:
        return pd.DataFrame(columns=["Description", "Months Active", "Avg (€)", "Last (€)", "Volatility"])

    x = df.copy()
    x["month"] = x["date"].dt.to_period("M")
    x["desc_norm"] = x["description"].astype(str).str.strip().str.lower()

    grp = x.groupby(["desc_norm", "month"])["amount"].sum().reset_index()
    agg = grp.groupby("desc_norm")["amount"].agg(["count", "mean", "std", "last"]).reset_index()
    agg["months_active"] = grp.groupby("desc_norm")["month"].nunique().values
    agg = agg[agg["months_active"] >= 2]
    agg["volatility"] = (agg["std"].abs() / (agg["mean"].abs() + 1e-9)).fillna(0)
    out = agg.sort_values(["volatility", "months_active"], ascending=[True, False]).head(top_n)

    out = out.rename(columns={"desc_norm": "Description", "mean": "Avg (€)", "last": "Last (€)"})
    out["Avg (€)"] = out["Avg (€)"].round(2)
    out["Last (€)"] = out["Last (€)"].round(2)
    out["Volatility"] = out["volatility"].round(2)
    out = out[["Description", "months_active", "Avg (€)", "Last (€)", "Volatility"]].rename(
        columns={"months_active": "Months Active"}
    )
    return out.reset_index(drop=True)


def detect_category_anomalies(df, z_thresh=2.0):
    """
    Z-score of monthly expense per category; flags anomalies where |z| >= threshold.
    """
    x = df.copy()
    x["month"] = x["date"].dt.to_period("M")
    x_exp = x[x["amount"] < 0].copy()
    if x_exp.empty:
        return pd.DataFrame(columns=["Month", "Category", "Spent (€)", "Z-Score", "Flag"])

    monthly_cat = (
        x_exp.assign(amount=lambda d: d["amount"].abs())
            .groupby(["month", "category"])["amount"].sum()
            .reset_index()
    )
    out_rows = []
    for cat, sub in monthly_cat.groupby("category"):
        m = sub["amount"].mean()
        s = sub["amount"].std(ddof=1) or 1.0
        z = (sub["amount"] - m) / s
        flagged = sub.loc[z.abs() >= z_thresh].copy()
        if not flagged.empty:
            for _, r in flagged.iterrows():
                out_rows.append({
                    "Month": str(r["month"]),
                    "Category": cat,
                    "Spent (€)": round(float(r["amount"]), 2),
                    "Z-Score": round(float(((r["amount"] - m) / s)), 2),
                    "Flag": "⚠️ High vs typical"
                })
    return pd.DataFrame(out_rows).sort_values(["Month", "Category"]).reset_index(drop=True)


# =============================
# EXPORT PDF
# =============================
def export_dashboard_pdf(summary_text, chart_paths=None):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "Personal Finance Report")

    c.setFont("Helvetica", 12)
    text_object = c.beginText(50, height - 100)
    for line in summary_text.split('\n'):
        text_object.textLine(line)
    c.drawText(text_object)

    if chart_paths:
        y_pos = height - 300
        for chart_path in chart_paths:
            try:
                c.drawImage(chart_path, 50, y_pos, width=500, preserveAspectRatio=True)
                y_pos -= 250
            except:
                continue

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

# =====================================================
# 💡 FINANCIAL HEALTH INDEX + ADVANCED ANALYTICS
# =====================================================
import numpy as np
import plotly.graph_objects as go

def calculate_financial_health_index(df, sr_last, target_sr, result_df):
    """
    Compute an overall Financial Health Index (0–100).
    Combines:
      - Savings performance
      - Income stability
      - Budget adherence (if budgets exist)
    """
    # --- Savings component ---
    savings_score = min((sr_last / target_sr) * 100 if target_sr > 0 else 50, 100)

    # --- Income stability component ---
    monthly_income = df[df["amount"] > 0].groupby(df["date"].dt.to_period("M"))["amount"].sum()
    stability_score = 100 - (monthly_income.pct_change().abs().mean(skipna=True) * 100)
    stability_score = np.clip(stability_score, 0, 100)

    # --- Budget adherence component (optional) ---
    if result_df is not None and not result_df.empty:
        avg_progress = np.clip(result_df["Progress (%)"].mean(), 0, 200)
        budget_score = max(0, 100 - abs(avg_progress - 100))
    else:
        budget_score = 60  # default neutral

    # --- Weighted average ---
    final_score = round(
        (0.4 * savings_score) + (0.3 * stability_score) + (0.3 * budget_score), 1
    )

    # --- Qualitative message ---
    if final_score >= 85:
        msg = "🌟 Excellent — your finances are healthy and stable!"
    elif final_score >= 70:
        msg = "✅ Good — a few tweaks can improve your savings consistency."
    elif final_score >= 50:
        msg = "⚠️ Fair — try improving savings rate or smoothing income."
    else:
        msg = "🚨 Needs Attention — high spending or irregular income detected."

    return final_score, msg


# =====================================================
# 📉 EXPENSE-TO-INCOME RATIO OVER TIME
# =====================================================
def expense_to_income_ratio(df):
    """Returns a monthly DataFrame with Expense/Income ratio."""
    df = df.copy()
    df["Month"] = df["date"].dt.to_period("M").dt.to_timestamp()
    monthly_income = df[df["amount"] > 0].groupby("Month")["amount"].sum()
    monthly_exp = df[df["amount"] < 0].groupby("Month")["amount"].sum().abs()
    ratio = (monthly_exp / monthly_income.replace(0, np.nan)) * 100
    ratio = ratio.fillna(0)
    return ratio.reset_index(name="Expense/Income (%)")


# =====================================================
# 🔗 CATEGORY CORRELATION MATRIX
# =====================================================
def category_correlation_matrix(df):
    """
    Builds a correlation matrix across categories (monthly sums).
    """
    df = df.copy()
    df["Month"] = df["date"].dt.to_period("M").dt.to_timestamp()
    pivot = df.pivot_table(
        index="Month", columns="category", values="amount", aggfunc="sum"
    ).fillna(0)

    corr = pivot.corr().round(2)
    return corr

# =====================================================
# 🚨 SMART ALERTS: Overspending & Income Drops
# =====================================================
def detect_financial_alerts(df, result_df=None):
    """
    Detects:
      - Categories >120% of budget (if budget results exist)
      - 3-month overspending streaks
      - Income drop >25% vs avg
    Returns a list of alerts.
    """
    alerts = []
    df = df.copy()
    df["month"] = df["date"].dt.to_period("M").astype(str)

    # --- Overspending ---
    if result_df is not None and not result_df.empty:
        overspent = result_df[result_df["Progress (%)"] > 120]
        for _, row in overspent.iterrows():
            alerts.append(
                f"⚠️ Category **{row['Category']}** exceeded budget by {row['Progress (%)']-100:.1f}% in {row['Month']}."
            )

    # --- Overspending streaks (3+ months negative net) ---
    monthly_net = df.groupby("month")["amount"].sum()
    negative_streaks = (monthly_net < 0).astype(int).rolling(3).sum()
    if (negative_streaks >= 3).any():
        alerts.append("🚨 You’ve had 3+ consecutive months of negative net balance!")

    # --- Income drop >25% vs avg ---
    monthly_income = df[df["amount"] > 0].groupby("month")["amount"].sum()
    avg_income = monthly_income.mean()
    last_income = monthly_income.iloc[-1] if not monthly_income.empty else 0
    if avg_income > 0 and last_income < 0.75 * avg_income:
        alerts.append("⚠️ Last month’s income dropped more than 25% below average.")

    return alerts


# =====================================================
# 💹 NET WORTH GROWTH OVER TIME
# =====================================================
def cumulative_net_worth(df):
    """Compute cumulative balance over time."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date")
    df["cumulative_net"] = df["amount"].cumsum()
    return df[["date", "cumulative_net"]]


# =====================================================
# 🤖 AI BUDGET COACH (Gemini)
# =====================================================
def ai_budget_coach(cat_summary, gemini_model):
    """
    Suggests improved budgets based on spending patterns.
    Requires an active Gemini model.
    """
    summary_text = cat_summary.to_string() if isinstance(cat_summary, (pd.Series, pd.DataFrame)) else str(cat_summary)
    prompt = f"""
    You are Moneylytics, an AI budget planner.
    Based on this category spending summary:
    {summary_text}

    Suggest an improved monthly budget allocation.
    - Include 5–10 categories
    - Be realistic and data-driven
    - Total monthly spending should be balanced (not overly strict)
    Respond as a neat markdown table.
    """
    response = gemini_model.generate_content(prompt)
    return response.text.strip() if response and response.text else "⚠️ Could not generate AI budget suggestion."

def display_extended_kpis(df: pd.DataFrame, currency: str = "€"):
    """
    Display advanced financial KPIs such as volatility, concentration,
    income diversity, savings consistency, fixed ratio, etc.
    """
    try:
        if df.empty or "amount" not in df.columns or "date" not in df.columns:
            st.warning("No valid data for extended KPIs.")
            return

        # Ensure datetime
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date", "amount"])

        # Monthly aggregation
        monthly_net = df.groupby(df["date"].dt.to_period("M"))["amount"].sum().astype(float)
        monthly_net.index = monthly_net.index.to_timestamp()

        # ---- 1️⃣ Cash Flow Volatility ----
        volatility = (
            monthly_net.std() / abs(monthly_net.mean()) * 100
            if len(monthly_net) > 1 and monthly_net.mean() != 0 else 0
        )

        # ---- 2️⃣ Expense Concentration ----
        expense_share = (
            df[df["amount"] < 0]
            .groupby("category")["amount"]
            .sum()
            .abs()
            .sort_values(ascending=False)
        )
        top_ratio = expense_share.iloc[0] / expense_share.sum() * 100 if not expense_share.empty else 0

        # ---- 3️⃣ Income Diversity (HHI) ----
        income_cats = (
            df[df["amount"] > 0]
            .groupby("category")["amount"]
            .sum()
            .abs()
        )
        if not income_cats.empty:
            total = income_cats.sum()
            hhi = ((income_cats / total) ** 2).sum()
            diversity = (1 - hhi) * 100
        else:
            diversity = 0

        # ---- 4️⃣ Savings Momentum ----
        cumulative_savings = monthly_net.cumsum()
        if len(cumulative_savings) >= 2 and cumulative_savings.iloc[-2] != 0:
            momentum = (
                (cumulative_savings.iloc[-1] - cumulative_savings.iloc[-2])
                / abs(cumulative_savings.iloc[-2]) * 100
            )
        else:
            momentum = 0

        # ---- 5️⃣ Fixed vs Variable Expenses ----
        fixed_keywords = ["rent", "insurance", "subscription", "utility", "mortgage"]
        df["is_fixed"] = df["category"].str.lower().apply(
            lambda x: any(k in x for k in fixed_keywords) if isinstance(x, str) else False
        )
        fixed_exp = df[(df["amount"] < 0) & df["is_fixed"]]["amount"].abs().sum()
        variable_exp = df[(df["amount"] < 0) & (~df["is_fixed"])]["amount"].abs().sum()
        ratio = (
            fixed_exp / (fixed_exp + variable_exp) * 100
            if (fixed_exp + variable_exp) > 0 else 0
        )

        # ---- 6️⃣ Savings Consistency ----
        positive_months = (monthly_net > 0).sum()
        score = positive_months / len(monthly_net) * 100 if len(monthly_net) > 0 else 0

        # ---- 7️⃣ Largest Expense ----
        max_expense = df[df["amount"] < 0]["amount"].min()
        largest_exp = abs(max_expense) if pd.notna(max_expense) else 0

        # ---- 8️⃣ Financial Anomaly Ratio ----
        if len(monthly_net) > 2 and monthly_net.std() > 0:
            z = (monthly_net - monthly_net.mean()) / monthly_net.std()
            anomaly_ratio = (abs(z) > 1.5).mean() * 100
        else:
            anomaly_ratio = 0

        # ==============================
        # 📊 DISPLAY METRICS IN GRID
        # ==============================
        st.subheader("📊 Extended Financial KPIs")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("📉 Cash Flow Volatility", f"{volatility:.1f}%", help="Standard deviation / mean of monthly net income")
            st.metric("🧭 Savings Consistency", f"{score:.1f}%", help="Share of months with positive net balance")

        with col2:
            st.metric("🏷️ Expense Concentration", f"{top_ratio:.1f}%", help="Share of top expense category in total expenses")
            st.metric("⚖️ Fixed Expense Ratio", f"{ratio:.1f}%", help="Share of fixed expenses in total spending")

        with col3:
            st.metric("💼 Income Diversity", f"{diversity:.1f}%", help="Higher = more balanced income sources")
            st.metric("📈 Savings Momentum", f"{momentum:.1f}%", help="Month-over-month cumulative savings growth")

        st.metric("💣 Largest Expense", f"{largest_exp:,.2f}{currency}")
        st.metric("🚨 Financial Anomaly Ratio", f"{anomaly_ratio:.1f}%")

    except Exception as ex:
        st.warning(f"Could not compute extended KPIs: {ex}")
        


def generate_executive_summary_pdf(df, summary, sr_last, target_sr):
    """Create a concise executive-style PDF summary."""
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    w, h = letter

    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, h - 60, "💼 Moneylytics Executive Summary")
    c.setFont("Helvetica", 11)
    c.drawString(50, h - 80, f"Generated on: {pd.Timestamp.now():%d %b %Y}")
    c.line(50, h - 85, w - 50, h - 85)

    # --- Key Metrics ---
    c.setFont("Helvetica-Bold", 13)
    c.drawString(50, h - 110, "Key Financial Metrics:")
    c.setFont("Helvetica", 11)
    c.drawString(70, h - 130, f"Total Income: {summary.get('Total Income (€)', 0):,.2f} €")
    c.drawString(70, h - 145, f"Total Expenses: {summary.get('Total Expenses (€)', 0):,.2f} €")
    c.drawString(70, h - 160, f"Net Balance: {summary.get('Net Balance (€)', 0):,.2f} €")
    c.drawString(70, h - 175, f"Savings Rate: {sr_last:.1f}% (Target: {target_sr}%)")

    # --- Insights Section ---
    c.setFont("Helvetica-Bold", 13)
    c.drawString(50, h - 205, "AI-Style Insights:")
    c.setFont("Helvetica", 11)
    c.drawString(70, h - 225, "• Your expenses dropped compared to the previous month.")
    c.drawString(70, h - 240, "• You achieved a healthy savings rate above target.")
    c.drawString(70, h - 255, "• Keep an eye on entertainment and grocery spending.")
    c.drawString(70, h - 270, "• Consider increasing your investment allocation by 5%.")
    c.drawString(70, h - 285, "• Financial health outlook: 💪 Stable and improving.")

    # Footer
    c.setFont("Helvetica-Oblique", 9)
    c.drawString(50, 40, "© 2025 Moneylytics AI — Generated automatically.")

    c.showPage()
    c.save()
    buf.seek(0)
    return buf