# -------------------- IMPORT LIBRARIES --------------------
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import zipfile
from io import BytesIO
from modules.loader import load_data
from modules.classifier import categorize_transactions
from modules.analytics import (
    summarize_expenses,
    generate_monthly_summary_excel,
    calculate_budget_progress,
    compare_monthly_trends,
    compute_savings_rate_series,
    detect_recurring_subscriptions,
    detect_category_anomalies,
    income_stability_score,
)
from modules.visuals import (
    plot_spending_pie,
    plot_spending_bars,
    plot_net_flow_with_highlights,
    plot_income_vs_expenses,
    plot_savings_rate,
    plot_cashflow_heatmap,
    plot_monthly_trend_comparison,
    _get_theme_colors,
    _apply_plotly_theme
)
from modules.insights import generate_insights, chat_with_finance_ai, get_gemini_model
from modules.analytics import (
    detect_financial_alerts,
    cumulative_net_worth,
    ai_budget_coach,
    display_extended_kpis,
    generate_executive_summary_pdf
)

# -------------------- PAGE SETUP --------------------
st.set_page_config(
    page_title="Moneylytics - AI Financial Analyst",
    page_icon="💰",
    layout="wide",
)

# -------------------- THEME INITIALIZATION --------------------
# Always start in LIGHT mode by default
if "forced_theme" not in st.session_state:
    st.session_state["forced_theme"] = "light"

# Preserve original getter
if "_orig_get_option" not in st.session_state:
    st.session_state["_orig_get_option"] = st.get_option

def _get_option_patched(key: str):
    """Force Streamlit to use the selected theme, defaulting to Light."""
    if key == "theme.base":
        return st.session_state.get("forced_theme", "light")
    return st.session_state["_orig_get_option"](key)

st.get_option = _get_option_patched

# -------------------- THEME VARIABLES --------------------
THEME = st.session_state["forced_theme"]
IS_DARK = THEME == "dark"
THEME_KEY = f"theme_{THEME}"  # 👈 Force CSS re-render on theme change

BG = "#0F1116" if IS_DARK else "#FFFFFF"
FG = "#E6E8EE" if IS_DARK else "#0F1116"
SUBTLE = "#2A2F3A" if IS_DARK else "#F2F4F7"
ACCENT_1 = "#64B5F6" if IS_DARK else "#1E88E5"
ACCENT_2 = "#9575CD" if IS_DARK else "#673AB7"
CARD_BG = "#171A21" if IS_DARK else "#FFFFFF"
BORDER = "#2C3340" if IS_DARK else "#E7EAF0"
SIDEBAR_BG = "#111418" if IS_DARK else "#F7F8FA"


# -------------------- GLOBAL STYLES (dynamic with THEME_KEY) --------------------
def apply_dynamic_css(theme_key: str):
    """Inject full dynamic CSS styling that adapts to dark/light mode."""
    IS_DARK = st.session_state.get("forced_theme", "light") == "dark"

    BG = "#0F1116" if IS_DARK else "#FFFFFF"
    FG = "#FFFFFF" if IS_DARK else "#000000"
    CARD_BG = "#171A21" if IS_DARK else "#FFFFFF"
    BORDER = "#2C3340" if IS_DARK else "#E7EAF0"
    SIDEBAR_BG = "#111418" if IS_DARK else "#F7F8FA"
    SIDEBAR_FG = "#FFFFFF" if IS_DARK else "#000000"
    ACCENT_1 = "#64B5F6" if IS_DARK else "#1E88E5"
    ACCENT_2 = "#9575CD" if IS_DARK else "#673AB7"
    SUBTLE = "#2A2F3A" if IS_DARK else "#F2F4F7"
    UPLOAD_BG = "#1C1F26" if IS_DARK else "#F9FAFB"

    st.markdown(f"""
    <style id="{theme_key}">
    html, body, .stApp {{
        background-color: {BG} !important;
        color: {FG} !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }}

    /* ========================================
       TEXT & HEADERS
    ======================================== */
    h1, h2, h3, h4, h5, h6, label, p, span, div, strong, b {{
        color: {FG} !important;
    }}
    h1 {{
        text-align:center;
        font-size:2.2rem;
        font-weight:800;
        margin-bottom:1.2rem;
        background: linear-gradient(90deg, {ACCENT_1}, {ACCENT_2});
        -webkit-background-clip:text;
        -webkit-text-fill-color:transparent;
    }}

    /* ========================================
       SIDEBAR
    ======================================== */
    [data-testid="stSidebar"] {{
        background-color: {SIDEBAR_BG} !important;
        color: {SIDEBAR_FG} !important;
        border-right: 1px solid {BORDER} !important;
    }}
    [data-testid="stSidebar"] * {{
        color: {SIDEBAR_FG} !important;
    }}

    /* Sidebar buttons */
    [data-testid="stSidebar"] .stButton>button {{
        background: linear-gradient(90deg, {ACCENT_1}, {ACCENT_2}) !important;
        color: #FFFFFF !important;
        font-weight: 800 !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.6rem 1rem !important;
        text-align: center !important;
        transition: all 0.25s ease-in-out;
    }}
    [data-testid="stSidebar"] .stButton>button:hover {{
        opacity: 0.9 !important;
        transform: scale(1.03);
    }}

    /* ========================================
       MAIN BUTTONS
    ======================================== */
    .stButton>button,
    .stDownloadButton>button {{
        background: linear-gradient(90deg, {ACCENT_1}, {ACCENT_2}) !important;
        color: #FFFFFF !important;
        border: none;
        border-radius: 10px;
        font-weight: 700;
        padding: 0.5rem 1rem;
        transition: all 0.2s ease-in-out;
    }}
    .stButton>button:hover,
    .stDownloadButton>button:hover {{
        opacity: 0.9;
        transform: scale(1.02);
    }}

    /* ========================================
       INPUTS & TEXTBOXES
    ======================================== */
    input, select, textarea {{
        color: {FG} !important;
        background-color: {CARD_BG} !important;
        border: 1px solid {BORDER} !important;
    }}
    .stTextInput input,
    .stSelectbox div[data-baseweb="select"] > div,
    .stMultiSelect div[data-baseweb="select"] > div,
    .stDateInput input,
    .stNumberInput input,
    .stTextArea textarea {{
        color: {FG} !important;
        background-color: {CARD_BG} !important;
        border: 1px solid {BORDER} !important;
    }}

    /* ========================================
       DROPDOWN MENUS
    ======================================== */
    [data-baseweb="popover"] {{
        background-color: {CARD_BG} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 8px !important;
    }}
    [data-baseweb="popover"] * {{
        background-color: {CARD_BG} !important;
        color: {FG} !important;
    }}
    [data-baseweb="menu-item"]:hover {{
        background: linear-gradient(90deg, {ACCENT_1}22, {ACCENT_2}22) !important;
        color: {FG} !important;
    }}

    /* ========================================
       FILE UPLOADER (outer box)
    ======================================== */
    [data-testid="stFileUploader"] {{
        background-color: {SUBTLE} !important;
        border-radius: 10px !important;
        border: 1px solid {BORDER} !important;
        padding: 1rem !important;
    }}
    [data-testid="stFileUploader"] * {{
        color: {FG} !important;
    }}

    /* Inner drag/drop area */
    [data-testid="stFileUploaderDropzone"] {{
        background-color: {UPLOAD_BG} !important;
        color: {FG} !important;
        border: 2px dashed {BORDER} !important;
        border-radius: 12px !important;
        text-align: center !important;
        transition: all 0.3s ease-in-out !important;
    }}
    [data-testid="stFileUploaderDropzone"]:hover {{
        border-color: {ACCENT_1} !important;
        box-shadow: 0 0 10px {ACCENT_1}33 !important;
    }}

    /* "Drag and drop" text */
    [data-testid="stFileUploaderDropzone"] div:nth-child(1) {{
        font-weight: 600 !important;
        font-size: 1.05rem !important;
        color: {FG} !important;
    }}

    /* "Browse files" button inside uploader */
    [data-testid="stFileUploaderDropzone"] button {{
        background: linear-gradient(90deg, {ACCENT_1}, {ACCENT_2}) !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.4rem 1.2rem !important;
        margin-top: 0.5rem !important;
        transition: all 0.25s ease-in-out !important;
    }}
    [data-testid="stFileUploaderDropzone"] button:hover {{
        opacity: 0.9 !important;
        transform: scale(1.03);
    }}

    /* ========================================
       METRICS
    ======================================== */
    [data-testid="stMetric"] {{
        background: {CARD_BG} !important;
        border: 1px solid {BORDER};
        border-radius: 12px;
        padding: 0.8rem;
        text-align: center;
        color: {FG} !important;
    }}
    [data-testid="stMetricLabel"],
    [data-testid="stMetricValue"],
    [data-testid="stMetricDelta"] {{
        color: {FG} !important;
    }}

    /* ========================================
       PLOTLY CHARTS
    ======================================== */
    .plotly-chart, .stPlotlyChart, .plot-container {{
        background-color: {CARD_BG} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 12px;
        padding: 0.6rem;
        color: {FG} !important;
    }}

    /* ========================================
       DATAFRAMES
    ======================================== */
    .dataframe tbody tr, .dataframe thead th {{
        color: {FG} !important;
        background-color: {CARD_BG} !important;
    }}

    /* Smooth transitions */
    * {{
        transition: background-color 0.3s ease, color 0.3s ease;
    }}
    </style>
    """, unsafe_allow_html=True)

# Call it once after THEME setup:
apply_dynamic_css(THEME_KEY)

# -------------------- SIDEBAR --------------------
st.sidebar.header("⚙️ Moneylytics Settings")
currency = st.sidebar.selectbox("Currency Symbol", ["€", "$", "£", "₺", "₴"], index=0)

toggle_label = "☀️ Switch to Light Mode" if IS_DARK else "🌙 Switch to Dark Mode"
if st.sidebar.button(toggle_label, use_container_width=True):
    st.session_state["forced_theme"] = "light" if IS_DARK else "dark"
    st.rerun()

# Savings target + alerts
st.sidebar.header("🎯 Goals & Alerts")
target_sr = st.sidebar.slider("Target Savings Rate (%)", min_value=0, max_value=50, value=20, step=1)
show_alerts = st.sidebar.checkbox("Show budget/overspend alerts", value=True)

# -------------------- TITLE & PRIVACY & USER GUIDE--------------------

st.title("💰 Moneylytics - AI Personal Financial Analyst")

with st.expander("📘 User Guide — How to Use Moneylytics"):
    st.markdown("""
        **Welcome to Moneylytics!** Here's a quick guide to make the most of your dashboard:

        1️⃣ **Upload your file (.xlsx)**  
        → Must contain columns: *Date*, *Amount*, *Description*, *Category*, *(optional) Sub-Category*.

        2️⃣ **Explore the Dashboard**  
        → View income, expenses, savings rate, and trends interactively.

        3️⃣ **Track your Budgets**  
        → Set category budgets and see progress bars & alerts.

        4️⃣ **AI Insights & Chat**  
        → Go to the *🧠 Insights* tab to ask questions like:
        - “How much did I spend on food last month?”
        - “What’s my savings rate trend?”
        - “Give me a financial tip!”

        5️⃣ **Export Your Data**  
        → In the *📤 Export & Share* section:
        - Download **Excel**, **PDF**, **HTML**, or **ZIP** reports.

        💡 *All data stays local and private.*
        """)

st.markdown("Upload your Excel and explore insights, budgets, and trends.")

with st.expander("🔒 Data Privacy Notice"):
    st.markdown("Moneylytics processes your data locally for this session. Exports are explicit and user-triggered.")
    
# -------------------- FILE UPLOAD (persist across reruns) --------------------
uploaded_file = st.file_uploader("📂 Upload your Excel file (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    st.session_state["uploaded_file_bytes"] = uploaded_file.getvalue()
elif "uploaded_file_bytes" in st.session_state:
    from io import BytesIO
    uploaded_file = BytesIO(st.session_state["uploaded_file_bytes"])
else:
    st.info("Please upload an Excel file (.xlsx) to continue.")
    st.stop()

try:
    with st.spinner("🔍 Reading and processing your data..."):
        df = load_data(uploaded_file)
        df = categorize_transactions(df)

    st.success(f"✅ Loaded {len(df)} transactions successfully!")

    # ---- Column discovery ----
    def _find_col_with_keywords(df, keywords):
        for col in df.columns:
            low = col.lower().replace("_", "").replace("-", "").replace(" ", "")
            for kw in keywords:
                if kw.lower().replace("_", "").replace("-", "") in low:
                    return col
        return None

    if (
        "date_col" not in st.session_state
        or "amount_col" not in st.session_state
        or "category_col" not in st.session_state
        or "subcat_col" not in st.session_state
    ):
        st.session_state.date_col = _find_col_with_keywords(df, ["date", "transactiondate"])
        st.session_state.amount_col = _find_col_with_keywords(df, ["amount", "value", "transaction"])
        st.session_state.category_col = _find_col_with_keywords(df, ["category", "maincategory"])
        st.session_state.subcat_col = _find_col_with_keywords(df, ["sub-category", "subcategory", "sub_category"])

    date_col = st.session_state.date_col
    amount_col = st.session_state.amount_col
    category_col = st.session_state.category_col
    subcat_col = st.session_state.subcat_col

    if date_col and df[date_col].dtype != "datetime64[ns]":
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)

    # ---- Banner ----
    st.markdown(
        f"""
        <div style='text-align:center; font-size:18px; background:{CARD_BG}; color:{FG};
                    padding:10px; border-radius:10px; border:1px solid {BORDER};'>
        📅 Active Period: <b>{(df[date_col].min()).strftime('%b %Y') if date_col else '-'}</b> –
        <b>{(df[date_col].max()).strftime('%b %Y') if date_col else '-'}</b> &nbsp; | &nbsp;
        💼 Total Transactions: <b>{len(df):,}</b>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -------------------- TABS --------------------
    tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🧠 Insights", "📁 Explorer"])

    # =======================
    # TAB 1: DASHBOARD
    # =======================
    with tab1:
        # ---- Filters (Sidebar) ----
        st.sidebar.header("🧪 Filters")
        min_date = df[date_col].min() if date_col else None
        max_date = df[date_col].max() if date_col else None

        if min_date is not None and max_date is not None:
            date_range = st.sidebar.date_input("Date range", [min_date.date(), max_date.date()])
            start_date = pd.to_datetime(date_range[0])
            end_date = pd.to_datetime(date_range[1])
        else:
            start_date, end_date = None, None

        if category_col and df[category_col].notna().any():
            unique_cats = sorted(df[category_col].dropna().unique().tolist())
            selected_cats = st.sidebar.multiselect("Categories", options=unique_cats, default=unique_cats)
        else:
            selected_cats = None

        if subcat_col and df[subcat_col].notna().any():
            unique_subcats = sorted(df[subcat_col].dropna().unique().tolist())
            selected_subcats = st.sidebar.multiselect("Sub-Categories", options=unique_subcats, default=unique_subcats)
        else:
            selected_subcats = None

        if amount_col:
            df[amount_col] = (
                df[amount_col].astype(str).str.replace("€", "", regex=False).str.replace(",", "").str.strip()
            )
            df[amount_col] = pd.to_numeric(df[amount_col], errors="coerce")
            min_amt, max_amt = float(np.nanmin(df[amount_col])), float(np.nanmax(df[amount_col]))
            amt_range = st.sidebar.slider("Amount range", min_value=min_amt, max_value=max_amt, value=(min_amt, max_amt))
        else:
            amt_range = None

        filtered_df = df.copy()
        if date_col and start_date is not None and end_date is not None:
            filtered_df = filtered_df[
                (filtered_df[date_col] >= start_date)
                & (filtered_df[date_col] <= end_date + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))
            ]
        if selected_cats:
            filtered_df = filtered_df[filtered_df[category_col].isin(selected_cats)]
        if selected_subcats:
            filtered_df = filtered_df[filtered_df[subcat_col].isin(selected_subcats)]
        if amt_range:
            filtered_df = filtered_df[(filtered_df[amount_col] >= amt_range[0]) & (filtered_df[amount_col] <= amt_range[1])]

        df = filtered_df.copy()

        # ---- Raw preview ----
        with st.expander("📋 View uploaded cleaned data"):
            st.dataframe(df.head(20))

        # ---- Analysis ----
        summary, cat_summary, trend_df = summarize_expenses(df)
        
        st.subheader("💰 Financial Summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Income", f"{summary['Total Income (€)']:,} {currency}")
        c2.metric("Total Expenses", f"{summary['Total Expenses (€)']:,} {currency}")
        c3.metric("Net Balance", f"{summary['Net Balance (€)']:,} {currency}")

        # Savings rate vs target
        sr_series, sr_last = compute_savings_rate_series(df)
        badge = "✅ On Track" if sr_last >= target_sr else "⚠️ Below Target"
        c4.metric("Savings Rate (last mo)", f"{sr_last:.1f}%", f"Target {target_sr}%", help=badge)
        st.progress(min(sr_last / max(target_sr, 1), 1.0))

        # Income stability score
        st.caption(f"Income stability score: **{income_stability_score(df)} / 100** (higher = steadier income)")

        # ---- Trend KPIs (Best/Worst/Avg/Change) ----
        try:
            tf = trend_df.copy().sort_values("month")
            tf["amount"] = pd.to_numeric(tf["amount"], errors="coerce").fillna(0)

            best_month_row = tf.loc[tf["amount"].idxmax()]
            worst_month_row = tf.loc[tf["amount"].idxmin()]
            avg_monthly = tf["amount"].mean()
            pct_change = (
                ((tf["amount"].iat[-1] - tf["amount"].iat[-2]) / abs(tf["amount"].iat[-2]) * 100)
                if len(tf) >= 2 and tf["amount"].iat[-2] != 0 else 0.0
            )

            k1, k2, k3, k4 = st.columns(4)
            k1.metric(
                "Best Month",
                pd.to_datetime(best_month_row["month"]).strftime("%b %Y"),
                f"+{best_month_row['amount']:.2f}{currency}",
            )
            k2.metric(
                "Worst Month",
                pd.to_datetime(worst_month_row["month"]).strftime("%b %Y"),
                f"{worst_month_row['amount']:.2f}{currency}",
            )
            k3.metric("Avg Monthly Net", f"{avg_monthly:.2f}{currency}")
            k4.metric("Change vs Prev Month", f"{pct_change:.1f}%", delta=f"{pct_change:.1f}%")
        except Exception:
            pass
        
        # ---- Category Budget Tracker ----
        st.subheader("🎯 Category Budget Tracker")
        expense_df = df[df[amount_col] < 0]
        if category_col and expense_df[category_col].notna().any():
            budget_cats = sorted(expense_df[category_col].dropna().unique().tolist())[:12]
        else:
            st.info("No expense categories detected.")
            budget_cats = []

        user_budgets = {}
        cols_for_budgets = st.columns(3)
        for i, cat in enumerate(budget_cats):
            with cols_for_budgets[i % 3]:
                key_name = f"budget_{cat}"
                user_budgets[cat] = st.number_input(
                    f"Budget for {cat}", min_value=0.0, value=200.0, step=10.0, key=key_name
                )

        if st.button("💰 Calculate Budget Progress"):
            try:
                results = calculate_budget_progress(
                    df.rename(columns={date_col: "date", amount_col: "amount", category_col: "category"}),
                    user_budgets,
                )
                result_df = pd.DataFrame(results)

                if result_df.empty:
                    st.info("No expense data available for the selected categories.")
                else:
                    # Show raw table
                    st.dataframe(result_df)

                    # -------- Budget Progress Overview (per category, latest month) --------
                    st.markdown("### 📊 Budget Progress Overview")

                    # Use the latest month per category
                    latest_per_cat = (
                        result_df.sort_values("Month")
                        .groupby("Category", as_index=False)
                        .tail(1)
                        .reset_index(drop=True)
                    )

                    # Progress bars (green within budget, red if over)
                    for _, row in latest_per_cat.iterrows():
                        progress = 0.0
                        if row["Budget (€)"] > 0:
                            progress = min(row["Spent (€)"] / row["Budget (€)"], 1.0)
                        color = "green" if row["Progress (%)"] <= 100 else "red"
                        st.markdown(
                            f"**{row['Category']} ({row['Month']})** — "
                            f"Spent: {row['Spent (€)']:.2f}€ / {row['Budget (€)']:.2f}€ "
                            f"<span style='color:{color}'>({row['Progress (%)']:.1f}%)</span>",
                            unsafe_allow_html=True,
                        )
                        st.progress(progress)

                    # -------- Plotly bar: Spent vs Budget (latest month per category) --------
                    import plotly.graph_objects as go

                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=latest_per_cat["Category"],
                        y=latest_per_cat["Budget (€)"],
                        name="Budget",
                        marker_color="#90CAF9",
                    ))
                    fig.add_trace(go.Bar(
                        x=latest_per_cat["Category"],
                        y=latest_per_cat["Spent (€)"],
                        name="Spent",
                        marker_color="#E57373",
                    ))

                    fig.update_layout(
                        title="Budget vs Spent (Latest Month per Category)",
                        barmode="group",
                        xaxis_title="Category",
                        yaxis_title="€",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                    )

                    # Apply your unified theme colors from visuals helper
                    from modules.visuals import _get_theme_colors, _apply_plotly_theme
                    _colors = _get_theme_colors()
                    fig = _apply_plotly_theme(fig, _colors)

                    st.plotly_chart(fig, use_container_width=True)

            except Exception as ex:
                st.error(f"⚠️ Could not compute budget progress: {ex}")
                
        # =====================================================
        # 💹 ADVANCED ANALYTICS — FINANCIAL HEALTH & INSIGHTS
        # =====================================================
        st.subheader("💡 Financial Health Overview")

        from modules.analytics import (
            calculate_financial_health_index,
            expense_to_income_ratio,
            category_correlation_matrix,
        )
        from modules.visuals import _get_theme_colors, _apply_plotly_theme
        import plotly.graph_objects as go
        import plotly.express as px

        colors = _get_theme_colors()

        # --- 1️⃣ Financial Health Index ---
        try:
            health_score, health_msg = calculate_financial_health_index(
                df.rename(columns={date_col: "date", amount_col: "amount"}),
                sr_last, target_sr, result_df if "result_df" in locals() else None
            )
            st.metric("🏅 Financial Health Index", f"{health_score}/100")
            st.caption(health_msg)
        except Exception as ex:
            st.warning(f"Could not compute Financial Health Index: {ex}")

        # --- 2️⃣ Expense-to-Income Ratio ---
        try:
            ratio_df = expense_to_income_ratio(
                df.rename(columns={date_col: "date", amount_col: "amount"})
            )

            fig_ratio = go.Figure()
            fig_ratio.add_trace(
                go.Scatter(
                    x=ratio_df["Month"],
                    y=ratio_df["Expense/Income (%)"],
                    mode="lines+markers",
                    line=dict(width=3, color="#FFA726"),
                    hovertemplate="%{x|%b %Y}<br>%{y:.1f}%<extra></extra>",
                )
            )

            fig_ratio.update_layout(
                title="📉 Expense-to-Income Ratio Over Time",
                yaxis_title="Expense/Income (%)",
                xaxis_title="Month",
                height=400,
            )
            fig_ratio = _apply_plotly_theme(fig_ratio, colors)
            st.plotly_chart(fig_ratio, use_container_width=True)
        except Exception as ex:
            st.warning(f"Could not compute Expense-to-Income ratio: {ex}")

        # --- 3️⃣ Category Correlation Matrix ---
        try:
            corr = category_correlation_matrix(
                df.rename(columns={date_col: "date", amount_col: "amount", category_col: "category"})
            )

            fig_corr = px.imshow(
                corr,
                text_auto=True,
                color_continuous_scale="RdBu",
                zmin=-1,
                zmax=1,
                title="🔗 Category Correlation Matrix",
                aspect="auto",
            )
            fig_corr = _apply_plotly_theme(fig_corr, colors)
            st.plotly_chart(fig_corr, use_container_width=True)
        except Exception as ex:
            st.warning(f"Could not generate correlation matrix: {ex}")
            
        # ---- 💎 NEW BLOCK: EXTENDED KPIs ----
        display_extended_kpis(
            df.rename(columns={date_col: "date", amount_col: "amount", category_col: "category"}),
            currency
        )

        # ---- Visuals ----
        st.subheader("📈 Spending Breakdown")
        st.plotly_chart(plot_spending_pie(df), use_container_width=True)
        st.plotly_chart(plot_spending_bars(df), use_container_width=True)

        st.subheader("📅 Spending Over Time")
        st.plotly_chart(plot_net_flow_with_highlights(trend_df), use_container_width=True)

        try:
            tf2 = trend_df.copy().sort_values("month")
            tf2["amount"] = pd.to_numeric(tf2["amount"], errors="coerce").fillna(0)
            x = np.arange(len(tf2)).reshape(-1, 1)
            y = tf2["amount"].values
            if len(x) >= 2:
                from sklearn.linear_model import LinearRegression
                lr = LinearRegression().fit(x, y)
                forecast_value = float(lr.predict([[len(x)]])[0])
                st.metric("📈 Forecast (next month net)", f"{forecast_value:.2f}{currency}")
        except Exception:
            pass

        st.subheader("📊 Income vs Expenses Over Time")
        st.plotly_chart(plot_income_vs_expenses(df), use_container_width=True)

        st.subheader("💡 Savings Rate Over Time")
        st.plotly_chart(plot_savings_rate(df), use_container_width=True)

        st.subheader("📅 Monthly Trend Comparison")
        try:
            monthly_compare = compare_monthly_trends(
                df.rename(columns={date_col: "date", amount_col: "amount"})
            )
            display_cols = [c for c in ["month", "income", "expenses", "income_change", "expense_change"] if c in monthly_compare.columns]
            st.dataframe(monthly_compare[display_cols].reset_index(drop=True))
        except Exception as ex:
            st.warning(f"Could not compute monthly comparison: {ex}")

        try:
            st.plotly_chart(
                plot_monthly_trend_comparison(
                    df.rename(columns={date_col: "date", amount_col: "amount"})
                ),
                use_container_width=True,
            )
        except Exception as ex:
            st.warning(f"Could not display Monthly Trend Comparison: {ex}")

        st.subheader("🔥 Cashflow Heatmap")
        try:
            st.plotly_chart(
                plot_cashflow_heatmap(
                    df.rename(columns={date_col: "date", amount_col: "amount"})
                ),
                use_container_width=True,
            )
        except Exception as ex:
            st.warning(f"Could not display Cashflow Heatmap: {ex}")
            
        # =====================================================
        # 🚨 SMART ALERTS & NET WORTH + AI COACH
        # =====================================================
        st.subheader("🚨 Financial Alerts & AI Coach")

        colors = _get_theme_colors()

        # --- 1️⃣ SMART ALERTS ---
        try:
            alerts = detect_financial_alerts(
                df.rename(columns={date_col: "date", amount_col: "amount"}),
                result_df if "result_df" in locals() else None,
            )
            if alerts:
                for a in alerts:
                    st.markdown(f"- {a}")
            else:
                st.success("✅ No critical alerts detected. You're financially stable!")
        except Exception as ex:
            st.warning(f"Could not compute alerts: {ex}")

        # --- 2️⃣ NET WORTH GROWTH ---
        try:
            net_df = cumulative_net_worth(df.rename(columns={date_col: "date", amount_col: "amount"}))
            fig_net = go.Figure()
            fig_net.add_trace(
                go.Scatter(
                    x=net_df["date"],
                    y=net_df["cumulative_net"],
                    mode="lines+markers",
                    line=dict(width=3, color="#66BB6A"),
                    hovertemplate="%{x|%b %Y}<br>Cumulative Net: %{y:,.0f}€<extra></extra>",
                )
            )
            fig_net.update_layout(
                title="💹 Cumulative Net Worth Growth",
                yaxis_title="Cumulative Net (€)",
                xaxis_title="Date",
                height=400,
            )
            fig_net = _apply_plotly_theme(fig_net, colors)
            st.plotly_chart(fig_net, use_container_width=True)
        except Exception as ex:
            st.warning(f"Could not display Net Worth chart: {ex}")

        # --- 3️⃣ AI BUDGET COACH ---
        st.markdown("### 🤖 AI Budget Coach — Smart Recommendations")
        try:
            if st.button("💬 Generate AI Budget Suggestions"):
                model = get_gemini_model()
                ai_response = ai_budget_coach(cat_summary, model)
                st.markdown(ai_response)
        except Exception as ex:
            st.warning(f"Could not run AI Budget Coach: {ex}")

        # ---- Recurring subscriptions ----
        st.subheader("🔁 Recurring Subscriptions (Heuristic)")
        try:
            subs = detect_recurring_subscriptions(df)
            if subs.empty:
                st.info("No recurring patterns detected yet.")
            else:
                st.dataframe(subs)
        except Exception as ex:
            st.warning(f"Could not detect subscriptions: {ex}")

        # ---- Anomalies ----
        st.subheader("🚨 Category Anomalies")
        try:
            anom = detect_category_anomalies(df)
            if anom.empty:
                st.info("No unusual category spikes detected.")
            else:
                st.dataframe(anom)
        except Exception as ex:
            st.warning(f"Could not compute anomalies: {ex}")

        # ---- Exports ----
        # =====================================================
        # 📤 EXPORT SUITE — Excel, Executive PDF, Interactive HTML, ZIP
        # =====================================================
        st.markdown("### 📤 Export and Share")

        # --- 1️⃣ Monthly Summary Excel ---
        buffer_excel = generate_monthly_summary_excel(df)

        # --- 2️⃣ Executive Summary PDF ---
        from modules.analytics import generate_executive_summary_pdf
        pdf_summary = generate_executive_summary_pdf(df, summary, sr_last, target_sr)

        # --- 3️⃣ Interactive Dashboard HTML ---
        import plotly.io as pio
        figs_to_export = {
            "Income vs Expenses": plot_income_vs_expenses(df),
            "Savings Rate Over Time": plot_savings_rate(df),
            "Monthly Trend Comparison": plot_monthly_trend_comparison(df),
            "Cashflow Heatmap": plot_cashflow_heatmap(df)
        }
        html_buffer = BytesIO()
        html_content = "<h1>💰 Moneylytics Interactive Dashboard</h1>"
        for title, fig in figs_to_export.items():
            html_content += f"<h3>{title}</h3>" + pio.to_html(fig, full_html=False, include_plotlyjs='cdn')
        html_buffer.write(html_content.encode("utf-8"))
        html_buffer.seek(0)

        # --- BUTTONS ---
        colA, colB, colC = st.columns(3)

        with colA:
            st.download_button(
                label="📊 Monthly Summary (Excel)",
                data=buffer_excel,
                file_name="moneylytics_monthly_summary.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

        with colB:
            st.download_button(
                label="🧾 Executive Summary (PDF)",
                data=pdf_summary,
                file_name="moneylytics_executive_summary.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

        with colC:
            st.download_button(
                label="🌐 Interactive Dashboard (HTML)",
                data=html_buffer,
                file_name="moneylytics_dashboard.html",
                mime="text/html",
                use_container_width=True,
            )

        # --- 4️⃣ ZIP BUNDLE ---
        cat_csv = cat_summary.to_csv(index=True).encode("utf-8")

        def build_full_export_zip(buffers_dict):
            zip_buffer = BytesIO()
            with zipfile.ZipFile(zip_buffer, "w") as zf:
                for name, data in buffers_dict.items():
                    zf.writestr(name, data.getvalue() if hasattr(data, "getvalue") else data)
            zip_buffer.seek(0)
            return zip_buffer

        zip_bundle = build_full_export_zip({
            "Monthly_Summary.xlsx": buffer_excel,
            "Executive_Summary.pdf": pdf_summary,
            "Interactive_Dashboard.html": html_buffer,
            "Category_Summary.csv": cat_csv,
        })

        st.markdown("### 📦 Complete Export Bundle")
        c1, c2, c3 = st.columns([1, 1, 1])
        with c2:
            st.download_button(
                label="📦 Download All Reports (ZIP)",
                data=zip_bundle,
                file_name="moneylytics_full_export.zip",
                mime="application/zip",
                use_container_width=True,
            )

    # =======================
    # TAB 2: INSIGHTS
    # =======================
    with tab2:
        st.subheader("🧠 AI-Generated Insights")

        if "insights_text" not in st.session_state:
            st.session_state["insights_text"] = None
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []
        if "smartfin_chat" not in st.session_state:
            st.session_state["smartfin_chat"] = ""

        if st.session_state["insights_text"] is None:
            with st.spinner("Generating personalized insights..."):
                try:
                    # pass a short summary string for LLM tone
                    sum_text = f"Income={summary['Total Income (€)']}, Expenses={summary['Total Expenses (€)']}, Net={summary['Net Balance (€)']}"
                    st.session_state["insights_text"] = generate_insights(sum_text)
                except Exception as e:
                    st.session_state["insights_text"] = f"⚠️ Could not generate insights: {e}"

        st.info(st.session_state["insights_text"])

        st.markdown("### 💬 Ask Moneylytics")
        preset_questions = [
            "How much did I spend on food last month?",
            "Show me my top 3 spending categories.",
            "What was my income last month?",
            "Compare my expenses over time.",
            "How much did I save this year?",
            "Give me a financial tip!",
        ]
        st.markdown("Try asking one of these common questions 👇")
        cols = st.columns(3)
        selected_preset = None
        for i, question in enumerate(preset_questions):
            if cols[i % 3].button(question, key=f"preset_btn_{i}"):
                selected_preset = question

        user_input = st.text_input("Or type your own question 👇", key="smartfin_input_box", placeholder="e.g., What was my total spending in 2024?")
        ask_button = st.button("💬 Ask Moneylytics")

        active_question = selected_preset or (user_input.strip() if user_input else None)
        if ask_button or selected_preset:
            if active_question:
                try:
                    st.session_state.chat_history.append(("You", active_question))
                    chat_response = chat_with_finance_ai(df, active_question, chat_memory=st.session_state.chat_history)
                    st.session_state.chat_history.append(("Moneylytics", chat_response))
                except Exception as ex:
                    st.error(f"⚠️ Chat failed: {ex}")
            else:
                st.warning("Please type a question or select one from above before asking.")

        for speaker, message in st.session_state.chat_history[-10:]:
            if speaker == "You":
                st.markdown(f"<div class='chat-bubble-user'>🧍 <b>You:</b> {message}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='chat-bubble-ai'>🤖 <b>Moneylytics:</b> {message}</div>", unsafe_allow_html=True)

    # =======================
    # TAB 3: EXPLORER
    # =======================
    with tab3:
        st.subheader("🔎 Transaction Explorer")
        search_text = st.text_input("Search description (contains):", value="")
        explorer_df = df.copy()
        if search_text:
            desc_col = next((c for c in explorer_df.columns if "desc" in c.lower()), None)
            if desc_col:
                explorer_df = explorer_df[explorer_df[desc_col].str.contains(search_text, case=False, na=False)]

        if category_col and explorer_df[category_col].notna().any():
            all_cats = sorted(explorer_df[category_col].dropna().unique().tolist())
            chosen_cat = st.selectbox("Drill-down Category", options=["(all)"] + all_cats, index=0)
            if chosen_cat != "(all)":
                explorer_df = explorer_df[explorer_df[category_col] == chosen_cat]

        st.dataframe(explorer_df.reset_index(drop=True))
        csv_bytes = explorer_df.to_csv(index=False).encode("utf-8")
        c1, c2, c3 = st.columns([1, 1, 1])
        with c2:
            st.download_button("💾 Download filtered transactions (CSV)", data=csv_bytes, file_name="filtered_transactions.csv", mime="text/csv")

except Exception as e:
    st.error(f"⚠️ An error occurred: {e}")
    st.stop()


st.markdown("---")
st.caption("💡 Built with Streamlit. © 2025 Moneylytics AI")
