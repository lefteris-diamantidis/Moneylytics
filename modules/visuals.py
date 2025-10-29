import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.graph_objects as go

# =====================================================
# 🔹 THEME COLOR DETECTOR (unified with app.py)
# =====================================================
def _get_theme_colors():
    base = (st.get_option("theme.base") or "light").lower()
    if base == "dark":
        return {
            "bg": "#0F1116",
            "text": "#E6E8EE",
            "grid": "#2C3340",
            "legend_bg": "#1A1C22",
            "mat_bg": "#0F1116",
            "bar_color": "#64B5F6",
        }
    else:
        return {
            "bg": "#FFFFFF",
            "text": "#0F1116",
            "grid": "#E7EAF0",
            "legend_bg": "#F7F8FA",
            "mat_bg": "#FFFFFF",
            "bar_color": "#1E88E5",
        }


# =====================================================
# 🔹 APPLY PLOTLY THEME
# =====================================================
def _apply_plotly_theme(fig, colors):
    fig.update_layout(
        paper_bgcolor=colors["bg"],
        plot_bgcolor=colors["bg"],
        font=dict(color=colors["text"], family="Inter, sans-serif"),
        title_font=dict(color=colors["text"]),
        legend=dict(bgcolor=colors["legend_bg"], font=dict(color=colors["text"])),
    )
    fig.update_xaxes(
        title_font=dict(color=colors["text"]),
        tickfont=dict(color=colors["text"]),
        gridcolor=colors["grid"],
        zerolinecolor=colors["grid"],
    )
    fig.update_yaxes(
        title_font=dict(color=colors["text"]),
        tickfont=dict(color=colors["text"]),
        gridcolor=colors["grid"],
        zerolinecolor=colors["grid"],
    )
    return fig


# =====================================================
# 🔹 PIE CHART — Spending by Category
# =====================================================
def plot_spending_pie(df):
    colors = _get_theme_colors()

    amount_col = next((c for c in df.columns if "amount" in c.lower()), None)
    category_col = next((c for c in df.columns if "category" in c.lower()), None)

    if not amount_col or not category_col:
        raise ValueError("Missing amount/category columns")

    df = df[df[amount_col] < 0].copy()
    df["Spending"] = df[amount_col].abs()
    cat_summary = df.groupby(category_col)["Spending"].sum().reset_index()

    fig = go.Figure(
        go.Pie(
            labels=cat_summary[category_col],
            values=cat_summary["Spending"],
            hole=0.25,
            textinfo="percent+label",
            textfont=dict(color=colors["text"]),
        )
    )

    fig.update_layout(
        title=dict(text="Spending by Category", x=0.5),
        legend=dict(bgcolor=colors["legend_bg"]),
    )
    return _apply_plotly_theme(fig, colors)


# =====================================================
# 🔹 BAR CHART — Spending by Subcategory
# =====================================================
def plot_spending_bars(df):
    colors = _get_theme_colors()

    amount_col = next((c for c in df.columns if "amount" in c.lower()), None)
    subcat_col = next(
        (c for c in df.columns if "sub" in c.lower() and "cat" in c.lower()), None
    )

    if not amount_col or not subcat_col:
        raise ValueError("Missing subcategory columns")

    df = df[df[amount_col] < 0].copy()
    df["Spending"] = df[amount_col].abs()
    subcat_summary = df.groupby(subcat_col)["Spending"].sum().reset_index()
    subcat_summary = subcat_summary.sort_values(by="Spending", ascending=True)

    fig = go.Figure(
        go.Bar(
            x=subcat_summary["Spending"],
            y=subcat_summary[subcat_col],
            orientation="h",
            marker_color=colors["bar_color"],
        )
    )

    fig.update_layout(
        title=dict(text="Spending by Subcategory", x=0.5),
        showlegend=False,
    )
    return _apply_plotly_theme(fig, colors)


# =====================================================
# 🔹 NET FLOW OVER TIME
# =====================================================
def plot_net_flow_with_highlights(df):
    colors = _get_theme_colors()
    df = df.copy()
    df.columns = [col.lower().replace(" ", "_") for col in df.columns]
    if "month" not in df.columns or "amount" not in df.columns:
        raise ValueError("Requires 'month' and 'amount' columns")

    df = df.sort_values("month")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["month"],
            y=df["amount"],
            mode="lines+markers",
            line=dict(color=colors["bar_color"], width=3),
            hovertemplate="%{x|%b %Y}<br>%{y:.2f}€<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(text="💸 Net Flow Over Time", x=0.5),
        height=500,
    )
    return _apply_plotly_theme(fig, colors)


# =====================================================
# 🔹 INCOME VS EXPENSES
# =====================================================
def plot_income_vs_expenses(df):
    colors = _get_theme_colors()

    date_col = next((c for c in df.columns if "date" in c.lower()), None)
    amount_col = next((c for c in df.columns if "amount" in c.lower()), None)
    if not date_col or not amount_col:
        raise ValueError("Missing date/amount")

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df["Month"] = df[date_col].dt.to_period("M").dt.to_timestamp()
    monthly_income = df[df[amount_col] > 0].groupby("Month")[amount_col].sum()
    monthly_expense = df[df[amount_col] < 0].groupby("Month")[amount_col].sum().abs()

    summary = pd.DataFrame({"Income": monthly_income, "Expenses": monthly_expense}).fillna(0)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=summary.index,
            y=summary["Income"],
            name="Income",
            mode="lines+markers",
            line=dict(color="green", width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=summary.index,
            y=summary["Expenses"],
            name="Expenses",
            mode="lines+markers",
            line=dict(color="red", width=3),
        )
    )

    fig.update_layout(title=dict(text="📊 Income vs Expenses", x=0.5))
    return _apply_plotly_theme(fig, colors)


# =====================================================
# 🔹 SAVINGS RATE
# =====================================================
def plot_savings_rate(df):
    colors = _get_theme_colors()
    date_col = next((c for c in df.columns if "date" in c.lower()), None)
    amount_col = next((c for c in df.columns if "amount" in c.lower()), None)
    if not date_col or not amount_col:
        raise ValueError("Missing date/amount")

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df["Month"] = df[date_col].dt.to_period("M").dt.to_timestamp()
    income = df[df[amount_col] > 0].groupby("Month")[amount_col].sum()
    exp = df[df[amount_col] < 0].groupby("Month")[amount_col].sum().abs()
    summary = pd.DataFrame({"Income": income, "Expenses": exp}).fillna(0)
    summary["Savings Rate (%)"] = ((summary["Income"] - summary["Expenses"]) / summary["Income"].replace(0, pd.NA)) * 100
    summary["Savings Rate (%)"] = summary["Savings Rate (%)"].fillna(0)

    fig = go.Figure(
        go.Scatter(
            x=summary.index,
            y=summary["Savings Rate (%)"],
            mode="lines+markers",
            line=dict(color=colors["bar_color"], width=3),
        )
    )

    fig.update_layout(title=dict(text="💰 Savings Rate Over Time", x=0.5))
    return _apply_plotly_theme(fig, colors)


# =====================================================
# 🔹 DAILY CASHFLOW HEATMAP (Plotly, red→green + adaptive)
# =====================================================
def plot_cashflow_heatmap(df):
    """
    Plotly version of the daily cashflow heatmap across months and weekdays.
    Compatible with Plotly ≥ 6 (uses colorbar.title.font).
    """
    colors = _get_theme_colors()

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["month"] = df["date"].dt.strftime("%b")
    df["weekday"] = df["date"].dt.day_name()

    # Ensure natural weekday & month order
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    # Order months by calendar order present in the data
    if not df["date"].isna().all():
        df["month_ord"] = df["date"].dt.month
        month_order = (
            df.dropna(subset=["month_ord"])
              .sort_values("month_ord")
              .drop_duplicates("month_ord")["month"]
              .tolist()
        )
    else:
        month_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

    pivot = (
        df.pivot_table(index="weekday", columns="month", values="amount", aggfunc="sum")
          .fillna(0)
          .reindex(index=weekday_order)
          .reindex(columns=month_order)
    )

    z_data = pivot.values
    x_labels = pivot.columns.tolist()
    y_labels = pivot.index.tolist()

    # Red → Yellow → Green
    heat_colorscale = [
        [0.0, "#b71c1c"],   # strong red (negative)
        [0.25, "#ef5350"],
        [0.5, "#fdd835"],   # yellow (neutral)
        [0.75, "#81c784"],
        [1.0, "#1b5e20"],   # strong green (positive)
    ]

    text_data = [[f"{v:,.0f}" for v in row] for row in z_data]
    hover_text = [
        [f"{y_labels[i]} - {x_labels[j]}<br>Net: {z_data[i][j]:,.0f}€"
         for j in range(len(x_labels))]
        for i in range(len(y_labels))
    ]

    fig = go.Figure(data=go.Heatmap(
        z=z_data,
        x=x_labels,
        y=y_labels,
        colorscale=heat_colorscale,
        zmid=0,
        text=text_data,
        texttemplate="%{text}",
        textfont={"color": colors["text"], "size": 11},
        hoverinfo="text",
        hovertext=hover_text,
        colorbar=dict(
            title=dict(text="Net (€)", font=dict(color=colors["text"])),  # <-- FIX
            tickfont=dict(color=colors["text"]),
            bgcolor=colors["bg"],
        ),
    ))

    fig.update_layout(
        title=dict(text="💸 Cashflow Heatmap (Income vs Expenses)", x=0.5, font=dict(size=22, color=colors["text"])),
        xaxis=dict(title="Month", showgrid=False, color=colors["text"], tickfont=dict(color=colors["text"])),
        yaxis=dict(title="Weekday", autorange="reversed", showgrid=False, color=colors["text"], tickfont=dict(color=colors["text"])),
        margin=dict(l=60, r=40, t=70, b=40),
        height=600,
    )

    return _apply_plotly_theme(fig, colors)

# =====================================================
# 🔹 MONTHLY TREND COMPARISON (Plotly, elegant + theme adaptive)
# =====================================================
def plot_monthly_trend_comparison(df):
    colors = _get_theme_colors()
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
    monthly = df.groupby("month")["amount"].sum().reset_index()

    # Separate positive / negative months
    monthly["color"] = monthly["amount"].apply(
        lambda x: "#4DB6AC" if x >= 0 else "#E57373"
    )

    fig = go.Figure()

    # Bars
    fig.add_trace(
        go.Bar(
            x=monthly["month"],
            y=monthly["amount"],
            marker_color=monthly["color"],
            text=[f"{x:,.0f}" for x in monthly["amount"]],
            textposition="outside",
            textfont=dict(color=colors["text"], size=11),
            hovertemplate="%{x|%b %Y}<br>Net Balance: %{y:,.2f}€<extra></extra>",
        )
    )

    # Style layout
    fig.update_layout(
        title=dict(
            text="📅 Monthly Net Balance Trend",
            x=0.5,
            font=dict(size=24, color=colors["text"]),
        ),
        xaxis=dict(
            title="Month",
            tickformat="%b %Y",
            tickangle=75,
            color=colors["text"],
            gridcolor=colors["grid"],
            showline=False,
            zeroline=False,
        ),
        yaxis=dict(
            title="Net (€)",
            color=colors["text"],
            gridcolor=colors["grid"],
            zeroline=True,
            zerolinecolor=colors["grid"],
        ),
        paper_bgcolor=colors["bg"],
        plot_bgcolor=colors["bg"],
        font=dict(color=colors["text"], family="Inter, sans-serif"),
        showlegend=False,
        bargap=0.2,
        height=500,
    )

    # Apply global plotly theming helper (ensures consistency)
    fig = _apply_plotly_theme(fig, colors)
    return fig