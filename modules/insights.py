import os
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
import streamlit as st
import google.generativeai as generativeai

# =====================================================
# 🔧 CONFIGURATION
# =====================================================
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("❌ GEMINI_API_KEY not found. Please set it in your .env file.")

genai.configure(api_key=api_key)
generativeai.configure(api_key=api_key)

# =====================================================
# 🧠 CACHED MODEL LOADERS
# =====================================================
@st.cache_resource
def get_gemini_model():
    """Cached Gemini model instance for general insights."""
    return genai.GenerativeModel("gemini-2.5-flash")

@st.cache_resource
def get_chat_model():
    """Cached Gemini model instance for Moneylytics Chat."""
    return genai.GenerativeModel("gemini-2.5-flash")

# =====================================================
# 1️⃣ INSIGHT GENERATOR (STATIC SUMMARY ANALYSIS)
# =====================================================
def generate_insights(summary_text: str, *args, **kwargs) -> str:
    """
    Uses Gemini to analyze summarized spending data and provide financial insights.
    """
    try:
        prompt = f"""
        You are Moneylytics, a friendly and data-savvy personal finance advisor for everyday people.  
        Analyze the following financial summary and generate 3–5 short, meaningful insights.

        Focus on:
        • Identifying trends (e.g., higher spending in certain categories or months)  
        • Noting any warning signs or irregular expenses  
        • Suggesting 1–2 realistic, small changes for better balance or savings  
        • Mentioning any positive progress (to keep motivation high)

        Keep your tone warm, conversational, and practical — like a supportive friend who knows finance.  
        Avoid jargon. Limit the response to 180-200 words.

        Summary:
        {summary_text}
        """

        model = generativeai.GenerativeModel("gemini-2.5-flash")  # free, fast model
        response = model.generate_content(prompt)

        return response.text.strip() if response and response.text else "No insights available."

    except Exception as e:
        return f"⚠️ An error occurred with Gemini API: {e}"

# =====================================================
# 2️⃣ MONEYLYTICS CHATBOT (REAL-TIME Q&A)
# =====================================================

# --- Setup ---
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


def get_financial_summary(df: pd.DataFrame) -> str:
    """Summarize user's financial context safely."""
    if not {"date", "amount", "category"}.issubset(df.columns):
        return "No valid financial data available."

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    total_income = df[df["amount"] > 0]["amount"].sum()
    total_expense = abs(df[df["amount"] < 0]["amount"].sum())
    net = total_income - total_expense

    top_cat = (
        df.groupby("category")["amount"]
        .sum()
        .abs()
        .sort_values(ascending=False)
        .head(5)
    )

    text = (
        f"Total income: €{total_income:,.2f}\n"
        f"Total expenses: €{total_expense:,.2f}\n"
        f"Net balance: €{net:,.2f}\n"
        f"Top categories:\n{top_cat.to_string()}"
    )

    return text


# ==========================================================
# 1️⃣ HELPER: Compute financial context (cached for speed)
# ==========================================================
def get_financial_context(df):
    """Pre-compute simple summary stats to reuse in chat."""
    total_income = df[df["amount"] > 0]["amount"].sum()
    total_expense = abs(df[df["amount"] < 0]["amount"].sum())
    top_category = (
        df.groupby("category")["amount"]
        .sum()
        .abs()
        .sort_values(ascending=False)
        .index[0]
        if "category" in df.columns
        else "General Spending"
    )
    return total_income, total_expense, top_category


# ==========================================================
# 2️⃣ CHAT FUNCTION: Smart conversational analysis
# ==========================================================

# ==========================================================
# MONEYLYTICS AGENT: Data-Aware Conversational Assistant
# ==========================================================
def chat_with_finance_ai(df, user_input, chat_memory=None):
    """
    💬 Moneylytics Chat Assistant (v3)
    Hybrid engine:
    - Answers factual numeric questions using Pandas
    - Uses Gemini for reasoning and advice
    - Includes built-in fallbacks for trend/tip/saving queries
    """
    import pandas as pd, re, google.generativeai as genai

    if chat_memory is None:
        chat_memory = []

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    user_input_lower = user_input.lower().strip()

    # --- Identify key columns ---
    if not {"amount", "category"}.issubset(df.columns):
        return "⚠️ Your dataset must contain at least 'amount' and 'category' columns."

    today = pd.Timestamp.today()
    df_filtered = df.copy()

    # --- Time filters ---
    if "last month" in user_input_lower:
        df_filtered = df[df["date"].dt.month == (today - pd.DateOffset(months=1)).month]
    elif "this month" in user_input_lower:
        df_filtered = df[df["date"].dt.month == today.month]
    elif "this year" in user_input_lower:
        df_filtered = df[df["date"].dt.year == today.year]

    # --- Detect top categories / spendings ---
    def get_top_categories(n=3):
        if df_filtered.empty:
            return []
        cat_sum = (
            df_filtered[df_filtered["amount"] < 0]
            .groupby("category")["amount"]
            .sum()
            .abs()
            .sort_values(ascending=False)
        )
        return cat_sum.head(n).to_dict()

    # ===============================
    # 1️⃣ Factual and descriptive queries
    # ===============================
    if re.search(r"\b(top|biggest|highest)\b.*\b(categories|spend|expenses?)\b", user_input_lower):
        top_cats = get_top_categories()
        if not top_cats:
            return "I couldn’t find any expense data to rank."
        top_str = "\n".join([f"{i+1}. {cat}: €{amt:.2f}" for i, (cat, amt) in enumerate(top_cats.items())])
        return f"Here are your top spending categories:\n{top_str}"

    if "save" in user_input_lower or "saved" in user_input_lower:
        income = df_filtered[df_filtered["amount"] > 0]["amount"].sum()
        expense = abs(df_filtered[df_filtered["amount"] < 0]["amount"].sum())
        savings = income - expense
        if savings > 0:
            rate = (savings / income) * 100 if income > 0 else 0
            return f"You saved approximately **€{savings:.2f}**, which is about **{rate:.1f}%** of your income."
        elif savings < 0:
            return f"You spent about **€{-savings:.2f}** more than you earned — try reducing spending in your top categories."
        return "I couldn’t calculate your savings — please check your income/expense data."

    if "income" in user_input_lower or "earned" in user_input_lower:
        total_income = df_filtered[df_filtered["amount"] > 0]["amount"].sum()
        return f"Your total income during this period is approximately **€{total_income:.2f}**."

    if "expense" in user_input_lower or "spend" in user_input_lower:
        total_expense = abs(df_filtered[df_filtered["amount"] < 0]["amount"].sum())
        return f"Your total expenses in this period are approximately **€{total_expense:.2f}**."

    if "tip" in user_input_lower or "advice" in user_input_lower:
        # fallback local financial tips
        tips = [
            "Try automating a 10% savings transfer each payday.",
            "Track your recurring subscriptions — small ones add up fast.",
            "Review your top 3 expense categories monthly to find easy cuts.",
            "Keep 3–6 months of expenses as an emergency fund.",
            "Set a monthly budget goal, not just expense tracking.",
        ]
        import random
        return f"💡 Financial Tip: {random.choice(tips)}"

    # ===============================
    # 2️⃣ Gemini reasoning fallback
    # ===============================
    try:
        summary_stats = f"""
        Data summary:
        - Records: {len(df)}
        - Date range: {df['date'].min().date()} → {df['date'].max().date()}
        - Total income: €{df[df['amount']>0]['amount'].sum():.2f}
        - Total expenses: €{abs(df[df['amount']<0]['amount'].sum()):.2f}
        """

        chat_context = "\n".join([f"{r}: {m}" for r, m in chat_memory[-5:]])

        prompt = f"""
        You are Moneylytics, a friendly, smart AI finance coach.
        Use this data summary and chat history to provide an insightful answer.

        {summary_stats}

        Chat history:
        {chat_context}

        User question: "{user_input}"

        Respond naturally (2–4 sentences).
        If question is unclear, guess what insight the user probably wants.
        Never say you “couldn’t generate an answer”; always give something practical or motivational.
        """

        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(
            prompt,
            generation_config={"max_output_tokens": 250, "temperature": 0.6},
        )

        text = ""
        if hasattr(response, "candidates"):
            for cand in response.candidates:
                if getattr(cand, "content", None):
                    for part in cand.content.parts:
                        if hasattr(part, "text"):
                            text += part.text
        if not text and hasattr(response, "text"):
            text = response.text

        return text.strip() or "💬 Based on your data, you’re doing well! Keep tracking and optimizing your spending regularly."

    except Exception as e:
        return f"⚠️ Gemini API Error (Chat): {e}"