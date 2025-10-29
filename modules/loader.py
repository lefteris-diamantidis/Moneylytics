import pandas as pd

def load_data(uploaded_file):
    df = pd.read_excel(uploaded_file)

    # --- Normalize column names ---
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # --- Ensure required columns ---
    if "date" not in df.columns or "amount" not in df.columns:
        raise ValueError("Excel file must include 'Date' and 'Amount' columns.")

    # --- Clean string columns (preserve readable words) ---
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()

    # --- Clean and convert amount ---
    df["amount"] = (
        df["amount"]
        .astype(str)
        .str.replace("€", "", regex=False)
        .str.replace(",", ".", regex=False)
        .str.replace(" ", "", regex=False)
    )
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

    # --- Parse dates ---
    df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)

    # --- Fix sign per row ---
    def fix_amount(row):
        category = str(row.get("category", "")).lower().replace(" ", "_")
        subcat = str(row.get("sub-category", "")).lower().replace(" ", "_")
        amt = row["amount"]

        if pd.isna(amt):
            return 0

        # Positive if income-related
        if any(word in category for word in ["income", "main_income", "side_income"]) \
        or any(word in subcat for word in ["income", "main_income", "side_income"]):
            return abs(amt)
        # Otherwise negative
        return -abs(amt)

    df["amount_fixed"] = df.apply(fix_amount, axis=1)

    # --- Keep relevant columns ---
    keep_cols = [
        c for c in ["date", "description", "amount_fixed", "category", "sub-category"]
        if c in df.columns
    ]
    df = df[keep_cols]

    # --- Rename for consistency ---
    df.rename(columns={"amount_fixed": "amount"}, inplace=True)

    # --- Drop rows without valid amounts or dates ---
    df = df.dropna(subset=["amount", "date"])

    return df