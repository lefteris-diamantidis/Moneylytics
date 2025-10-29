import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# --- Basic keyword-based rules ---
CATEGORY_KEYWORDS = {
    "food & drinks": ["restaurant", "coffee", "breakfast", "lunch", "mcdonald", "burger", "pizza", "cafe", "dining", "meal", "food"],
    "transport": ["uber", "taxi", "bus", "train", "metro", "fuel", "gas", "parking"],
    "groceries": ["supermarket", "grocery", "store", "market", "aldi", "lidl", "smarket", "sklavenitis", "ab vasilopoulos", "butcher"],
    "entertainment": ["cinema", "movie", "music", "theater", "gym", "club"],
    "subscriptions": ["netflix", "disneyxd", "amazon prime", "hbo", "nova", "spotify", "apple music", "icloud"],
    "utilities": ["electric", "water", "internet", "wifi", "phone", "gas bill"],
    "housing": ["rent", "mortgage", "insurance", "property"],
    "shopping": ["amazon", "eshop", "clothes", "zara", "h&m", "shopping", "purchase"],
    "income": ["salary", "payment from", "deposit", "refund", "side salary", "bonus", "tips"],
    "health": ["pharmacy", "doctor", "hospital", "clinic"],
    "other": []
}

def keyword_based_category(description: str) -> str:
    """Assign a category using simple keyword rules."""
    desc = str(description).lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in desc for kw in keywords):
            return category.capitalize()
    return "Other"


def cluster_uncategorized(df: pd.DataFrame, text_column="description", n_clusters=5) -> pd.DataFrame:
    """
    Use KMeans to group uncategorized transactions by description similarity.
    Helps discover new hidden patterns like 'subscriptions' or 'fuel'.
    """
    if df.empty:
        return df

    df = df.copy()  # make an explicit copy to avoid SettingWithCopyWarning
    vectorizer = TfidfVectorizer(stop_words="english", max_features=100)
    X = vectorizer.fit_transform(df[text_column].astype(str))

    kmeans = KMeans(n_clusters=min(n_clusters, len(df)), random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(X)

    return df


def categorize_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect and fill missing or incorrect categories using keyword rules.
    Optionally, cluster unknown items for pattern discovery.
    """
    df = df.copy()

    if "category" not in df.columns:
        df["category"] = None

    df["predicted_category"] = df["category"]

    # Apply keyword-based classification
    mask_missing = df["predicted_category"].isna() | (df["predicted_category"] == "")
    df.loc[mask_missing, "predicted_category"] = df.loc[mask_missing, "description"].apply(keyword_based_category)

    # Cluster 'Other' if enough samples exist
    others = df[df["predicted_category"] == "Other"]
    if len(others) > 10:
        clustered = cluster_uncategorized(others)
        df.loc[others.index, "cluster"] = clustered["cluster"]

    return df