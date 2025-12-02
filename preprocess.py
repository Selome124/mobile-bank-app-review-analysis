# scripts/preprocess.py (NLTK fixed for Python 3.12 / 3.13)

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# FIX: download missing punkt_tab
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)   # <-- REQUIRED FIX
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

STOPWORDS = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess_text_nltk(text):
    text = clean_text(text)

    # SAFER TOKENIZER (does NOT depend on punkt_tab)
    tokens = nltk.tokenize.RegexpTokenizer(r"\w+").tokenize(text)

    lemmas = [
        lemmatizer.lemmatize(t)
        for t in tokens
        if t not in STOPWORDS and len(t) > 1
    ]
    return " ".join(lemmas)

def detect_text_column(df):
    possible = ["review_text", "text", "review", "comment", "content", "body"]
    for col in df.columns:
        if col.lower() in possible:
            return col
    return None

def load_and_preprocess(path):
    print(f"Loading dataset from: {path}")
    df = pd.read_csv(path)
    print("\nCSV Columns Found:", list(df.columns), "\n")

    text_col = detect_text_column(df)
    if text_col is None:
        raise ValueError("❌ No text column found in the CSV.")

    print(f"Using text column: {text_col}")

    df[text_col] = df[text_col].fillna("").astype(str)
    df["clean_text"] = df[text_col].apply(clean_text)
    df["preprocessed"] = df["clean_text"].apply(preprocess_text_nltk)
    return df


if __name__ == "__main__":
    path = r"c:\Users\deres\OneDrive\Desktop\week2\mobile-bank-app-review-analysis\bank-reviews-task2\data\reviews.csv"

    df = load_and_preprocess(path)

    import os
    if not os.path.exists("outputs"):
        os.makedirs("outputs")

    output_path = "outputs/preprocessed.csv"
    df.to_csv(output_path, index=False)

    print(f"Saved {output_path} — rows:", len(df))
