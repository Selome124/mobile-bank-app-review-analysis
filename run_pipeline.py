# scripts/run_pipeline.py
import pandas as pd
import os
import glob
from collections import Counter

def find_data_files():
    """Find available CSV files."""
    print("Searching for data files...")
    
    # Look in common locations
    locations = [
        "data/*.csv",
        "*.csv",
        "../*.csv",
        "outputs/*.csv"
    ]
    
    found_files = []
    for pattern in locations:
        files = glob.glob(pattern)
        for file in files:
            if os.path.getsize(file) > 1000:  # At least 1KB
                found_files.append(file)
    
    # Remove duplicates
    found_files = list(set(found_files))
    
    if found_files:
        print(f"Found {len(found_files)} CSV files:")
        for i, file in enumerate(found_files, 1):
            size_kb = os.path.getsize(file) / 1024
            print(f"{i}. {file} ({size_kb:.1f} KB)")
        return found_files
    return []

def continue_from_existing():
    """Continue analysis from existing output files."""
    print("\nCONTINUING FROM EXISTING FILES...")
    
    # Check what we have
    files_to_check = [
        ("with_themes.csv", "Complete analysis"),
        ("with_keywords.csv", "Keywords extracted"),
        ("with_sentiment.csv", "Sentiment analyzed"),
        ("preprocessed.csv", "Preprocessed data")
    ]
    
    latest_file = None
    for filename, description in files_to_check:
        filepath = f"outputs/{filename}"
        if os.path.exists(filepath):
            latest_file = filepath
            print(f"Found: {filepath} ({description})")
            break
    
    if latest_file:
        df = pd.read_csv(latest_file)
        print(f"Loaded {len(df):,} reviews from {latest_file}")
        return df, os.path.basename(latest_file)
    
    return None, None

def run_simple_sentiment(df):
    """Run basic sentiment analysis."""
    print("\nRUNNING SENTIMENT ANALYSIS...")
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        analyzer = SentimentIntensityAnalyzer()
        
        def get_sentiment(text):
            if pd.isna(text):
                return 0.0
            text_str = str(text)
            if not text_str.strip():
                return 0.0
            return analyzer.polarity_scores(text_str)['compound']
        
        # Find text column
        text_col = None
        for col in ['preprocessed', 'clean_text', 'content', 'review']:
            if col in df.columns:
                text_col = col
                break
        
        if text_col:
            df['sent_score_vader'] = df[text_col].apply(get_sentiment)
            df['sent_label_vader'] = df['sent_score_vader'].apply(
                lambda s: "POSITIVE" if s >= 0.05 else ("NEGATIVE" if s <= -0.05 else "NEUTRAL")
            )
            print("Sentiment analysis complete")
        else:
            print("No text column found for sentiment analysis")
            
    except Exception as e:
        print(f"Sentiment analysis error: {e}")
    
    return df

def run_simple_keywords(df):
    """Run basic keyword extraction."""
    print("\nEXTRACTING KEYWORDS...")
    
    # Find text column
    text_col = None
    for col in ['preprocessed', 'clean_text', 'content', 'review']:
        if col in df.columns:
            text_col = col
            break
    
    if not text_col:
        print("No text column found for keyword extraction")
        df['keywords'] = [[] for _ in range(len(df))]
        return df
    
    try:
        # Combine all text
        all_text = ' '.join(df[text_col].fillna('').astype(str).tolist()).lower()
        
        # Simple word extraction
        words = [word for word in all_text.split() if len(word) > 3]
        word_counts = Counter(words)
        
        # Remove common words
        stopwords = {
            'this', 'that', 'with', 'from', 'have', 'they', 'what', 'when',
            'were', 'your', 'will', 'would', 'could', 'should', 'about',
            'their', 'there', 'which', 'were', 'been', 'some', 'more',
            'very', 'just', 'like', 'only', 'also', 'than', 'then'
        }
        
        # Get top keywords
        filtered_words = {word: count for word, count in word_counts.items() 
                         if word not in stopwords and count > 5}
        
        top_keywords = sorted(filtered_words.items(), key=lambda x: x[1], reverse=True)[:30]
        keywords_list = [word for word, count in top_keywords]
        
        print(f"Found {len(keywords_list)} keywords")
        
        # Assign keywords to reviews
        def find_review_keywords(text):
            text_lower = str(text).lower()
            found = []
            for keyword in keywords_list:
                if keyword in text_lower:
                    found.append(keyword)
                    if len(found) >= 5:  # Max 5 keywords per review
                        break
            return found
        
        df['keywords'] = df[text_col].apply(find_review_keywords)
        
    except Exception as e:
        print(f"Keyword extraction error: {e}")
        df['keywords'] = [[] for _ in range(len(df))]
    
    return df

def run_simple_themes(df):
    """Run basic theme identification."""
    print("\nIDENTIFYING THEMES...")
    
    if 'keywords' not in df.columns:
        print("No keywords found. Skipping theme analysis.")
        df['identified_themes'] = [[] for _ in range(len(df))]
        return df
    
    try:
        # Simple theme categories based on keywords
        theme_categories = {
            'positive_feedback': ['good', 'great', 'best', 'excellent', 'nice', 'love', 'perfect'],
            'negative_feedback': ['bad', 'worst', 'terrible', 'horrible', 'awful', 'poor', 'disappointed'],
            'app_issues': ['problem', 'issue', 'error', 'bug', 'crash', 'fix', 'update'],
            'user_experience': ['easy', 'simple', 'fast', 'slow', 'difficult', 'hard', 'complex'],
            'features': ['feature', 'function', 'option', 'setting', 'menu', 'button'],
            'transactions': ['money', 'transfer', 'payment', 'transaction', 'bank', 'account']
        }
        
        def assign_themes(keywords):
            themes = []
            if isinstance(keywords, list):
                for keyword in keywords:
                    for theme, theme_keywords in theme_categories.items():
                        if any(theme_kw in keyword for theme_kw in theme_keywords):
                            if theme not in themes:
                                themes.append(theme)
            return themes
        
        df['identified_themes'] = df['keywords'].apply(assign_themes)
        print("Theme identification complete")
        
    except Exception as e:
        print(f"Theme analysis error: {e}")
        df['identified_themes'] = [[] for _ in range(len(df))]
    
    return df

def show_results(df):
    """Display analysis results."""
    print("\n" + "="*70)
    print("ANALYSIS RESULTS")
    print("="*70)
    
    print(f"\nTotal Reviews Analyzed: {len(df):,}")
    
    # Sentiment results
    if 'sent_label_vader' in df.columns:
        print("\n--- SENTIMENT DISTRIBUTION ---")
        sentiment_counts = df['sent_label_vader'].value_counts()
        total = len(df)
        
        for sentiment, count in sentiment_counts.items():
            percentage = (count / total) * 100
            bar = "#" * int(percentage / 2)
            print(f"{sentiment:8s}: {count:6,d} ({percentage:5.1f}%) {bar}")
    
    # Keyword results
    if 'keywords' in df.columns:
        print("\n--- KEYWORD ANALYSIS ---")
        
        # Count reviews with keywords
        reviews_with_keywords = df[df['keywords'].apply(lambda x: len(x) > 0)]
        print(f"Reviews with keywords: {len(reviews_with_keywords):,}/{len(df):,} ({len(reviews_with_keywords)/len(df)*100:.1f}%)")
        
        # Get all keywords
        all_keywords = []
        for kw_list in df['keywords']:
            if isinstance(kw_list, list):
                all_keywords.extend(kw_list)
        
        if all_keywords:
            keyword_counts = Counter(all_keywords)
            print(f"Unique keywords found: {len(keyword_counts):,}")
            print("\nTop 10 keywords:")
            for kw, count in keyword_counts.most_common(10):
                print(f"  {kw:20s}: {count:4,d}")
    
    # Theme results
    if 'identified_themes' in df.columns:
        print("\n--- THEME ANALYSIS ---")
        
        # Count reviews with themes
        reviews_with_themes = df[df['identified_themes'].apply(lambda x: len(x) > 0)]
        print(f"Reviews with themes: {len(reviews_with_themes):,}/{len(df):,} ({len(reviews_with_themes)/len(df)*100:.1f}%)")
        
        # Get all themes
        all_themes = []
        for theme_list in df['identified_themes']:
            if isinstance(theme_list, list):
                all_themes.extend(theme_list)
        
        if all_themes:
            theme_counts = Counter(all_themes)
            print(f"Themes identified: {len(theme_counts)}")
            print("\nTheme frequency:")
            for theme, count in theme_counts.most_common():
                percentage = (count / len(df)) * 100
                print(f"  {theme:20s}: {count:4,d} reviews ({percentage:.1f}%)")
    
    print("\n" + "="*70)
    print("FILES SAVED:")
    print("="*70)

def save_results(df, step):
    """Save results at each step."""
    os.makedirs("outputs", exist_ok=True)
    
    if step == "sentiment":
        df.to_csv("outputs/with_sentiment.csv", index=False)
        print("Saved: outputs/with_sentiment.csv")
    elif step == "keywords":
        df.to_csv("outputs/with_keywords.csv", index=False)
        print("Saved: outputs/with_keywords.csv")
    elif step == "themes":
        df.to_csv("outputs/with_themes.csv", index=False)
        print("Saved: outputs/with_themes.csv")
        df.to_csv("outputs/final_results.csv", index=False)
        print("Saved: outputs/final_results.csv")

def run_pipeline():
    """Main pipeline function."""
    print("="*70)
    print("MOBILE BANK APP REVIEW ANALYSIS")
    print("="*70)
    
    # Try to continue from existing files
    df, source = continue_from_existing()
    
    if df is None:
        print("\nNo existing analysis files found.")
        print("Looking for raw data files...")
        
        data_files = find_data_files()
        if not data_files:
            print("\nERROR: No data files found.")
            print("Please place your CSV file in one of these locations:")
            print("1. data/reviews.csv")
            print("2. Current directory (as reviews.csv)")
            print("3. outputs/ folder")
            return
        
        # Use the largest file
        largest_file = max(data_files, key=lambda x: os.path.getsize(x))
        print(f"\nUsing data file: {largest_file}")
        
        df = pd.read_csv(largest_file)
        print(f"Loaded {len(df):,} reviews")
        
        # Basic preprocessing
        print("\nPREPROCESSING DATA...")
        if 'content' in df.columns:
            df['clean_text'] = df['content'].fillna('').astype(str).str.lower()
            df['preprocessed'] = df['clean_text']
        elif 'review' in df.columns:
            df['clean_text'] = df['review'].fillna('').astype(str).str.lower()
            df['preprocessed'] = df['clean_text']
        
        df.to_csv("outputs/preprocessed.csv", index=False)
        print("Saved: outputs/preprocessed.csv")
        source = "preprocessed.csv"
    
    print(f"\nStarting analysis from: {source}")
    
    # Run analysis steps
    if 'sent_label_vader' not in df.columns:
        df = run_simple_sentiment(df)
        save_results(df, "sentiment")
    
    if 'keywords' not in df.columns:
        df = run_simple_keywords(df)
        save_results(df, "keywords")
    
    if 'identified_themes' not in df.columns:
        df = run_simple_themes(df)
        save_results(df, "themes")
    
    # Show results
    show_results(df)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print("\nOutput files available in 'outputs/' folder:")
    for file in os.listdir("outputs"):
        if file.endswith('.csv'):
            size_kb = os.path.getsize(f"outputs/{file}") / 1024
            print(f"  • {file} ({size_kb:.1f} KB)")

if __name__ == "__main__":
    run_pipeline()