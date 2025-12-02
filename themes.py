# scripts/themes.py
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import re
from collections import Counter

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle NumPy data types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def cluster_keywords(keywords, n_clusters=4):
    """
    Cluster keywords into themes using TF-IDF and hierarchical clustering.
    
    Parameters:
    -----------
    keywords : list of strings
        List of keywords to cluster
    n_clusters : int
        Number of clusters/themes to create
        
    Returns:
    --------
    dict: Cluster assignments with cluster ID as key and list of keywords as value
    """
    if len(keywords) <= 1:
        # If only 0 or 1 keyword, return as single cluster
        return {0: keywords} if keywords else {}
    
    # Limit n_clusters to number of keywords
    n_clusters = min(n_clusters, len(keywords))
    
    # Create TF-IDF vectors for keywords
    vec = TfidfVectorizer().fit_transform(keywords)
    
    # Use agglomerative clustering
    clustering = AgglomerativeClustering(n_clusters=n_clusters).fit(vec.toarray())
    labels = clustering.labels_
    
    # Organize keywords by cluster
    clusters = {}
    for k, lab in zip(keywords, labels):
        clusters.setdefault(lab, []).append(k)
    
    return clusters

def create_theme_names(clusters):
    """
    Create descriptive names for themes based on keywords in each cluster.
    
    Parameters:
    -----------
    clusters : dict
        Dictionary of clusters with keywords
        
    Returns:
    --------
    dict: Themes with descriptive names
    """
    themes = {}
    
    for cluster_id, keywords in clusters.items():
        if not keywords:
            continue
        
        # Take first 2-3 keywords to create theme name
        if len(keywords) >= 3:
            name_parts = keywords[:3]
        else:
            name_parts = keywords
        
        # Create theme name
        theme_name = " & ".join(name_parts)
        
        # Convert any numpy types to Python native types for JSON serialization
        theme_name = str(theme_name)
        
        # Store theme information
        themes[theme_name] = {
            'keywords': [str(k) for k in keywords],
            'cluster_id': int(cluster_id),
            'size': int(len(keywords))
        }
    
    return themes

def extract_keywords_for_themes(df, text_col='preprocessed', min_freq=5):
    """
    Extract keywords for theme analysis with better filtering.
    
    Parameters:
    -----------
    df : pandas DataFrame
        Input dataframe
    text_col : str
        Column containing text
    min_freq : int
        Minimum frequency for a word/phrase to be considered
        
    Returns:
    --------
    dict: Keywords for theme analysis
    """
    print("Extracting keywords for theme analysis...")
    
    # Combine all texts
    all_texts = df[text_col].fillna('').astype(str).tolist()
    
    # Extract both single words and bi-grams
    single_words = []
    bigrams = []
    
    for text in all_texts:
        text_lower = text.lower()
        
        # Extract single words (3+ characters)
        words = re.findall(r'\b[a-z]{3,}\b', text_lower)
        single_words.extend(words)
        
        # Extract bi-grams (pairs of words)
        word_tokens = re.findall(r'\b[a-z]{3,}\b', text_lower)
        for i in range(len(word_tokens) - 1):
            bigram = f"{word_tokens[i]} {word_tokens[i+1]}"
            bigrams.append(bigram)
    
    # Count frequencies
    single_word_counts = Counter(single_words)
    bigram_counts = Counter(bigrams)
    
    # Filter out common stopwords
    stopwords = set([
        'the', 'and', 'to', 'a', 'i', 'is', 'in', 'it', 'you', 'of',
        'for', 'on', 'that', 'with', 'are', 'be', 'this', 'have', 'not',
        'but', 'they', 'at', 'what', 'so', 'if', 'my', 'or', 'was', 'as',
        'has', 'an', 'there', 'we', 'all', 'can', 'your', 'will', 'one',
        'just', 'dont', 'like', 'get', 'very', 'good', 'bad', 'app',
        'bank', 'mobile', 'account', 'money', 'transfer', 'login',
        'password', 'user', 'update', 'version', 'phone', 'need',
        'please', 'help', 'customer', 'service', 'use', 'time', 'day',
        'cant', 'would', 'could', 'should', 'make', 'also', 'know',
        'really', 'even', 'still', 'back', 'when', 'where', 'why', 'how',
        'some', 'any', 'only', 'than', 'then', 'them', 'these', 'those',
        'from', 'had', 'has', 'her', 'here', 'him', 'his', 'into'
    ])
    
    # Filter and combine keywords
    keywords = []
    
    # Add single words
    for word, count in single_word_counts.items():
        if count >= min_freq and word not in stopwords and len(word) >= 4:
            keywords.append((word, count, 'word'))
    
    # Add bi-grams
    for bigram, count in bigram_counts.items():
        if count >= min_freq:
            # Check if bigram contains stopwords
            words = bigram.split()
            if not any(word in stopwords for word in words):
                keywords.append((bigram, count, 'bigram'))
    
    # Sort by frequency and get top keywords
    keywords.sort(key=lambda x: x[1], reverse=True)
    top_keywords = [kw[0] for kw in keywords[:100]]  # Top 100
    
    print(f"Extracted {len(top_keywords)} keywords for theme analysis")
    
    # Debug: print top keywords
    print(f"\nTop 20 keywords by frequency:")
    for i, (kw, count, kw_type) in enumerate(keywords[:20], 1):
        print(f"  {i:2d}. {kw:25s} (freq: {count}, type: {kw_type})")
    
    return {"all_reviews": top_keywords}

def create_themes_from_keywords(keyword_dict, n_themes=5):
    """
    Create themes from keywords for each bank/app.
    """
    bank_themes = {}
    
    for bank, keywords in keyword_dict.items():
        if not keywords or len(keywords) < 3:
            print(f"Not enough keywords for {bank} to create themes")
            bank_themes[bank] = {}
            continue
        
        print(f"Creating themes for {bank} with {len(keywords)} keywords...")
        
        # Cluster keywords
        clusters = cluster_keywords(keywords, n_clusters=n_themes)
        
        # Create theme names
        themes = create_theme_names(clusters)
        
        bank_themes[bank] = themes
    
    return bank_themes

def assign_themes_to_reviews(df, bank_themes, text_col='preprocessed'):
    """
    Assign themes to individual reviews based on keyword matching.
    """
    def find_themes_for_review(row):
        bank = "all_reviews"
        text = str(row.get(text_col, '')).lower()
        
        if not text or text == 'nan':
            return []
        
        themes = bank_themes.get(bank, {})
        found_themes = []
        
        for theme_name, theme_info in themes.items():
            keywords = theme_info.get('keywords', [])
            
            # Check if any keyword from this theme appears in the text
            for keyword in keywords:
                # Use word boundaries for matching
                if re.search(r'\b' + re.escape(keyword.lower()) + r'\b', text):
                    found_themes.append(theme_name)
                    break
        
        return found_themes
    
    print("Assigning themes to reviews...")
    df['identified_themes'] = df.apply(find_themes_for_review, axis=1)
    
    # Count reviews with themes
    reviews_with_themes = df[df['identified_themes'].apply(lambda x: len(x) > 0)]
    print(f"Reviews with identified themes: {len(reviews_with_themes)}/{len(df)} ({len(reviews_with_themes)/len(df)*100:.1f}%)")
    
    return df

def print_theme_summary(bank_themes):
    """Print a summary of created themes."""
    print("\n" + "="*60)
    print("THEME ANALYSIS SUMMARY")
    print("="*60)
    
    for bank, themes in bank_themes.items():
        print(f"\nApp: {bank}")
        print("-" * 40)
        
        if not themes:
            print("  No themes created (insufficient keywords)")
            continue
        
        print(f"  Created {len(themes)} themes:")
        
        for i, (theme_name, theme_info) in enumerate(themes.items(), 1):
            keywords = theme_info['keywords']
            size = theme_info['size']
            print(f"\n  Theme {i}: {theme_name}")
            print(f"    Keywords ({size}): {', '.join(keywords[:8])}" + 
                  ("..." if len(keywords) > 8 else ""))

def analyze_sentiment_by_theme(df):
    """
    Analyze sentiment distribution for each theme.
    """
    print("\n" + "="*60)
    print("SENTIMENT ANALYSIS BY THEME")
    print("="*60)
    
    # Flatten themes for analysis
    theme_data = []
    
    for _, row in df.iterrows():
        themes = row.get('identified_themes', [])
        sentiment = row.get('sent_label_vader', 'NEUTRAL')
        sentiment_score = row.get('sent_score_vader', 0)
        
        for theme in themes:
            theme_data.append({
                'theme': theme,
                'sentiment': sentiment,
                'sentiment_score': sentiment_score
            })
    
    if not theme_data:
        print("No theme data to analyze.")
        return
    
    # Convert to DataFrame
    theme_df = pd.DataFrame(theme_data)
    
    # Group by theme and analyze sentiment
    theme_stats = {}
    
    for theme in theme_df['theme'].unique():
        theme_reviews = theme_df[theme_df['theme'] == theme]
        
        # Sentiment counts
        sentiment_counts = theme_reviews['sentiment'].value_counts().to_dict()
        
        # Average sentiment score
        avg_score = theme_reviews['sentiment_score'].mean()
        
        # Number of reviews
        count = len(theme_reviews)
        
        theme_stats[theme] = {
            'count': count,
            'sentiment_distribution': sentiment_counts,
            'avg_sentiment_score': avg_score
        }
    
    # Print results
    print(f"\nFound {len(theme_stats)} themes with sentiment data:")
    
    # Sort by number of reviews
    sorted_themes = sorted(theme_stats.items(), key=lambda x: x[1]['count'], reverse=True)
    
    for theme, stats in sorted_themes[:10]:  # Top 10 themes
        count = stats['count']
        avg_score = stats['avg_sentiment_score']
        sentiment_dist = stats['sentiment_distribution']
        
        print(f"\nTheme: {theme}")
        print(f"  Number of reviews: {count}")
        print(f"  Average sentiment score: {avg_score:.3f}")
        print(f"  Sentiment distribution:")
        
        for sentiment in ['POSITIVE', 'NEUTRAL', 'NEGATIVE']:
            sentiment_count = sentiment_dist.get(sentiment, 0)
            percentage = (sentiment_count / count) * 100 if count > 0 else 0
            print(f"    {sentiment}: {sentiment_count} ({percentage:.1f}%)")
    
    return theme_stats

def main():
    """Main function to run theme analysis."""
    try:
        # Read the data
        print("Loading data from outputs/with_keywords.csv...")
        df = pd.read_csv("outputs/with_keywords.csv")
        
        print(f"Data shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Determine text column to use
        text_col = 'preprocessed'
        if text_col not in df.columns:
            # Try other common text columns
            possible_text_cols = ['clean_text', 'content', 'review', 'text']
            for col in possible_text_cols:
                if col in df.columns:
                    text_col = col
                    print(f"Using text column: '{text_col}'")
                    break
        
        if text_col not in df.columns:
            print("Error: No suitable text column found.")
            return
        
        # Check if sentiment columns exist
        if 'sent_label_vader' not in df.columns:
            print("Warning: Sentiment columns not found. Running sentiment analysis...")
            # You could call your sentiment function here if needed
            pass
        
        # Step 1: Extract keywords for theme analysis
        keyword_dict = extract_keywords_for_themes(df, text_col, min_freq=10)
        
        # Step 2: Create themes from keywords
        print("\nCreating themes from keywords...")
        bank_themes = create_themes_from_keywords(keyword_dict, n_themes=6)
        
        # Step 3: Print theme summary
        print_theme_summary(bank_themes)
        
        # Step 4: Assign themes to reviews
        df = assign_themes_to_reviews(df, bank_themes, text_col)
        
        # Step 5: Add theme statistics
        print("\n" + "="*60)
        print("THEME DISTRIBUTION IN REVIEWS")
        print("="*60)
        
        # Count theme occurrences
        all_themes = []
        for themes_list in df['identified_themes']:
            all_themes.extend(themes_list)
        
        if all_themes:
            theme_counts = Counter(all_themes)
            print(f"\nFound {len(theme_counts)} unique themes")
            print("\nMost common themes across all reviews:")
            
            for theme, count in theme_counts.most_common(15):
                print(f"  {theme}: {count} reviews")
        else:
            print("\nNo themes found in reviews.")
        
        # Step 6: Analyze sentiment by theme
        if 'sent_label_vader' in df.columns:
            theme_stats = analyze_sentiment_by_theme(df)
        else:
            print("\nSkipping sentiment by theme analysis (sentiment data not available)")
        
        # Step 7: Save results
        output_csv = "outputs/with_themes.csv"
        df.to_csv(output_csv, index=False)
        print(f"\nSaved results to: {output_csv}")
        
        # Save themes to JSON (using custom encoder)
        output_json = "outputs/themes.json"
        with open(output_json, "w") as f:
            json.dump(bank_themes, f, indent=2, cls=NumpyEncoder)
        print(f"Saved themes to: {output_json}")
        
        # Save theme statistics to JSON
        if 'sent_label_vader' in df.columns:
            output_stats = "outputs/theme_sentiment_stats.json"
            with open(output_stats, "w") as f:
                json.dump(theme_stats, f, indent=2, cls=NumpyEncoder)
            print(f"Saved theme sentiment statistics to: {output_stats}")
        
        # Print sample reviews with themes
        print("\n" + "="*60)
        print("SAMPLE REVIEWS WITH IDENTIFIED THEMES")
        print("="*60)
        
        # Get reviews that have themes
        reviews_with_themes = df[df['identified_themes'].apply(lambda x: len(x) > 0)]
        
        if len(reviews_with_themes) > 0:
            sample_reviews = reviews_with_themes.head(3)
            
            for i, (idx, row) in enumerate(sample_reviews.iterrows(), 1):
                text_preview = str(row[text_col])[:200] + "..." if len(str(row[text_col])) > 200 else str(row[text_col])
                themes = row['identified_themes']
                sentiment = row.get('sent_label_vader', 'N/A')
                sentiment_score = row.get('sent_score_vader', 'N/A')
                
                print(f"\nSample {i}:")
                print(f"  Text: {text_preview}")
                print(f"  Themes: {', '.join(themes) if themes else 'None'}")
                print(f"  Sentiment: {sentiment} (score: {sentiment_score:.3f})")
        else:
            print("\nNo reviews with identified themes to display.")
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE!")
        print("="*60)
        
        # Final summary
        print(f"\nSUMMARY:")
        print(f"  Total reviews analyzed: {len(df)}")
        print(f"  Reviews with themes identified: {len(reviews_with_themes)} ({len(reviews_with_themes)/len(df)*100:.1f}%)")
        
        if 'sent_label_vader' in df.columns:
            overall_sentiment = df['sent_label_vader'].value_counts()
            print(f"\n  Overall sentiment distribution:")
            for sentiment, count in overall_sentiment.items():
                print(f"    {sentiment}: {count} ({count/len(df)*100:.1f}%)")
        
    except FileNotFoundError:
        print("Error: File 'outputs/with_keywords.csv' not found.")
        print("Please run keyword extraction first.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()