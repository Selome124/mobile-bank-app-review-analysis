# scripts/keywords.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from collections import Counter
import re

def check_columns(df):
    """Print available columns to help debug."""
    print("Available columns in DataFrame:")
    for col in df.columns:
        print(f"  - '{col}'")

def extract_keywords_single_group(df, text_col='preprocessed', ngram_range=(1,3), top_n=20):
    """
    Extract keywords when there's only one group of documents.
    """
    print("Only one group found. Analyzing all reviews together...")
    
    # Combine all texts
    all_texts = df[text_col].fillna('').astype(str).tolist()
    all_text_combined = " ".join(all_texts)
    
    # Simple word frequency analysis as fallback
    words = re.findall(r'\b\w+\b', all_text_combined.lower())
    word_freq = Counter(words)
    
    # Remove common stopwords
    common_stopwords = set(['the', 'and', 'to', 'a', 'i', 'is', 'in', 'it', 'you', 'of', 
                           'for', 'on', 'that', 'with', 'are', 'be', 'this', 'have', 'not',
                           'but', 'they', 'at', 'what', 'so', 'if', 'my', 'or', 'was', 'as',
                           'has', 'an', 'there', 'we', 'all', 'can', 'your', 'will', 'one',
                           'just', 'dont', 'like', 'get', 'very', 'good', 'bad', 'app',
                           'bank', 'mobile', 'account', 'money', 'transfer', 'login',
                           'password', 'user', 'update', 'version', 'phone', 'need',
                           'please', 'help', 'customer', 'service', 'use', 'time', 'day'])
    
    filtered_words = {word: count for word, count in word_freq.items() 
                     if word not in common_stopwords and len(word) > 2 and count > 2}
    
    # Get most common words
    top_words = sorted(filtered_words.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    return {
        "all_reviews": {
            'keywords': [word for word, count in top_words],
            'scores': [count for word, count in top_words],
            'method': 'frequency_analysis'
        }
    }, "all_reviews"

def extract_keywords_multiple_groups(df, text_col='preprocessed', group_col='app_name', 
                                    ngram_range=(1,3), top_n=20):
    """
    Extract keywords when there are multiple groups.
    """
    print(f"Found multiple groups in column '{group_col}'. Using TF-IDF...")
    
    groups = df[group_col].unique()
    print(f"Number of groups: {len(groups)}")
    
    # Concatenate all texts for each group
    group_docs = {}
    for g in groups:
        group_texts = df[df[group_col] == g][text_col].fillna('').astype(str).tolist()
        group_docs[g] = " ".join(group_texts)
    
    group_names = list(group_docs.keys())
    docs = [group_docs[g] for g in group_names]
    
    # Adjust TF-IDF parameters based on number of documents
    min_df_val = max(1, len(docs) // 10)  # Adjust min_df based on number of documents
    max_df_val = 0.95  # More lenient max_df
    
    # Apply TF-IDF
    vect = TfidfVectorizer(
        ngram_range=ngram_range, 
        min_df=min_df_val, 
        max_df=max_df_val, 
        stop_words='english'
    )
    
    tfidf = vect.fit_transform(docs)
    feature_names = np.array(vect.get_feature_names_out())
    
    # Extract top keywords for each group
    group_keywords = {}
    for i, group in enumerate(group_names):
        row = tfidf[i].toarray().flatten()
        
        # Check if we have valid TF-IDF scores
        if row.sum() > 0:
            # Get top N keywords
            top_idx = row.argsort()[-top_n:][::-1]
            keywords = feature_names[top_idx].tolist()
            scores = row[top_idx].tolist()
        else:
            # Fallback to frequency analysis for this group
            print(f"Warning: No valid TF-IDF scores for group '{group}'. Using frequency analysis.")
            group_text = group_docs[group]
            words = re.findall(r'\b\w+\b', group_text.lower())
            word_freq = Counter(words)
            
            # Remove common stopwords
            common_stopwords = set(['the', 'and', 'to', 'a', 'i', 'is', 'in', 'it', 'you', 'of'])
            filtered_words = {word: count for word, count in word_freq.items() 
                            if word not in common_stopwords and len(word) > 2 and count > 1}
            top_words = sorted(filtered_words.items(), key=lambda x: x[1], reverse=True)[:top_n]
            keywords = [word for word, count in top_words]
            scores = [count for word, count in top_words]
        
        group_keywords[group] = {
            'keywords': keywords,
            'scores': scores,
            'method': 'tfidf'
        }
    
    return group_keywords, group_col

def extract_tfidf_keywords(df, text_col='preprocessed', group_col=None, ngram_range=(1,3), top_n=20):
    """
    Main function to extract keywords with proper handling for single/multiple groups.
    """
    # If no group column specified, check for common column names
    if group_col is None:
        possible_group_cols = ['bank', 'app_name', 'bank_name', 'app', 'name', 'company', 'score']
        for col in possible_group_cols:
            if col in df.columns:
                group_col = col
                print(f"Auto-detected group column: '{group_col}'")
                break
    
    if group_col is None or group_col not in df.columns:
        print("Warning: No group column found. Analyzing all text as one group.")
        return extract_keywords_single_group(df, text_col, ngram_range, top_n)
    
    # Check if we have multiple groups
    unique_groups = df[group_col].nunique()
    
    if unique_groups <= 1:
        # Only one group, use frequency analysis
        return extract_keywords_single_group(df, text_col, ngram_range, top_n)
    else:
        # Multiple groups, use TF-IDF
        return extract_keywords_multiple_groups(df, text_col, group_col, ngram_range, top_n)

def attach_keywords_to_reviews(df, group_keywords, group_col, text_col='preprocessed', max_per_review=5):
    """
    Attach relevant keywords to each review.
    """
    def find_keywords(row):
        ks = []
        # Get the group for this row
        if group_col == "all_reviews":
            group_val = "all_reviews"
        else:
            group_val = row.get(group_col, None)
        
        if group_val is None or group_val not in group_keywords:
            return ks
        
        text = str(row.get(text_col, '')).lower()
        if not text or text == 'nan':
            return ks
        
        # Get keywords for this group
        keywords_info = group_keywords.get(group_val, {})
        keywords_list = keywords_info.get('keywords', [])
        
        # Find keywords that appear in the text
        for k in keywords_list:
            # Check if keyword appears in text (case-insensitive)
            if re.search(r'\b' + re.escape(k.lower()) + r'\b', text):
                ks.append(k)
                if len(ks) >= max_per_review:
                    break
        
        return ks
    
    df['keywords'] = df.apply(find_keywords, axis=1)
    return df

def print_keyword_summary(group_keywords, group_col):
    """Print a summary of extracted keywords."""
    print("\n" + "="*50)
    print("KEYWORD EXTRACTION SUMMARY")
    print("="*50)
    
    for group, info in group_keywords.items():
        keywords = info['keywords']
        scores = info['scores']
        method = info.get('method', 'unknown')
        
        print(f"\n{group_col.upper()}: {group} (Method: {method})")
        print("-" * 40)
        print(f"Top {len(keywords)} keywords:")
        
        for i, (keyword, score) in enumerate(zip(keywords, scores), 1):
            if method == 'frequency_analysis':
                print(f"  {i:2d}. {keyword:20s} (frequency: {int(score)})")
            else:
                print(f"  {i:2d}. {keyword:20s} (TF-IDF score: {score:.4f})")

def main():
    """Main function to run keyword extraction."""
    try:
        # Read the data
        print("Loading data from outputs/with_sentiment.csv...")
        df = pd.read_csv("outputs/with_sentiment.csv")
        
        # Check available columns
        check_columns(df)
        
        # Determine text column to use
        text_col = None
        possible_text_cols = ['preprocessed', 'clean_text', 'review', 'text', 'content']
        for col in possible_text_cols:
            if col in df.columns:
                text_col = col
                print(f"\nUsing text column: '{text_col}'")
                break
        
        if text_col is None:
            print("Error: No suitable text column found.")
            return
        
        # Extract keywords
        print("\nExtracting keywords...")
        group_keywords, group_col = extract_tfidf_keywords(
            df, 
            text_col=text_col,
            group_col=None,  # Auto-detect
            ngram_range=(1, 3),
            top_n=20
        )
        
        # Print summary
        print_keyword_summary(group_keywords, group_col)
        
        # Attach keywords to reviews
        print("\nAttaching keywords to individual reviews...")
        df = attach_keywords_to_reviews(df, group_keywords, group_col, text_col=text_col)
        
        # Calculate keyword statistics
        print("\n" + "="*50)
        print("KEYWORD DISTRIBUTION IN REVIEWS")
        print("="*50)
        
        # Count reviews with/without keywords
        reviews_with_keywords = df[df['keywords'].apply(lambda x: len(x) > 0)]
        reviews_without_keywords = df[df['keywords'].apply(lambda x: len(x) == 0)]
        
        print(f"Total reviews: {len(df)}")
        print(f"Reviews with keywords: {len(reviews_with_keywords)} ({len(reviews_with_keywords)/len(df)*100:.1f}%)")
        print(f"Reviews without keywords: {len(reviews_without_keywords)} ({len(reviews_without_keywords)/len(df)*100:.1f}%)")
        
        # Get all keywords and their frequencies
        all_keywords = []
        for keywords_list in df['keywords']:
            all_keywords.extend(keywords_list)
        
        if all_keywords:
            keyword_freq = Counter(all_keywords)
            print(f"\nMost common keywords across all reviews:")
            for keyword, freq in keyword_freq.most_common(10):
                print(f"  {keyword}: {freq} occurrences")
        
        # Save results
        output_file = "outputs/with_keywords.csv"
        df.to_csv(output_file, index=False)
        print(f"\nSaved results to: {output_file}")
        
        # Print some examples
        print("\n" + "="*50)
        print("SAMPLE REVIEWS WITH KEYWORDS")
        print("="*50)
        
        # Get reviews that have keywords
        reviews_with_keywords_sample = reviews_with_keywords.head(5)
        
        if len(reviews_with_keywords_sample) > 0:
            for i, (idx, row) in enumerate(reviews_with_keywords_sample.iterrows(), 1):
                text_preview = str(row[text_col])[:150] + "..." if len(str(row[text_col])) > 150 else str(row[text_col])
                keywords = row['keywords']
                sentiment = row.get('sent_label_vader', 'N/A')
                sentiment_score = row.get('sent_score_vader', 'N/A')
                
                print(f"\nSample {i}:")
                print(f"  Text preview: {text_preview}")
                print(f"  Keywords: {keywords}")
                print(f"  Sentiment: {sentiment} (score: {sentiment_score})")
        else:
            print("No reviews with keywords found to display as samples.")
        
    except FileNotFoundError:
        print("Error: File 'outputs/with_sentiment.csv' not found.")
        print("Please run sentiment analysis first.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()