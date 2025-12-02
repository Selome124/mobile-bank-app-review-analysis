# scripts/sentiment_vader_only.py
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

def get_vader_sentiment(text):
    """
    Get VADER sentiment score for a text.
    Handles NaN values and converts to string.
    """
    # Handle NaN, None, or non-string values
    if pd.isna(text):
        return 0.0  # Return neutral sentiment for missing values
    
    # Convert to string if it isn't already
    text = str(text)
    
    # Get sentiment scores
    v = analyzer.polarity_scores(text)
    
    # VADER gives compound between -1 (very negative) and +1 (very positive)
    return v['compound']

def label_from_score(score, positive_thresh=0.05, negative_thresh=-0.05):
    """
    Convert sentiment score to label.
    """
    if score >= positive_thresh:
        return "POSITIVE"
    elif score <= negative_thresh:
        return "NEGATIVE"
    else:
        return "NEUTRAL"

def compute_vader_sentiment(df, text_column='clean_text'):
    """
    Compute VADER sentiment for a dataframe.
    
    Parameters:
    -----------
    df : pandas DataFrame
        Input dataframe with text column
    text_column : str
        Name of the column containing text to analyze
        
    Returns:
    --------
    pandas DataFrame with added sentiment columns
    """
    print(f"Running VADER sentiment analysis on column: '{text_column}'")
    
    # Check if the text column exists
    if text_column not in df.columns:
        print(f"Error: Column '{text_column}' not found in dataframe.")
        print(f"Available columns: {list(df.columns)}")
        return df
    
    # Apply VADER sentiment analysis
    df['sent_score_vader'] = df[text_column].apply(get_vader_sentiment)
    
    # Create labels from scores
    df['sent_label_vader'] = df['sent_score_vader'].apply(label_from_score)
    
    # Print summary statistics
    print("\n=== VADER Sentiment Analysis Summary ===")
    print(f"Total reviews analyzed: {len(df)}")
    print(f"Positive reviews: {len(df[df['sent_label_vader'] == 'POSITIVE'])} ({len(df[df['sent_label_vader'] == 'POSITIVE'])/len(df)*100:.1f}%)")
    print(f"Neutral reviews: {len(df[df['sent_label_vader'] == 'NEUTRAL'])} ({len(df[df['sent_label_vader'] == 'NEUTRAL'])/len(df)*100:.1f}%)")
    print(f"Negative reviews: {len(df[df['sent_label_vader'] == 'NEGATIVE'])} ({len(df[df['sent_label_vader'] == 'NEGATIVE'])/len(df)*100:.1f}%)")
    
    # Print average sentiment score
    avg_score = df['sent_score_vader'].mean()
    print(f"Average sentiment score: {avg_score:.3f}")
    
    # Show some examples
    print("\n=== Sample Reviews with Sentiment ===")
    sample_size = min(5, len(df))
    for i in range(sample_size):
        text = str(df.iloc[i][text_column])[:100] + "..." if len(str(df.iloc[i][text_column])) > 100 else str(df.iloc[i][text_column])
        score = df.iloc[i]['sent_score_vader']
        label = df.iloc[i]['sent_label_vader']
        print(f"Review {i+1}:")
        print(f"  Text: {text}")
        print(f"  Score: {score:.3f} | Label: {label}")
        print()
    
    return df

def main():
    """
    Main function to run sentiment analysis.
    """
    try:
        # Read the preprocessed data
        print("Loading data from outputs/preprocessed.csv...")
        df = pd.read_csv("outputs/preprocessed.csv")
        
        # Check which text column to use
        print(f"Data loaded. Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Try different possible text columns
        text_columns_to_try = ['clean_text', 'preprocessed', 'review', 'text', 'content']
        text_column = None
        
        for col in text_columns_to_try:
            if col in df.columns:
                text_column = col
                print(f"Using text column: '{text_column}'")
                break
        
        if text_column is None and len(df.columns) > 0:
            # Use the first non-id column
            for col in df.columns:
                if 'id' not in col.lower() and 'date' not in col.lower():
                    text_column = col
                    print(f"No standard text column found. Using: '{text_column}'")
                    break
        
        if text_column is None:
            print("Error: Could not find a suitable text column.")
            return
        
        # Run sentiment analysis
        df = compute_vader_sentiment(df, text_column)
        
        # Save results
        output_file = "outputs/with_sentiment_vader.csv"
        df.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")
        
    except FileNotFoundError:
        print("Error: File 'outputs/preprocessed.csv' not found.")
        print("Please make sure the file exists or update the file path.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()