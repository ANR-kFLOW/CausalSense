import os
import pandas as pd
import re

# Define file paths
train_path = "/data/Youss/RE/REBEL/data/news_data_with_cnc/data_with_neg_sample/combined_dataset.csv"
#val_path = "/data/Youss/RE/REBEL/data/news_data_with_cnc/data_with_neg_sample/updated_dev.csv"
#test_path="/data/Youss/RE/REBEL/data/news_data_with_cnc/data_with_neg_sample/updated_test_joined.csv"

# Function to extract triplets
def extract_triplet(triplet_text):
    """
    Extracts subject, object, and relation using regular expressions.

    Args:
        triplet_text (str): The triplet annotation containing <triplet>, <subj>, and <obj> tags.

    Returns:
        tuple: (subject, object, relation) extracted from the triplet text.
    """
    pattern = r"<triplet>(.*?)<subj>(.*?)<obj>(.*)"
    match = re.match(pattern, triplet_text.strip()) if isinstance(triplet_text, str) else None

    if match:
        subject = match.group(1).strip()  # Extract text between <triplet> and <subj>
        object_ = match.group(2).strip()  # Extract text between <subj> and <obj>
        relation = match.group(3).strip()  # Extract text after <obj>
        return subject, object_, relation
    else:
        return None, None, "no_relation"  # Default to "no_relation" if extraction fails

# Function to process dataset
def process_dataset(file_path, output_path):
    if not os.path.exists(file_path):
        print(f"Error: File not found - {file_path}")
        return
    
    df = pd.read_csv(file_path)

    if 'triplets' not in df.columns or 'context' not in df.columns:
        print(f"Error: Expected columns not found in {file_path}")
        return
    
    # Extract triplets
    df[['subject', 'object', 'relation']] = df['triplets'].apply(lambda x: pd.Series(extract_triplet(x)))
    df['text']=df['context']

    # Keep only the required columns
    df = df[['text', 'subject', 'object', 'relation']]

    # Save the extracted triplets
    df.to_csv(output_path, index=False)
    print(f"Extracted triplets saved to: {output_path}")

# Output file paths
train_output_path = "/data/Youss/RE/Bert_based_classification/combined.csv"
#val_output_path = "/data/Youss/RE/Bert_based_classification/dev.csv"
#test_output_path="/data/Youss/RE/Bert_based_classification/test.csv"

# Process both datasets
process_dataset(train_path, train_output_path)
#process_dataset(val_path, val_output_path)
#
#process_dataset(test_path, test_output_path)

