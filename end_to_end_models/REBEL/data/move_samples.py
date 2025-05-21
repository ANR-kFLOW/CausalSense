import pandas as pd
import re

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

# Load datasets
df_dev = pd.read_csv('/data/Youss/RE/REBEL/data/news_data_with_cnc/data_with_neg_sample/dev.csv')
df_test = pd.read_csv('/data/Youss/RE/REBEL/data/news_data_with_cnc/data_with_neg_sample/test_joined.csv')

# Define column name
column = 'triplets'

if column not in df_dev.columns or column not in df_test.columns:
    print(f"Error: Column '{column}' not found in one of the datasets.")
else:
    # Extract relations
    df_dev[['subject', 'object', 'relation']] = df_dev[column].apply(lambda x: pd.Series(extract_triplet(x)))
    
    # Define target relations and their counts to move
    target_relations = {"intend": 55, "enable": 70, "prevent": 43}
    df_moved = pd.DataFrame()
    
    for relation, count in target_relations.items():
        to_move = df_dev[df_dev['relation'] == relation].sample(n=min(count // 2, len(df_dev[df_dev['relation'] == relation])), random_state=42)
        df_moved = pd.concat([df_moved, to_move])
        df_dev = df_dev.drop(to_move.index)
    
    # Add moved samples to test set
    df_test = pd.concat([df_test, df_moved])
    df_dev.drop(columns=['subject', 'object', 'relation'], inplace=True)
    df_test.drop(columns=['subject', 'object', 'relation'], inplace=True)
    
    
    # Save updated datasets
    df_dev.to_csv('/data/Youss/RE/REBEL/data/news_data_with_cnc/data_with_neg_sample/updated_dev.csv', index=False)
    df_test.to_csv('/data/Youss/RE/REBEL/data/news_data_with_cnc/data_with_neg_sample/updated_test_joined.csv', index=False)
    
    print("Updated datasets saved.")
    print("Moved samples:")
    print(df_moved['relation'].value_counts())
