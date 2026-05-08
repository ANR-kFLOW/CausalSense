import os
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
from torch.nn.functional import cosine_similarity

# Define file paths
data_path = "/data/Youss/RE/REBEL/data/news_data_with_cnc/data_with_neg_sample"
train_file = os.path.join(data_path, "train.csv")
dev_file = os.path.join(data_path, "dev.csv")
#test_file = os.path.join(data_path, "test.csv")

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)

# Load datasets
def load_data(file_path):
    df = pd.read_csv(file_path)
    if "context" not in df.columns:
        raise ValueError(f"Column 'context' not found in {file_path}")
    return df

# Remove duplicates from each file
def remove_duplicates(df, file_path):
    before_count = len(df)
    df = df.drop_duplicates()
    after_count = len(df)
    df.to_csv(file_path, index=False)
    print(f"Removed {before_count - after_count} duplicates from {file_path}")
    return df

# Check for similar contexts between test/dev and train

def filter_similar_samples(train_df, compare_df, file_path, threshold=0.95):
    train_embeddings = torch.cat([get_embedding(text) for text in train_df["context"].tolist()])
    indices_to_remove = []
    
    for i, text in enumerate(compare_df["context"].tolist()):
        compare_embedding = get_embedding(text)
        similarities = cosine_similarity(compare_embedding, train_embeddings)
        max_similarity = similarities.max().item()
        print(max_similarity)
        
        if max_similarity >= threshold:
            print(f"Removing context from {file_path}:\n{text}\n")
            indices_to_remove.append(i)
    
    compare_df = compare_df.drop(indices_to_remove)
    compare_df.to_csv(file_path, index=False)
    print(f"Removed {len(indices_to_remove)} similar samples from {file_path}")
    return compare_df

# Process files
train_df = load_data(train_file)
dev_df = load_data(dev_file)
#test_df = load_data(test_file)

#train_df = remove_duplicates(train_df, train_file)
dev_df = remove_duplicates(dev_df, dev_file)
#test_df = remove_duplicates(test_df, test_file)

dev_df = filter_similar_samples(train_df, dev_df, dev_file)
#test_df = filter_similar_samples(train_df, test_df, test_file)
