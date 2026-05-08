import os
import pandas as pd
import re

# Define the directory containing the CSV files
directory = "/data/Youss/RE/REBEL/data/news_data_with_cnc/data_with_neg_sample"

# Regular expression patterns to clean spaces before the label and ensure space after <triplet>
def clean_triplet(triplet_str):
    triplet_str = re.sub(r"(\<obj\>)(\s+)", r"\1 ", triplet_str)  # Ensure only one space after <obj>
    triplet_str = re.sub(r"(\<triplet\>)(\S)", r"\1 \2", triplet_str)  # Ensure space after <triplet>
    return triplet_str

# Iterate through all CSV files in the directory
for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        file_path = os.path.join(directory, filename)
        
        # Load the CSV file
        df = pd.read_csv(file_path)
        
        # Check if 'triplets' column exists
        if 'triplets' in df.columns:
            df['triplets'] = df['triplets'].astype(str).apply(clean_triplet)
            
            # Save the cleaned data back to the same file
            df.to_csv(file_path, index=False)
            print(f"Updated: {file_path}")
        else:
            print(f"Skipped (no 'triplets' column): {file_path}")
