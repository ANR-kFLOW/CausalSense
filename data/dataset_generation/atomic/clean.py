import pandas as pd

# Load the datasets
files = ["test_checked.csv", "dev_checked.csv", "train_checked.csv"]
data = {file: pd.read_csv(file) for file in files}

# Print initial lengths
print("Initial lengths:")
for file, df in data.items():
    print(f"{file}: {len(df)} rows")

# Step 1: Remove duplicates within each file
data = {file: df.drop_duplicates() for file, df in data.items()}

# Step 2: Remove redundancies between files (from test set if found in train or dev)
test_df = data["test_checked.csv"]
dev_df = data["dev_checked.csv"]
train_df = data["train_checked.csv"]

# Concatenating train and dev for reference
combined_train_dev = pd.concat([train_df, dev_df]).drop_duplicates()

# Filtering test set to remove rows found in train or dev
test_df_cleaned = test_df[~test_df.apply(tuple, axis=1).isin(combined_train_dev.apply(tuple, axis=1))]

# Updating the test data
data["test_checked.csv"] = test_df_cleaned

# Print final lengths
print("\nFinal lengths:")
for file, df in data.items():
    print(f"{file}: {len(df)} rows")

# Save the cleaned data
for file, df in data.items():
    df.to_csv(f"cleaned_{file}", index=False)
