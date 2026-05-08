import pandas as pd
from itertools import combinations
import random
import re 

import pandas as pd
from itertools import combinations
import random

def extract_triplet_components(triplet):
    """Extract subject, object, and label from a triplet using regular expressions."""
    try:
        pattern = r"<triplet>\s*(.*?)\s*<subj>\s*(.*?)\s*<obj>\s*(.*?)\s*(\S+)$"
        match = re.search(pattern, triplet)
        
        if match:
            subj = match.group(1).strip()
            obj = match.group(2).strip()
            label = match.group(4).strip()
            return subj, obj, label
        else:
            return None, None, None
    except IndexError:
        return None, None, None

def generate_negative_samples(df):
    print(len(df))
    negative_samples = []
    subset_size = len(df) // 5
    sampled_df = df.sample(n=subset_size, random_state=42)
    

    for (idx1, row1), (idx2, row2) in combinations(sampled_df.iterrows(), 2):
        subj1, obj1, label1 = extract_triplet_components(row1["triplets"])
       
        subj2, obj2, label2 = extract_triplet_components(row2["triplets"])
        
        
        if not all([subj1, obj1, label1, subj2, obj2, label2]):
            print('missing component')

            continue
        
       
        if label1 != label2:
           
           
            new_triplet1 = f"<triplet>{subj2} <subj> {obj1} <obj> no_relation"
            new_triplet2 = f"<triplet> {subj1} <subj>  {obj2}  <obj> no_relation"
            
           
            text1 = " ".join(new_triplet1.replace("<triplet>", "").replace("<subj>", "").replace("<obj>", "").split()[:-1])
            text2 = " ".join(new_triplet2.replace("<triplet>", "").replace("<subj>", "").replace("<obj>", "").split()[:-1])
            
            negative_samples.extend([
                {"context": text1, "triplets": new_triplet1},
                {"context": text2, "triplets": new_triplet2}
                
            ])
        if len(negative_samples) >= len(sampled_df):
           break
        
    
    df = pd.concat([df, pd.DataFrame(negative_samples)], ignore_index=True)
    print(len(df))
    return df

# Example usage
# df = pd.read_csv("your_dataset.csv")
# df = generate_negative_samples(df)
# df.to_csv("updated_dataset.csv", index=False)

# Example usage
df = pd.read_csv("/data/Youss/RE/REBEL/data/news_data_with_cnc/validation_augmented.csv")
df = generate_negative_samples(df)
df.to_csv("/data/Youss/RE/REBEL/data/news_data_with_cnc/data_with_neg_sample/dev.csv", index=False)
