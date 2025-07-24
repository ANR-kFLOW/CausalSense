import argparse
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification, get_linear_schedule_with_warmup


# Define dataset class
class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

def run_filter(args):
    
    test_data = args.base_df
    if 'num_rs' in test_data.columns:
        test_data.loc[test_data['num_rs'] > 1, 'num_rs'] = 1
        
    else:
        test_data['num_rs'] = 0
        test_data.loc[1, 'num_rs'] = 1
    


    print("Length of test_data:", len(test_data))

    # Print distribution of num_rs column for each dataset

    print("\nDistribution of num_rs column for test_data:")
    print(test_data['num_rs'].value_counts(normalize=True))
    
    # Set device
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    cuda_flag = False
    if torch.cuda.is_available():
        cuda_flag = True
    else:
        cuda_flag = False
    # Define parameters
    MAX_LEN = 128
    TRAIN_BATCH_SIZE = 32
    VALID_BATCH_SIZE = 64
    EPOCHS = 10
    LEARNING_RATE = 2e-5

    # Initialize tokenizer and model
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base', cache_dir='data/huggingface/')
    model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2, cache_dir='data/huggingface/')
    model.to(device)


    # Define optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    # Define loss function
    loss_fn = nn.CrossEntropyLoss()
    best_f1_score = 0.0
    #best_model_path = "best_model_st1.pt"
    best_model_path = args.filter_model_path
    #print(best_model_path)


    # Test loop
    test_dataset = CustomDataset(test_data['text'], test_data['num_rs'], tokenizer, MAX_LEN)
    test_loader = DataLoader(test_dataset, batch_size=VALID_BATCH_SIZE, shuffle=False)

    # Load the best model
    best_model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2, cache_dir='data/huggingface/')
    if cuda_flag == True:
        best_model.load_state_dict(torch.load(best_model_path))
    else:
        best_model.load_state_dict(torch.load(best_model_path, map_location=torch.device('cpu')))
    best_model.to(device)

    best_model.eval()
    test_predictions = []
    test_true_labels = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = best_model(input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits
            # Apply the custom threshold
            probs = torch.softmax(logits, dim=1)[:, 1]  # Probability of class 1
            predicted = (probs > args.filter_threshold).long()  # Apply threshold
            test_predictions.extend(predicted.cpu().numpy())
            test_true_labels.extend(labels.cpu().numpy())
            
    # Calculate test accuracy and classification report
    #test_accuracy = accuracy_score(test_true_labels, test_predictions)
    #test_classification_report = classification_report(test_true_labels, test_predictions)

    #print(f"Best Model - Test Accuracy: {test_accuracy:.4f}")
    #print("Best Model - Test Classification Report:")
    #print(test_classification_report)
    t_data = test_data
    predicted_df = test_data
    predicted_df['num_rs'] = test_predictions
    predicted_df.loc[predicted_df['num_rs'] < 1, 'causal_text_w_pairs'] = '[]'

    condition = (predicted_df['num_rs'] > 0) & (t_data['num_rs'] > 0)
    #print(t_data['num_rs'].head(20))
    predicted_df.loc[condition, 'num_rs'] = t_data.loc[condition, 'num_rs'].values
    predicted_df['label'] = 0
    predicted_df.loc[1, 'label'] = 1
    predicted_df.loc[2, 'label'] = 2
    predicted_df.loc[3, 'label'] = 3
    s = '<triplet> reaching <subj> agreed <obj> cause'
    predicted_df['triplets'] = s
    predicted_df.to_csv('causal_outs/predicted_as_causal.csv', index=False)
    predicted_df.loc[predicted_df['num_rs'] > 0].to_csv('causal_outs/only_causal.csv', index=False)
    c =  predicted_df.loc[predicted_df['num_rs'] > 0]
    predicted_df.loc[predicted_df['num_rs'] == 1].to_csv('causal_outs/only_causal_single.csv', index=False)
    c = c.reset_index(drop=True)
    return c




if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Binary Classification with Custom Threshold')
    parser.add_argument('--train_file', type=str, help='Path to the training data file')
    parser.add_argument('--val_file', type=str, help='Path to the validation data file')
    parser.add_argument('--test_file', type=str, help='Path to the test data file')
    parser.add_argument('--filter_threshold', type=float, required=True, help='Threshold for classification')
    parser.add_argument('--filter_model_path', type=str, help='Path to model')
    args , unknown = parser.parse_known_args()

    # Load the dataset

    test_data = pd.read_csv(args.test_file)

    if 'num_rs' in test_data.columns:
        test_data.loc[test_data['num_rs'] > 1, 'num_rs'] = 1
        
    else:
        test_data['num_rs'] = 0
        test_data.loc[1, 'num_rs'] = 1


    print("Length of test_data:", len(test_data))

    # Print distribution of num_rs column for each dataset

    print("\nDistribution of num_rs column for test_data:")
    print(test_data['num_rs'].value_counts(normalize=True))

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cuda_flag = False
    if torch.cuda.is_available():
        cuda_flag = True
    else:
        cuda_flag = False
    # Define parameters
    MAX_LEN = 128
    TRAIN_BATCH_SIZE = 32
    VALID_BATCH_SIZE = 64
    EPOCHS = 10
    LEARNING_RATE = 2e-5

    # Initialize tokenizer and model
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)
    model.to(device)


    # Define optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    # Define loss function
    loss_fn = nn.CrossEntropyLoss()
    best_f1_score = 0.0
    #best_model_path = "best_model_st1.pt"
    best_model_path = args.filter_model_path


    # Test loop
    test_dataset = CustomDataset(test_data['text'], test_data['num_rs'], tokenizer, MAX_LEN)
    test_loader = DataLoader(test_dataset, batch_size=VALID_BATCH_SIZE, shuffle=False)

    # Load the best model
    best_model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2, cache_dir='data/huggingface/')
    
    if cuda_flag == True:
        best_model.load_state_dict(torch.load(best_model_path))
    else:
        best_model.load_state_dict(torch.load(best_model_path, map_location=torch.device('cpu')))
    best_model.to(device)

    best_model.eval()
    test_predictions = []
    test_true_labels = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = best_model(input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits
            # Apply the custom threshold
            probs = torch.softmax(logits, dim=1)[:, 1]  # Probability of class 1
            predicted = (probs > args.filter_threshold).long()  # Apply threshold
            test_predictions.extend(predicted.cpu().numpy())
            test_true_labels.extend(labels.cpu().numpy())
            
    # Calculate test accuracy and classification report
    #test_accuracy = accuracy_score(test_true_labels, test_predictions)
    #test_classification_report = classification_report(test_true_labels, test_predictions)

    #print(f"Best Model - Test Accuracy: {test_accuracy:.4f}")
    #print("Best Model - Test Classification Report:")
    #print(test_classification_report)
    t_data = pd.read_csv(args.test_file)
    predicted_df = test_data
    predicted_df['num_rs'] = test_predictions
    predicted_df.loc[predicted_df['num_rs'] < 1, 'causal_text_w_pairs'] = '[]'

    condition = (predicted_df['num_rs'] > 0) & (t_data['num_rs'] > 0)
    #print(t_data['num_rs'].head(20))
    predicted_df.loc[condition, 'num_rs'] = t_data.loc[condition, 'num_rs'].values
    predicted_df['label'] = 0
    predicted_df.loc[1, 'label'] = 1
    predicted_df.loc[2, 'label'] = 2
    predicted_df.loc[3, 'label'] = 3
    s = '<triplet> reaching <subj> agreed <obj> cause'
    predicted_df['triplets'] = s
    predicted_df.to_csv('causal_outs/predicted_as_causal.csv', index=False)
    predicted_df.loc[predicted_df['num_rs'] > 0].to_csv('causal_outs/only_causal.csv', index=False)
    predicted_df.loc[predicted_df['num_rs'] == 1].to_csv('causal_outs/only_causal_single.csv', index=False)

