import pandas as pd
import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaTokenizer, RobertaPreTrainedModel
import numpy as np

class MultiHeadRoBERTa(RobertaPreTrainedModel):
    def __init__(self, config, num_relation_types=5, num_bio_labels=5):
        super().__init__(config)
        self.roberta = RobertaModel(config)

        # Relation existence classification (binary)
        self.relation_head = nn.Linear(config.hidden_size, 2)

        # Relation type classification (multiclass)
        self.relation_type_head = nn.Linear(config.hidden_size, num_relation_types)

        # Span detection head (BIO tagging)
        self.token_classification_head = nn.Linear(config.hidden_size, num_bio_labels)

        # Loss functions
        self.loss_fn_relation = nn.CrossEntropyLoss()
        self.loss_fn_type = nn.CrossEntropyLoss()
        self.loss_fn_bio = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, labels_relation=None, labels_type=None, labels_bio=None):
        device = input_ids.device

        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        pooled_output = outputs.pooler_output

        # Compute relation existence
        relation_logits = self.relation_head(pooled_output)
        relation_probs = torch.softmax(relation_logits, dim=-1)
        relation_preds = torch.argmax(relation_probs, dim=-1)

        loss = 0
        type_logits = torch.zeros((input_ids.shape[0], 5), device=device)  # Default "no_relation"

        span_logits = torch.zeros_like(sequence_output[:, :, :5])  # Default BIO "O"

        # If predicted relation exists
        for i in range(input_ids.shape[0]):
            if relation_preds[i] == 1:
                type_logits[i] = self.relation_type_head(pooled_output[i].unsqueeze(0))
                span_logits[i] = self.token_classification_head(sequence_output[i].unsqueeze(0))

        if labels_relation is not None:
            labels_relation = labels_relation.to(device)
            loss += self.loss_fn_relation(relation_logits, labels_relation)

            if labels_type is not None and relation_preds.sum() > 0:
                labels_type = labels_type.to(device)
                loss += self.loss_fn_type(type_logits.view(-1, type_logits.shape[-1]), labels_type.view(-1))

            if labels_bio is not None and relation_preds.sum() > 0:
                labels_bio = labels_bio.to(device)
                loss += self.loss_fn_bio(span_logits.view(-1, span_logits.shape[-1]), labels_bio.view(-1))

        return {
            "loss": loss,
            "relation_logits": relation_logits,
            "relation_preds": relation_preds,
            "relation_type_logits": type_logits,
            "span_logits": span_logits
        }


from torch.utils.data import Dataset

label_map_bio = {"O": 0, "B-SUBJ": 1, "I-SUBJ": 2, "B-OBJ": 3, "I-OBJ": 4}
label_map_relation = {"enable": 0, "cause": 1, "intend": 2, "prevent": 3,"no_relation":4}


class RelationDataset(Dataset):
    def __init__(self, texts, subjects, objects, relation_exists, relation_types, tokenizer, max_len=128):
        self.texts = texts
        self.subjects = subjects
        self.objects = objects
        self.relation_exists = relation_exists
        self.relation_types = relation_types
        self.tokenizer = tokenizer
        self.max_len = max_len

    def tokenize_and_align_labels(self, text, subject, object):
        """
        Tokenizes text and aligns BIO labels manually for a slow tokenizer.
        """
        # Tokenize text with truncation & padding
        encoded = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        input_ids = encoded["input_ids"].squeeze().tolist()
        attention_mask = encoded["attention_mask"].squeeze().tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)

        labels = ["O"] * len(tokens)  # Initialize all tokens with "O"

        # Tokenize subject and object separately
        subject_tokens = self.tokenizer.tokenize(subject)
        object_tokens = self.tokenizer.tokenize(object)

        # Align subject
        for i in range(len(tokens) - len(subject_tokens) + 1):
            if tokens[i:i + len(subject_tokens)] == subject_tokens:
                labels[i] = "B-SUBJ"
                for j in range(1, len(subject_tokens)):
                    labels[i + j] = "I-SUBJ"

        # Align object
        for i in range(len(tokens) - len(object_tokens) + 1):
            if tokens[i:i + len(object_tokens)] == object_tokens:
                labels[i] = "B-OBJ"
                for j in range(1, len(object_tokens)):
                    labels[i + j] = "I-OBJ"

        # Convert labels to numeric and ensure correct length
        numeric_labels = [label_map_bio[label] for label in labels]
        numeric_labels = numeric_labels[:self.max_len]  # Truncate
        numeric_labels += [0] * (self.max_len - len(numeric_labels))  # Pad

        return input_ids, attention_mask, numeric_labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text, subject, object = self.texts[idx], self.subjects[idx], self.objects[idx]

        input_ids, attention_mask, numeric_labels = self.tokenize_and_align_labels(text, subject, object)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels_relation": torch.tensor(self.relation_exists[idx], dtype=torch.long),
            "labels_type": torch.tensor(label_map_relation[self.relation_types[idx]], dtype=torch.long) if
            self.relation_exists[idx] == 1 else torch.tensor(0, dtype=torch.long),
            "labels_bio": torch.tensor(numeric_labels, dtype=torch.long)
        }


# Example dataset
# texts = ["Barack Obama was born in Honolulu.", "Elon Musk founded SpaceX.","Barack Obama was born in Honolulu.", "Elon Musk founded SpaceX."]
# subjects = ["Barack Obama", "Elon Musk","Barack Obama", "Elon Musk"]
# objects = ["Honolulu", "SpaceX", "Honolulu", "SpaceX"]
# relation_exists = [1, 1,0,0]  # Both relations exist
# relation_types = ["cause", "enable","prevent", "no_relation"]
#

# training_args = TrainingArguments(
#     output_dir="./relation_extraction",
#     eval_strategy="no",  # No evaluation dataset yet
#     learning_rate=2e-5,
#     per_device_train_batch_size=8,
#     per_device_eval_batch_size=8,
#     num_train_epochs=3,
#     weight_decay=0.01,
#     logging_dir="./logs",
#     logging_steps=10,
# )
#
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=dataset,
# )
#
# trainer.train()

import torch
import torch.nn as nn
from transformers import RobertaTokenizer, TrainingArguments, Trainer
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import accuracy_score

# Load tokenizer
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

# Expanded dataset
texts = [
    "Barack Obama was born in Honolulu.", "Elon Musk founded SpaceX.", "Jeff Bezos founded Amazon.",
    "Marie Curie discovered radium.", "Albert Einstein developed the theory of relativity.",
    "Steve Jobs co-founded Apple.", "Mark Zuckerberg created Facebook.",
    "Nikola Tesla invented the alternating current.", "Isaac Newton formulated the laws of motion.",
    "Ada Lovelace contributed to early computing."
]
subjects = [
    "Barack Obama", "Elon Musk", "Jeff Bezos", "Marie Curie", "Albert Einstein",
    "Steve Jobs", "Mark Zuckerberg", "Nikola Tesla", "Isaac Newton", "Ada Lovelace"
]
objects = [
    "Honolulu", "SpaceX", "Amazon", "radium", "theory of relativity",
    "Apple", "Facebook", "alternating current", "laws of motion", "early computing"
]
relation_exists = [1, 0, 1, 1, 1, 1, 0, 1, 0, 1]  # Relations exist
relation_types = [
    "no_relation", "enable", "cause", "prevent", "cause",
    "enable", "enable", "cause", "cause", "intend"
]
train_df=pd.read_csv('/data/Youss/RE/Bert_based_classification/train.csv')
dev_df=pd.read_csv('/data/Youss/RE/Bert_based_classification/dev.csv')

texts=train_df['text'].to_list()+dev_df['text'].to_list()
subjects=train_df['subject'].to_list() +dev_df['subject'].to_list()
objects=train_df['object'].to_list()+dev_df['object'].to_list()
relation_types=train_df['relation'].to_list()+dev_df['relation'].to_list()
relation_exists = [0 if relation == "no_relation" else 1 for relation in relation_types]




dataset = RelationDataset(texts, subjects, objects, relation_exists, relation_types, tokenizer)

# Splitting dataset into train, validation, and test sets
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Define Trainer arguments
training_args = TrainingArguments(
    output_dir="./relation_extraction",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    load_best_model_at_end=True,  # Load best model for testing
    save_strategy="epoch",  # Ensure best model is saved
)
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
dataset = RelationDataset(texts, subjects, objects, relation_exists, relation_types, tokenizer)
from transformers import TrainingArguments, Trainer
#
model = MultiHeadRoBERTa.from_pretrained("roberta-base")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
#
def compute_metrics(pred):
    labels = pred.label_ids  # label_ids is a tuple
    preds = pred.predictions  # predictions is also a tuple

    # Extracting labels correctly from the tuple
    labels_relation = np.array(labels[0])  # First element is relation labels

    labels_type = np.array(labels[1])  # Second element is type labels
    labels_bio = np.array(labels[2])  # Ensuring correct shape (batch_size, seq_len)

    # Extract predictions correctly
    preds_relation = np.argmax(np.array(preds[0]), axis=-1)
    preds_type = np.array(preds[1])
    preds_bio =np.array(preds[2])  # Ensuring correct shape (batch_size, seq_len)
    print("Shape of labels_bio:", labels_bio.shape)
    print("Shape of preds_bio:", preds_bio.shape)
    print("Shape of tye:", labels_type.shape)
    print("Shape of type:", preds_type.shape)
    print("Shape of relation:", labels_relation.shape)
    print("Shape of relation:", preds_relation.shape)

    # # Ensure labels_bio and preds_bio have the same shape
    # if labels_bio.shape != preds_bio.shape:
    #     min_length = min(labels_bio.shape[-1], preds_bio.shape[-1])  # Get the correct sequence length
    #     labels_bio = labels_bio[:, :min_length]  # Truncate or reshape
    #     preds_bio = preds_bio[:, :min_length]  # Truncate or reshape

    # Ensure labels_type and labels_bio are arrays before computing accuracy
    if labels_type.ndim == 0:
        labels_type = np.array([labels_type])
    if preds_type.ndim == 0:
        preds_type = np.array([preds_type])
    if labels_bio.ndim == 0:
        labels_bio = np.array([labels_bio])
    if preds_bio.ndim == 0:
        preds_bio = np.array([preds_bio])

    # Compute accuracy for each head
    relation_acc = accuracy_score(labels_relation, preds_relation)
    type_acc = accuracy_score(labels_type, preds_type) if len(labels_type) > 0 else 0.0
    # bio_acc = accuracy_score(labels_bio.flatten(), preds_bio.flatten()) if len(labels_bio) > 0 else 0.0

    return {
        "relation_accuracy": relation_acc,
        "type_accuracy": type_acc,
        # "bio_accuracy": bio_acc
    }




# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)


# Train model
trainer.train()

# # Evaluate model
# results = trainer.evaluate(test_dataset)
# print("Test Set Accuracy:", results["eval_accuracy"])
def predict_relation(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = {key: val.to(device) for key, val in inputs.items()}  # Move to GPU

    with torch.no_grad():
        outputs = model(**inputs)

    relation_pred = torch.argmax(outputs["relation_logits"], dim=-1).item()

    if relation_pred == 1 and outputs["relation_type_logits"] is not None:
        relation_type = torch.argmax(outputs["relation_type_logits"], dim=-1).item()
    else:
        relation_type = None  # No relation type if relation_pred == 0

    if relation_pred == 1 and outputs["span_logits"] is not None:
        spans = torch.argmax(outputs["span_logits"], dim=-1).tolist()
    else:
        spans = None  # No spans if relation_pred == 0

    return relation_pred, relation_type, spans

# Example Call
print(predict_relation("Jeff Bezos founded Amazon."))