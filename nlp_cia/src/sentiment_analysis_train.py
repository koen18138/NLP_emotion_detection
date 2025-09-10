import os
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, EvalPrediction
from sklearn.model_selection import train_test_split
import pandas as pd
from bs4 import BeautifulSoup
import re
import uuid
import json
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

def load_dataset(directory):
    data = {"sentence": [], "sentiment": []}
    for file_name in os.listdir(directory):
        print(file_name)
        if file_name == 'pos':
            positive_dir = os.path.join(directory, file_name)
            for text_file in os.listdir(positive_dir):
                text = os.path.join(positive_dir, text_file)
                with open(text, "r", encoding="utf-8") as f:
                    data["sentence"].append(f.read())
                    data["sentiment"].append(1)
        elif file_name == 'neg':
            negative_dir = os.path.join(directory, file_name)
            for text_file in os.listdir(negative_dir):
                text = os.path.join(negative_dir, text_file)
                with open(text, "r", encoding="utf-8") as f:
                    data["sentence"].append(f.read())
                    data["sentiment"].append(0)
            
    return pd.DataFrame.from_dict(data)

def text_cleaning(text):
    soup = BeautifulSoup(text, "html.parser")
    text = re.sub(r'\[[^]]*\]', '', soup.get_text())
    pattern = r"[^a-zA-Z0-9\s,']"
    text = re.sub(pattern, '', text)
    return text

def compute_metrics(p: EvalPrediction):
    """
    Computes accuracy and f1 score for evaluation.
    """
    preds = np.argmax(p.predictions, axis=1)
    print(f"{preds = }")
    print(f"{p.label_ids = }")
    return {
        'accuracy': accuracy_score(p.label_ids, preds),
        'f1': f1_score(p.label_ids, preds, average='weighted'),
    }

def encode_data(texts, tokenizer, labels=None, max_len=128,):
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_len)
    if labels is None:
        return {
            'input_ids': torch.tensor(encodings['input_ids']),
            'attention_mask': torch.tensor(encodings['attention_mask'])
        }
    if labels:
        return {
            'input_ids': torch.tensor(encodings['input_ids']),
            'attention_mask': torch.tensor(encodings['attention_mask']),
            'labels': torch.tensor(labels)
        }

class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        # Store only the necessary tensors from the encodings
        self.encodings = {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask']
        }

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        # Return a dictionary containing the tensors for a single item
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx]
        }    
    
if __name__ == "__main__":

    # Get the current working directory
    current_folder = os.getcwd()
    print(current_folder)

    # dataset = tf.keras.utils.get_file(
    #     fname ="aclImdb.tar.gz", 
    #     origin ="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
    #     cache_dir=  current_folder,
    #     extract = True)
    
    # dataset_path = os.path.dirname(dataset)
    # # Dataset directory
    # dataset_dir = os.path.join(dataset_path, 'aclImdb')

    # train_dir = os.path.join(dataset_dir,'train')

    # # Load the dataset from the train_dir
    # train_df = load_dataset(train_dir)
    
    # print(train_df.head())


    # test_dir = os.path.join(dataset_dir,'test')

    # # Load the dataset from the train_dir
    # test_df = load_dataset(test_dir)
    # print(test_df.head())

    # train_df['sentence'] = train_df['sentence'].apply(text_cleaning).tolist()
    # test_df['sentence'] = test_df['sentence'].apply(text_cleaning)
    
    # train_df.to_csv('data/train_data_aclImdb_v1.csv', index=False)
    # test_df.to_csv('data/test_data_aclImdb_v1.csv', index=False)

    train_df = pd.read_csv(f"data/train_data_Sp1786.csv")
    test_df = pd.read_csv(f"data/test_data_Sp1786.csv")
    # Shuffle the full dataframes to ensure labels are mixed
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True).iloc[:100]
    test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True).iloc[:40]
    print(f"Train data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")

    # Training data
    Reviews = train_df['sentence'].tolist()
    Target = train_df['sentiment'].tolist()

    # Test data
    test_reviews = test_df['sentence'].tolist()
    test_targets = test_df['sentiment'].tolist()

    x_val, x_test, y_val, y_test = train_test_split(test_reviews, test_targets, test_size=0.5, stratify=test_targets)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    max_len = 128

    train_dataset = encode_data(Reviews, Target)
    val_dataset = encode_data(x_val, y_val)
    test_dataset = encode_data(x_test, y_test)

    train_ds = SentimentDataset(train_dataset)
    val_ds = SentimentDataset(val_dataset)
    test_ds = SentimentDataset(test_dataset)

    num_labels = train_df['sentiment'].nunique()
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        logging_dir='./logs',
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    eval_results = trainer.evaluate(test_ds)
    print(f"Test results: {eval_results}")

    run_id = str(uuid.uuid4())[:8]
    save_dir = f'./Model_{run_id}'
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    with open(os.path.join(save_dir, 'eval_results.json'), 'w') as f:
        json.dump(eval_results, f, indent=2)
    if hasattr(trainer, 'state') and hasattr(trainer.state, 'log_history'):
        with open(os.path.join(save_dir, 'train_history.json'), 'w') as f:
            json.dump(trainer.state.log_history, f, indent=2)