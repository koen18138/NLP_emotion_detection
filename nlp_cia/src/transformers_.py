import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from bs4 import BeautifulSoup
import re
from typing import List, Dict
import torch

from sklearn.preprocessing import LabelEncoder
# Use the emotion mapping to convert string labels to integers
LABEL_MAP = {
    'happiness': 0,
    'anger': 1,
    'surprise': 2,
    'sadness': 3,
    'disgust': 4,
    'fear': 5,
    'neutral': 6
}

# Initialize and fit the label encoder
label_encoder = LabelEncoder()
label_encoder.fit(list(LABEL_MAP.keys()))

def load_model_and_tokenizer(model_dir: str = 'models/model_pretrained'):
    """
    Loads a fine-tuned BERT model and tokenizer from a specified directory.

    Args:
        model_dir (str): The directory where the model and tokenizer are saved.

    Returns:
        tuple: A tuple containing the loaded tokenizer and model.
    """
    print(f"Loading model and tokenizer from: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    return tokenizer, model

def load_inference_data(tokenizer, inference_data: pd.DataFrame=None, file_path: str="data\\transcription\\csv\\transcription.csv", text_column:str='Translation') -> list:
	"""
	Loads inference data from a CSV file.

	Args:
		file_path (str): The path to the CSV file containing inference data.
	Returns:
		list: A list of sentences for inference.
	"""
	print(os.getcwd())
	if inference_data is not None:
		df = inference_data
	else:
		df = pd.read_csv(file_path)
	sentences = df[text_column].apply(text_cleaning).tolist()
	encoded_sentences = encode_data(sentences, tokenizer)
	return encoded_sentences, df

def text_cleaning(text):
    soup = BeautifulSoup(text, "html.parser")
    text = re.sub(r'\[[^]]*\]', '', soup.get_text())
    pattern = r"[^a-zA-Z0-9\s,']"
    text = re.sub(pattern, '', text)
    return text

# def compute_metrics(p):
#     """
#     Computes accuracy, f1 score, precision, and recall for evaluation.
#     """
#     preds = np.argmax(p.predictions, axis=1)
    
#     # Calculate metrics
#     accuracy = accuracy_score(p.label_ids, preds)
#     f1 = f1_score(p.label_ids, preds, average='weighted')
#     precision = precision_score(p.label_ids, preds, average='weighted')
#     recall = recall_score(p.label_ids, preds, average='weighted')
    
#     return {
#         'accuracy': accuracy,
#         'f1': f1,
#         'precision': precision,
#         'recall': recall
#     }

def encode_data(texts, tokenizer, labels=None, max_len=128):
    """
    Encode texts and labels using a tokenizer.

    Args:
        texts (list): A list of sentences to encode.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use.
        labels (list, optional): A list of emotion labels. Defaults to None.
        max_len (int, optional): The maximum length for the tokenized output. Defaults to 128.

    Returns:
        dict: A dictionary containing 'input_ids', 'attention_mask', and 'labels' (if provided).
    """
    print(f"Texts to process: {len(texts)} sentences")
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_len)

    if labels is None:
        return {
            'input_ids': torch.tensor(encodings['input_ids']),
            'attention_mask': torch.tensor(encodings['attention_mask'])
        }

    # Encode labels using the label encoder
    labels = label_encoder.transform(labels)

    return {
        'input_ids': torch.tensor(encodings['input_ids']),
        'attention_mask': torch.tensor(encodings['attention_mask']),
        'labels': torch.tensor(labels, dtype=torch.long)  # Change dtype to torch.long
    }

# class SentimentDataset(torch.utils.data.Dataset):
#     def __init__(self, encodings):
#         self.encodings = {
#             'input_ids': encodings['input_ids'],
#             'attention_mask': encodings['attention_mask'],
#             'labels': encodings['labels']
#         }

#     def __len__(self):
#         return len(self.encodings['input_ids'])

#     def __getitem__(self, idx):
#         return {
#             'input_ids': self.encodings['input_ids'][idx],
#             'attention_mask': self.encodings['attention_mask'][idx],
#             'labels': self.encodings['labels'][idx]
#         }
    
# def evaluate_model(model, tokenizer, test_df: pd.DataFrame, output_dir: str):
#     """
#     Evaluates a loaded model on a test dataset, prints a classification report,
#     and saves the report to a file.

#     Args:
#         model (BertForSequenceClassification): The loaded fine-tuned model.
#         tokenizer (BertTokenizer): The loaded tokenizer.
#         test_df (pd.DataFrame): The test dataset containing 'sentence' and 'sentiment' columns.
#         output_file_path (str): Path to save the classification report.
#     """

#     y = encode_data(test_df['Sentence'].tolist(), tokenizer, test_df['Emotion_core'].tolist())
#     inference_data = test_df['Sentence'].tolist()
#     x = encode_data(inference_data, tokenizer)
#     # Use the new get_predictions function to get the model's output
#     predictions = get_predictions(model, x)

#     # Generate and save the classification report
#     report_string = classification_report(
#         y['labels'],
#         predictions,
#         target_names=['happiness', 'anger', 'surprise', 'sadness', 'disgust', 'fear', 'neutral'],
#         zero_division=0
#     )

#     print("\n--- Classification Report ---")
#     print(report_string)

#     with open(os.path.join(output_dir,  "classification_report.txt"), "w") as f:
#         f.write(report_string)

#     print(f"\nClassification report saved to {output_dir}/classification_report.txt")

def get_predictions(model: AutoModelForSequenceClassification, encodings: Dict[str, torch.Tensor]) -> List[str]:
    """
    Performs inference on a list of sentences using a fine-tuned BERT model.

    Args:
        model (AutoModelForSequenceClassification): The loaded fine-tuned model.
        encodings (Dict[str, torch.Tensor]): The encoded input data.

    Returns:
        List[str]: A list of predicted labels.
    """
    print("Tokenizing input data for inference...")

    model.eval()
    predictions = []

    print("Making predictions...")
    with torch.no_grad():
        outputs = model(
            input_ids=encodings['input_ids'], 
            attention_mask=encodings['attention_mask']
        )
        logits = outputs.logits
        predicted_indices = torch.argmax(logits, dim=1).tolist()

        # Map predicted indices to labels using the label encoder
        predictions = label_encoder.inverse_transform(predicted_indices)
    
    return predictions

# def run_experiment(model_name: str, dataset_path: str, output_dir: str):
#     """
#     Runs an experiment with the specified model and dataset.

#     Args:
#         model_name (str): The name of the model to use (e.g., 'bert-base-uncased').
#         dataset_path (str): The path to the dataset file.
#         output_dir (str): The directory to save the results.
#     """
#     print(f"Running experiment with model: {model_name} and dataset: {dataset_path}")

#     # Load the dataset
#     df = pd.read_csv(dataset_path)
#     df_test = pd.read_csv('data\\group_4_url_1_transcript.csv')

#     # Shuffle the full dataframes to ensure labels are mixed
#     df_subset = df.sample(frac=1, random_state=42).reset_index(drop=True)
#     df_subset = df_subset.head(10_000)  # Limit to first 500 rows for testing purposes

#     # Training data
#     x_data = df_subset['Sentence'].tolist()
#     y_data = df_subset['Emotion_core'].tolist()

#     x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.3)
#     print(f"Train data shape: {x_train = } - {y_train = }")
#     print(f"Val data shape: {x_val = } - {y_val = }")

#     unique_types = df['Emotion_encoded'].nunique()
#     print(f"Number of unique emotion types: {unique_types}")

#     tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=True)
#     max_len = 128

#     train_dataset = encode_data(x_train, tokenizer, y_train)
#     val_dataset = encode_data(x_val, tokenizer, y_val)
#     test_dataset = encode_data(df_test['Sentence'].tolist(), tokenizer, df_test['Emotion_core'].tolist())

#     train_ds = SentimentDataset(train_dataset)
#     val_ds = SentimentDataset(val_dataset)
#     test_ds = SentimentDataset(test_dataset)

#     model = BertForSequenceClassification.from_pretrained(model_name, num_labels=unique_types)

#     training_args = TrainingArguments(
#         output_dir=f'{output_dir}\\results',
#         num_train_epochs=3,
#         per_device_train_batch_size=32,
#         per_device_eval_batch_size=32,
#         logging_dir=f'{output_dir}\\logs',
#         logging_steps=10,
#     )

#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=train_ds,
#         eval_dataset=val_ds,
#         compute_metrics=compute_metrics,
#     )

#     # Train and evaluate the model
#     trainer.train()
#     eval_results = trainer.evaluate(test_ds)
#     evaluate_model(model, tokenizer, df_test, output_dir)

#     print(f"Test results: {eval_results}")

#     # Save the model and results
#     os.makedirs(output_dir, exist_ok=True)
#     model.save_pretrained(output_dir)
#     tokenizer.save_pretrained(output_dir)
#     with open(os.path.join(output_dir, 'eval_results.json'), 'w') as f:
#         json.dump(eval_results, f, indent=2)
#     if hasattr(trainer, 'state') and hasattr(trainer.state, 'log_history'):
#         with open(os.path.join(output_dir, 'train_history.json'), 'w') as f:
#             json.dump(trainer.state.log_history, f, indent=2)

# if __name__ == "__main__":
#     # Define experiments
#     experiments = [
#         # {"model_name": "GroNLP/bert-base-dutch-cased", "dataset_path": "data\\dataset\\processed\\go_emotion_dutch.csv"},
#         # {"model_name": "bert-base-uncased", "dataset_path": "data\\dataset\\processed\\go_emotion_dutch.csv"},
#         {"model_name": "GroNLP/bert-base-dutch-cased", "dataset_path": "data\\features\\NLP_features.csv"},
#         {"model_name": "bert-base-uncased", "dataset_path": "data\\features\\NLP_features.csv"},
#     ]

#     # Run each experiment
#     for i, experiment in enumerate(experiments):
#         run_id = str(uuid.uuid4())[:8]
#         output_dir = f'models\\experiment_{i + 1}_{run_id}'
#         run_experiment(experiment["model_name"], experiment["dataset_path"], output_dir)

if __name__ == "__main__":
    pass