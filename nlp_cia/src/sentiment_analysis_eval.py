import os
import torch
import pandas as pd
from sklearn.metrics import classification_report
from utils import load_model_and_tokenizer, load_inference_data
from transformers import BertTokenizer, BertForSequenceClassification
from typing import List, Dict


def get_predictions(model: BertForSequenceClassification, encodings: Dict[str, torch.Tensor]) -> List[int]:
    """
    Performs inference on a list of sentences using a fine-tuned BERT model.
    """
    print("Tokenizing input data for inference...")

    model.eval()
    predictions = []

    print("Making predictions...")
    with torch.no_grad():
        # Pass only the relevant arguments to the model
        outputs = model(
            input_ids=encodings['input_ids'], 
            attention_mask=encodings['attention_mask']
        )
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1).tolist()
    
    return predictions

def evaluate_model(model, tokenizer, test_df: pd.DataFrame, output_dir: str):
    """
    Evaluates a loaded model on a test dataset, prints a classification report,
    and saves the report to a file.

    Args:
        model (BertForSequenceClassification): The loaded fine-tuned model.
        tokenizer (BertTokenizer): The loaded tokenizer.
        test_df (pd.DataFrame): The test dataset containing 'sentence' and 'sentiment' columns.
        output_file_path (str): Path to save the classification report.
    """

    df_y = test_df['sentiment'].tolist()
    x, _ = load_inference_data(tokenizer, test_df)
    print(f"{x = }")
    # Use the new get_predictions function to get the model's output
    predictions = get_predictions(model, x)

    # Generate and save the classification report
    report_string = classification_report(
        df_y,
        predictions,
        target_names=['negative', 'neutral', 'positive'],
        zero_division=0
    )

    print("\n--- Classification Report ---")
    print(report_string)

    with open(os.path.join(output_dir,  "classification_report.txt"), "w") as f:
        f.write(report_string)

    print(f"\nClassification report saved to {output_dir}/classification_report.txt")

if __name__ == "__main__":
    try:
        test_df = pd.read_csv("data\\wie_is_de_mol_sentiment.csv").dropna()
        test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)
    except FileNotFoundError:
        print("Test data file not found. Please ensure 'data/sentence_sentiment.csv' exists.")
        exit()

    save_dir = 'models\\Model_14e03c00'
    
    # Load the model and tokenizer
    tokenizer, model = load_model_and_tokenizer(model_dir=save_dir)

    # Evaluate the loaded model
    evaluate_model(model=model, tokenizer=tokenizer, test_df=test_df, output_dir=save_dir)