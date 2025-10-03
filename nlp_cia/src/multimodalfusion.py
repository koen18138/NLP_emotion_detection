from ast import literal_eval
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from typing import List, Dict
from sklearn.metrics import classification_report
import os
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

class EmotionDataset(Dataset):
    def __init__(self, text_data, numeric_data_df, y_data):
        self.text = text_data
        self.numeric_df = numeric_data_df
        self.labels = y_data
        # Build label mapping if labels are not integers
        if not all(isinstance(lbl, (int, np.integer)) for lbl in self.labels):
            unique_labels = sorted(set(self.labels))
            self.label2id = {label: i for i, label in enumerate(unique_labels)}
        else:
            self.label2id = None
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # A. Process Text
        encoding = tokenizer(
            self.text[idx],
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=128
        )

        # B. Process Numeric Features (The main change)
        # Helper function to map categorical tags to numeric indices
        def get_mapping(df_col: pd.Series):
            # if df_col is a pandas series
            if isinstance(df_col, pd.Series): # The ID '0' is reserved for padding
                return {tag: i+1 for i, tag in enumerate(df_col.explode().unique()) if pd.notna(tag)}
            # if df_col is a Python list wrap it into a series
            elif isinstance(df_col, list): # The ID '0' is reserved for padding
                ser = pd.Series(df_col)
                return {tag: i+1 for i, tag in enumerate(ser.explode().unique()) if pd.notna(tag)}

        # Helper function for padding/truncation/conversion
        def process_list_feature(feature_list, is_categorical=False):
            # 1. Truncate
            truncated = feature_list[:MAX_SEQ_LEN]
            
            # 2. Convert to numeric indices if categorical
            if is_categorical:
                # Assuming you have a NER_TAG_MAP similar to POS_TAG_MAP
                # For this example, we'll use the POS_TAG_MAP for both
                tag_map = get_mapping(truncated) 
                numeric_list = [tag_map.get(tag, PAD_TOKEN_ID) for tag in truncated]
            else:
                numeric_list = truncated

            # 3. Pad
            padded = numeric_list + [PAD_TOKEN_ID] * (MAX_SEQ_LEN - len(numeric_list))
            return padded

        # Process the list features
        pos_features = process_list_feature(self.numeric_df.iloc[idx]['POS_Tags'], is_categorical=True)
        tfidf_features = process_list_feature(self.numeric_df.iloc[idx]['TF_IDF'], is_categorical=False)
        ner_features = process_list_feature(self.numeric_df.iloc[idx]['NER_Tags'], is_categorical=True)
        
        # Flatten the features (This creates the fixed-size vector)
        # We assume 'Embedding' is already a flat list/array of fixed length E (e.g., 100)
        numeric_list = tfidf_features + pos_features + ner_features + \
                       [self.numeric_df.iloc[idx]['Sentiment_Score']] + \
                       self.numeric_df.iloc[idx]['Embedding'].tolist() # Convert array/series to list
        numeric_tensor = torch.tensor(numeric_list, dtype=torch.float)
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'numeric_features': numeric_tensor,
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }
class TextEncoder(nn.Module):
    def __init__(self, bert_model_name):
        super().__init__()
        # Load a pre-trained BERT model
        self.bert = AutoModel.from_pretrained(bert_model_name)
        # Note: BERT's output is the "Embeddings" in your diagram

    def forward(self, input_ids, attention_mask):
        # Pass the tokenized text through BERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Use the pooled output (representation of the [CLS] token)
        cls_output = outputs.pooler_output
        return cls_output

class NumericEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        # MLP layers to transform numeric features into a fixed-size embedding
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), # Input: numeric feature vector
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim) # Output: the "Embeddings" vector
        )

    def forward(self, numeric_features):
        return self.mlp(numeric_features)
    

class MultiModalClassifier(nn.Module):
    def __init__(self, bert_output_dim, numeric_embed_dim, num_emotion_classes, input_dim):
        super().__init__()
        # 1. Initialize the encoders
        self.text_encoder = TextEncoder('bert-base-uncased') # Use any pre-trained BERT
        self.numeric_encoder = NumericEncoder(
            input_dim=input_dim,
            hidden_dim=64, 
            output_dim=numeric_embed_dim 
        )

        # 2. Define the Fusion/Classification MLP
        # The concatenated dimension: BERT output dim (e.g., 768) + Numeric embedding dim (e.g., 128)
        fused_dim = bert_output_dim + numeric_embed_dim
        
        # This is the final "MLP" block in your diagram
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fused_dim, 256), # Fusion MLP hidden layer
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_emotion_classes) # Final output layer
        )

    def forward(self, input_ids, attention_mask, numeric_features):
        # 1. Process Text Input
        text_embeds = self.text_encoder(input_ids, attention_mask)
        
        # 2. Process Numeric Input
        numeric_embeds = self.numeric_encoder(numeric_features)
        
        # 3. Concatenate (Fusion)
        fused_features = torch.cat((text_embeds, numeric_embeds), dim=1) 
        
        # 4. Final Classification (Output)
        output = self.fusion_mlp(fused_features)
        return output


def get_multi_modal_predictions(model: nn.Module, data_loader: DataLoader, device: torch.device) -> List[int]:
    """
    Performs inference on a test dataset using the custom MultiModalClassifier.
    
    Args:
        model (nn.Module): The loaded MultiModalClassifier.
        data_loader (DataLoader): DataLoader for the test dataset.
        device (torch.device): Device (cuda or cpu) to run inference on.
    
    Returns:
        List[int]: A list of predicted class labels.
    """
    print("\nStarting multi-modal inference...")
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in data_loader:
            # 1. Move all data to the device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            numeric_features = batch['numeric_features'].to(device)
            
            # 2. Forward pass for the multi-modal model
            outputs = model(
                input_ids=input_ids, 
                attention_mask=attention_mask,
                numeric_features=numeric_features # <--- Crucial line for multi-modal
            )
            
            # 3. Get predictions from logits
            logits = outputs # The model's forward returns logits directly
            batch_predictions = torch.argmax(logits, dim=1).tolist()
            predictions.extend(batch_predictions)
    
    return predictions

def evaluate_model(model, test_dataloader: DataLoader, test_labels: List[int], output_dir: str, device: torch.device):
    """
    Evaluates a loaded multi-modal model on a test dataset and saves the report.
    """
    # Use the new multi-modal prediction function
    predictions = get_multi_modal_predictions(model, test_dataloader, device)
    
    # Generate and save the classification report
    report_string = classification_report(
        test_labels, # The true labels
        predictions, # The predicted labels
        target_names=['happiness', 'anger', 'surprise', 'sadness', 'disgust', 'fear', 'neutral'],
        zero_division=0
    )

    print("\n--- Classification Report ---")
    print(report_string)

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "classification_report.txt")
    
    with open(report_path, "w") as f:
        f.write(report_string)

    print(f"\nClassification report saved to {report_path}")

def str_eval(df):
    df['TF_IDF'] = df['TF_IDF'].apply(literal_eval)
    df['Embedding'] = df['Embedding'].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))
    df['POS_Tags'] = df['POS_Tags'].apply(literal_eval)
    df['NER_Tags'] = df['NER_Tags'].apply(literal_eval)
    return df
def data_split(df):
    try:
        y = df["Emotion_encoded"].tolist()
    except KeyError:
        y = df["Emotion_core"].tolist()
    finally:   
        text = df["Sentence"].tolist()
        numeric = df[[
            'POS_Tags',
            'TF_IDF', 
            'Sentiment_Score', 
            'Embedding', 
            'NER_Tags'
        ]]

 
   
    return y, text, numeric
if __name__ == "__main__":
    import pandas as pd

    from os import getcwd
    from sklearn.model_selection import train_test_split
    import re
    import numpy as np

    print(getcwd())

    df = pd.read_csv('data/features/NLP_features.csv')
    df_test = pd.read_csv('data/features/NLP_features copy 2.csv')
    # Shuffle the full dataframes to ensure labels are mixed
    df_subset = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df_subset = df_subset.head(500)  # Limit to first 500 rows for testing purposes

    df = str_eval(df)
    df_test = str_eval(df_test)

    print(df['Embedding'])
    print(df.head())
    print(df.columns)
    y_train, text_train, numeric_train = data_split(df)
    y_test, text_test, numeric_test = data_split(df_test)

    
    train_texts, val_texts, \
    train_numeric, val_numeric, \
    train_labels, val_labels = train_test_split(
        text_train,       # X1 (Text data)
        numeric_train,    # X2 (Numeric data)
        y_train,          # y (Labels)
        test_size=0.2, 
        random_state=42
    )
    
    MAX_SEQ_LEN = 128 # Max length for POS/TF-IDF lists
    PAD_TOKEN_ID = 0 # ID for padding POS/NER tags
    # Calculate the new INPUT_DIM
    # Assuming E=100 for the 'Embedding' feature
    EMBEDDING_DIM = 300 

    FINAL_INPUT_DIM = (3 * MAX_SEQ_LEN) + 1 + EMBEDDING_DIM 
    
    # --- Model Initialization ---
    BERT_DIM = 768 
    NUMERIC_EMBED_DIM = 128 
    NUM_CLASSES = 7 

    model = MultiModalClassifier(
        bert_output_dim=BERT_DIM, 
        numeric_embed_dim=NUMERIC_EMBED_DIM, 
        num_emotion_classes=NUM_CLASSES,
        input_dim=FINAL_INPUT_DIM # Use the calculated dimension
    )

    # --- Create Data Loaders ---
    train_dataset = EmotionDataset(train_texts, train_numeric, train_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # --- Training Loop Example ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(3): # Simple example for 3 epochs
        model.train()
        for batch in train_dataloader:
            optimizer.zero_grad()
            
            # Move data to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            numeric_features = batch['numeric_features'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(input_ids, attention_mask, numeric_features)
            
            # Calculate loss and backpropagate
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch+1} done. Loss: {loss.item():.4f}")

    test_dataset = EmotionDataset(val_texts, val_numeric, val_labels)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False) 

    OUTPUT_DIR = "model_evaluation_results"
    
    print("\n--- Starting Model Evaluation on Validation Set ---")
    
    evaluate_model(
        model=model, 
        test_dataloader=test_dataloader, 
        test_labels=val_labels, # The true labels for the validation set
        output_dir=OUTPUT_DIR,
        device=device
    ) 