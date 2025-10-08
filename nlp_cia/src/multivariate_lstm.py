import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import Counter
from itertools import chain
from typing import List, Dict, Any, Tuple
import os
import ast

# Fine-grain emotion map to simple emotion
emotion_mapping = {
    0: 'happiness',  # admiration
    1: 'happiness',  # amusement
    2: 'anger',      # anger
    3: 'anger',      # annoyance
    4: 'happiness',  # approval
    5: 'happiness',  # caring
    6: 'surprise',   # confusion
    7: 'neutral',    # curiosity
    8: 'neutral',    # desire
    9: 'sadness',    # disappointment
    10: 'anger',     # disapproval
    11: 'disgust',   # disgust
    12: 'sadness',   # embarrassment
    13: 'happiness', # excitement
    14: 'fear',      # fear
    15: 'happiness', # gratitude
    16: 'sadness',   # grief
    17: 'happiness', # joy
    18: 'happiness', # love
    19: 'fear',      # nervousness
    20: 'happiness', # optimism
    21: 'happiness', # pride
    22: 'surprise',  # realization
    23: 'happiness', # relief
    24: 'sadness',   # remorse
    25: 'sadness',   # sadness
    26: 'surprise',  # surprise
    27: 'neutral'    # neutral
}

# Function to map labels to emotion
def map_labels(labels_list):
    if len(labels_list) == 0:
        return 'neutral'
    return emotion_mapping[labels_list[0]]

def build_vocab(texts: List[str], max_words: int = 5000) -> Dict[str, int]:
        """
        Builds a word-to-index mapping (vocabulary) from a list of texts.

        Args:
            texts: A list of text strings.
            max_words: The maximum number of words to include in the vocabulary 
                    (excluding <PAD> and <OOV>).

        Returns:
            A dictionary mapping words to their corresponding integer indices.
        """
        # Count word frequencies across all texts
        counter: Counter = Counter(chain.from_iterable(t.split() for t in texts))
        # Get the most common words up to max_words
        most_common: List[Tuple[str, int]] = counter.most_common(max_words)
        # Create the vocabulary, starting indices from 2
        word2idx: Dict[str, int] = {word: idx + 2 for idx, (word, _) in enumerate(most_common)}
        # Add special tokens
        word2idx["<PAD>"] = 0 # Padding token
        word2idx["<OOV>"] = 1 # Out-of-Vocabulary token
        print(f"Vocabulary built with {len(word2idx)} tokens (including <PAD> and <OOV>).")
        return word2idx

def texts_to_sequences(texts: List[str], word2idx: Dict[str, int]) -> List[List[int]]:
    """
    Converts a list of text strings into a list of integer sequences using the vocabulary.

    Args:
        texts: A list of text strings.
        word2idx: The vocabulary (word-to-index mapping).

    Returns:
        A list of lists, where each inner list is the integer sequence of a text.
    """
    oov_token_idx: int = word2idx["<OOV>"]
    sequences: List[List[int]] = [
        [word2idx.get(word, oov_token_idx) for word in text.split()]
        for text in texts
    ]
    print(f"Converted {len(texts)} texts to integer sequences.")
    return sequences

def pad_sequences_torch(sequences: List[List[int]], maxlen: int, padding_value: int = 0) -> torch.Tensor:
        """
        Pads and truncates sequences to a fixed maximum length using PyTorch tensors.

        Args:
            sequences: A list of integer sequences.
            maxlen: The maximum length for padding/truncation.
            padding_value: The value to use for padding (typically 0 for <PAD>).

        Returns:
            A PyTorch Tensor of shape (num_sequences, maxlen).
        """
        padded_tensors: List[torch.Tensor] = []
        for seq in sequences:
            seq_tensor = torch.tensor(seq, dtype=torch.long)
            # Truncate if longer than maxlen
            if len(seq_tensor) > maxlen:
                seq_tensor = seq_tensor[:maxlen]
            # Pad if shorter than maxlen
            padding_needed = maxlen - len(seq_tensor)
            if padding_needed > 0:
                padding = torch.full((padding_needed,), padding_value, dtype=torch.long)
                padded_seq = torch.cat([seq_tensor, padding])
            else:
                padded_seq = seq_tensor
            padded_tensors.append(padded_seq)

        print(f"Padded {len(sequences)} sequences to max length {maxlen}.")
        return torch.stack(padded_tensors)

class TextFeatureDataset(Dataset):
    """
    A custom PyTorch Dataset for text classification incorporating multiple feature types.
    """
    def __init__(self, 
                 text_X: torch.Tensor, 
                 pos_X: torch.Tensor, 
                 tfidf_X: torch.Tensor, 
                 sentiment_X: torch.Tensor, 
                 embed_X: torch.Tensor, 
                 y: torch.Tensor):
        """
        Initializes the dataset with all features.

        Args:
            text_X: Padded word sequence tensor (for main LSTM).
            pos_X: Padded POS tag sequence tensor (for POS LSTM/Embedding).
            tfidf_X: TF-IDF vector tensor.
            sentiment_X: Sentiment score scalar tensor.
            embed_X: Static sentence embedding vector tensor.
            y: Target labels tensor.
        """
        self.text_X = text_X
        self.pos_X = pos_X
        self.tfidf_X = tfidf_X
        self.sentiment_X = sentiment_X
        self.embed_X = embed_X
        self.y = y

    def __len__(self) -> int:
        """Returns the total number of samples."""
        return len(self.text_X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        """Returns the features and label for a given index."""
        # Return all features as a tuple
        return (self.text_X[idx], self.pos_X[idx], self.tfidf_X[idx], 
                self.sentiment_X[idx], self.embed_X[idx], self.y[idx])

class MultiFeatureLSTMClassifier(nn.Module):
    """
    A Bidirectional LSTM-based classifier incorporating multiple NLP features.
    """
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, num_classes: int,
                 pos_vocab_size: int, pos_embed_dim: int,
                 tfidf_dim: int, static_embed_dim: int):
        """
        Initializes the multi-feature LSTM classifier.

        Args:
            vocab_size: Size of the word vocabulary.
            embed_dim: Dimension of word embeddings.
            hidden_dim: LSTM hidden state dimension.
            num_classes: Number of output classes.
            pos_vocab_size: Size of the POS tag vocabulary.
            pos_embed_dim: Dimension of POS tag embeddings.
            tfidf_dim: Dimension of the TF-IDF vector.
            static_embed_dim: Dimension of the static sentence embedding.
        """
        super().__init__()
        # 1. Word Sequence Processing (LSTM)
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm1 = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(hidden_dim * 2, hidden_dim, batch_first=True, bidirectional=True)

        # 2. POS Tag Processing (Embedding + Pooling)
        self.pos_embedding = nn.Embedding(pos_vocab_size, pos_embed_dim, padding_idx=0)
        # Using a simple mean pooling for POS tags
        
        # 3. Final Classification Layer
        lstm_output_dim = hidden_dim * 2
        
        # Determine the total dimension after concatenation:
        # LSTM output + POS Mean Pooling + TFIDF + Sentiment + Static Embedding
        self.final_input_dim = (
            lstm_output_dim +      # Final LSTM hidden state (h_n[-2:] concat)
            pos_embed_dim +        # Mean-pooled POS embedding
            tfidf_dim +            # TFIDF vector
            1 +                    # Sentiment score (scalar)
            static_embed_dim       # Static sentence embedding
        )
        
        self.fc = nn.Sequential(
            nn.Linear(self.final_input_dim, hidden_dim), # Additional dense layer for feature mixing
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes)
        )
        print(f"Total input dimension to FC layers: {self.final_input_dim}")

    def forward(self, 
                text_x: torch.Tensor, 
                pos_x: torch.Tensor, 
                tfidf_x: torch.Tensor, 
                sentiment_x: torch.Tensor, 
                embed_x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass using all feature inputs.
        """
        batch_size = text_x.size(0)

        # 1. Word Sequence (LSTM)
        text_embedded = self.embedding(text_x)
        lstm_out1, _ = self.lstm1(text_embedded)
        lstm_out2, (h_n, _) = self.lstm2(lstm_out1)
        # Concatenate final hidden states: (batch_size, hidden_dim * 2)
        lstm_out = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)

        # 2. POS Tags (Embedding + Mean Pooling)
        pos_embedded = self.pos_embedding(pos_x)
        # Mask padded elements (where pos_x != 0)
        mask = (pos_x != 0).unsqueeze(-1).float()
        pos_embedded_masked = pos_embedded * mask
        # Sum non-padded embeddings, divide by count (mean pooling)
        pos_sum = pos_embedded_masked.sum(dim=1)
        # Calculate sequence length (non-padded tokens)
        seq_len = mask.sum(dim=1).clamp(min=1) # clamp min=1 to avoid division by zero
        pos_mean_pooled = pos_sum / seq_len.float()
        
        # 3. Global Features (Concatenation)
        # sentiment_x is (batch_size), needs to be (batch_size, 1)
        sentiment_x_reshaped = sentiment_x.float().unsqueeze(1) 

        # Concatenate all features
        combined_features = torch.cat([
            lstm_out,               # LSTM output (batch_size, hidden_dim * 2)
            pos_mean_pooled,        # Mean-pooled POS (batch_size, pos_embed_dim)
            tfidf_x.float(),        # TFIDF (batch_size, tfidf_dim)
            sentiment_x_reshaped,   # Sentiment (batch_size, 1)
            embed_x.float()         # Static Embedding (batch_size, static_embed_dim)
        ], dim=1)

        # Final classification
        return self.fc(combined_features)
    
def safe_literal_eval(value: Any) -> Any:
    """
    Safely evaluates a string literal only if the value is a string.
    If the value is already a list, array, or other object, it is returned unchanged.
    """
    # 1. Check if the value is a string that needs parsing
    if isinstance(value, str):
        # 1a. Clean the string (optional, but good practice)
        cleaned_value = value.strip()
        
        # 1b. Check for common NaN string representations
        if cleaned_value.lower() in ('nan', 'none', 'null', ''):
             # Return an empty list or appropriate zero-vector if needed
            return [] 

        # 1c. Attempt the literal evaluation
        try:
            return ast.literal_eval(cleaned_value)
        except ValueError as e:
            # Handle malformed strings if they exist
            print(f"Warning: Failed to evaluate string: '{cleaned_value[:50]}...'. Error: {e}")
            return []
            
    # 2. If it's already a list, np.ndarray, float (like NaN), or other non-string, return it
    # Note: Checking for np.ndarray is crucial based on your traceback
    return value


# --- How to apply this in your code (replace the original faulty lines) ---

# This function assumes you have a function named str_eval that performs this conversion
def str_eval(df: pd.DataFrame) -> pd.DataFrame:
    # Use the column names that typically get saved as string representations of arrays
    for col in ['Embedding', 'POS_Tags', 'TF_IDF']:
        if col in df.columns:
            # Apply the safer, type-checking parser
            df[col] = df[col].apply(safe_literal_eval)
    return df
    
def pos_to_sequences(pos_lists: List[List[str]], pos_word2idx: Dict[str, int]) -> List[List[int]]:
    oov_idx = pos_word2idx["<OOV>"]
    return [[pos_word2idx.get(tag, oov_idx) for tag in pos_list] for pos_list in pos_lists]   

def pad_feature_vectors(vectors: np.ndarray, max_len: int = 128) -> np.ndarray:
    """
    Pads a list/array of 1D NumPy arrays to the length of max_len.
    
    Args:
        vectors: A NumPy array containing 1D NumPy arrays of different lengths.
        
    Returns:
        A 2D NumPy array where all vectors are padded with 0.0 to the same length.
    """

    # 1. Pad each vector
    padded_vectors = []
    for vector in vectors:
        if len(vector) < max_len:
            # Create padding array of zeros
            padding_needed = max_len - len(vector)
            # Pad with 0.0 (float)
            padded_vector = np.pad(vector, (0, padding_needed), 'constant', constant_values=(0.0, 0.0))
        else:
            padded_vector = vector
        padded_vectors.append(padded_vector)
        
    return np.stack(padded_vectors)

if __name__ == "__main__":
    from feature_extraction import (
        sent_embedding_spacy, 
        add_sentiment_scores, 
        create_tfidf_dataframe, 
        get_nlp_features_spacy,
        nlp
    )
    # --- Configuration and Data Loading ---

    print("--- Configuration and Data Loading ---")

    # Define split paths
    splits: Dict[str, str] = {
        'train': 'simplified/train-00000-of-00001.parquet',
        'validation': 'simplified/validation-00000-of-00001.parquet',
        'test': 'simplified/test-00000-of-00001.parquet'
    }
    # Base URL for the Hugging Face dataset
    HF_DATASET_BASE_URL: str = "hf://datasets/google-research-datasets/go_emotions/"
    # Maximum vocabulary size
    MAX_VOCAB_SIZE: int = 5000
    # Maximum sequence length for padding/truncation
    MAX_LENGTH: int = 128
    # Training configuration
    BATCH_SIZE: int = 32
    LEARNING_RATE: float = 1e-3
    NUM_EPOCHS: int = 50
    PATIENCE: int = 5 # Early stopping patience

    # Load the merged data
    train_path = r'data/features/go_emotion_eng.csv'
    test_path = r"task 6/test_improved.xlsx"

    for filepath in [train_path, test_path]:
        # Extract the base filename (without extension) and the extension
        basefilename, ext = os.path.basename(filepath).split('.')
        
        # Define the output path for the processed features CSV
        output_filepath = f'data/features/{basefilename}_features.parquet'
        
        # Check if the feature file already exists to avoid re-extraction
        if os.path.exists(output_filepath):
            # If the file exists, skip feature extraction for this file
            print(f"Features already extracted: {output_filepath}. Skipping.")
            continue
        else:
            print(f"Starting feature extraction for: {filepath}")
            
            # Read the input file based on its extension
            if ext == "xlsx":
                df = pd.read_excel(filepath)
                df = df.drop('Sentence', axis=1)
                df = df.rename(columns={'Translation':'Sentence'})
            else:
                # Default to reading as CSV
                df = pd.read_parquet(HF_DATASET_BASE_URL + splits["train"])
                df = df.rename(columns={
                    'text':'Sentence',
                    'labels':"Emotion_core"
                })
                        # Map simple emotions to fine-grain emotion labels
                df['Emotion_core'] = df['Emotion_core'].apply(map_labels)
                df = df.sample(frac=1, random_state=42).reset_index(drop=True) # Shuffle training data
                df = df.head(5_000) # Get small subset for testing

            
            # 1. Generate dense sentence embeddings (using spaCy's built-in vectors)
            df['Embedding'] = df['Sentence'].apply(sent_embedding_spacy)
            
            # 2. Generate TF-IDF features for the text data
            df = create_tfidf_dataframe(df)
            
            # 3. Extract NLP features using spaCy
            full_df = get_nlp_features_spacy(df, nlp)
            
            # 4. Add sentiment scores using a pre-trained model
            add_sentiment_scores(full_df, 'models/Model_14e03c00')

            
            # Convert string representations of arrays/lists back to actual Python objects
            df = str_eval(df)

            # Save the augmented DataFrame with all extracted features
            print(f'Saving Dataframe with features to: {output_filepath}')
            full_df = full_df[[
                'Sentence', 
                'Emotion_core', 
                'Embedding', 
                'TF_IDF',
                'POS_Tags',
                'Sentiment_Score'
            ]]
            print(full_df.columns)

            full_df.to_parquet(output_filepath.replace('.csv', '.parquet'), index=False)
            

    # --- Load Feature DataFrames ---
    print("\n--- Loading Pre-processed Feature Files ---")

    # Load the pre-processed feature CSVs
    df_train = pd.read_parquet(r'data/features/go_emotion_eng_features.parquet')
    print(f"{df_train.columns = }")
    df_test = pd.read_parquet(r'data/features/test_improved_features.parquet')
    print(f"{df_test.columns = }")



    train_labels_raw: List[str] = df_train['Emotion_core'].tolist()
    test_labels_raw: List[str] = df_test['Emotion_core'].tolist()
    train_texts: List[str] = df_train['Sentence'].astype(str).tolist()
    test_texts: List[str] = df_test['Sentence'].astype(str).tolist()

    print(f"Loaded training features: {len(df_train)} samples")
    print(f"Loaded test features: {len(df_test)} samples")

    # --- Vocabulary and Tokenization (Text) ---
    print("\n--- Vocabulary and Tokenization (Text) ---")
    word2idx: Dict[str, int] = build_vocab(train_texts + test_texts, max_words=MAX_VOCAB_SIZE)
    vocab_size: int = min(len(word2idx), MAX_VOCAB_SIZE + 2)
    train_sequences: List[List[int]] = texts_to_sequences(train_texts, word2idx)
    test_sequences: List[List[int]] = texts_to_sequences(test_texts, word2idx)
    train_padded: torch.Tensor = pad_sequences_torch(train_sequences, MAX_LENGTH, padding_value=word2idx["<PAD>"])
    test_padded: torch.Tensor = pad_sequences_torch(test_sequences, MAX_LENGTH, padding_value=word2idx["<PAD>"])
    
    # --- POS Tag Processing ---
    print("\n--- POS Tag Processing ---")
    # 1. Build POS Tag Vocabulary
    # Flatten all POS tags from both datasets
    all_pos_tags: List[str] = list(chain.from_iterable(df_train['POS_Tags'].tolist() + df_test['POS_Tags'].tolist()))
    pos_counter: Counter = Counter(all_pos_tags)
    
    # Assign indices: 0 for PAD, 1 for OOV, 2+ for actual tags
    pos_word2idx: Dict[str, int] = {tag: idx + 2 for idx, (tag, _) in enumerate(pos_counter.most_common())}
    pos_word2idx["<PAD>"] = 0
    pos_word2idx["<OOV>"] = 1
    pos_vocab_size: int = len(pos_word2idx)
    print(f"POS Tag Vocabulary size: {pos_vocab_size}")

    # 2. Convert POS Tags to Sequences and Pad
    train_pos_sequences = pos_to_sequences(df_train['POS_Tags'].tolist(), pos_word2idx)
    test_pos_sequences = pos_to_sequences(df_test['POS_Tags'].tolist(), pos_word2idx)
    
    train_pos_padded: torch.Tensor = pad_sequences_torch(train_pos_sequences, MAX_LENGTH, padding_value=pos_word2idx["<PAD>"])
    test_pos_padded: torch.Tensor = pad_sequences_torch(test_pos_sequences, MAX_LENGTH, padding_value=pos_word2idx["<PAD>"])

    # --- TF-IDF, Sentiment, and Static Embedding Tensors ---
    print("\n--- Auxiliary Feature Tensors ---")
    
    # Pad TF-IDF vectors to a uniform length before stacking
    train_tfidf_padded = pad_feature_vectors(df_train['TF_IDF'].values)
    test_tfidf_padded = pad_feature_vectors(df_test['TF_IDF'].values)

    # Determine dimensions based on the padded array
    tfidf_dim = train_tfidf_padded.shape[1]
    static_embed_dim = len(df_train['Embedding'].iloc[0]) # Assuming static embeds are fine

    # Convert the padded NumPy arrays to PyTorch tensors
    train_tfidf = torch.tensor(train_tfidf_padded, dtype=torch.float)
    test_tfidf = torch.tensor(test_tfidf_padded, dtype=torch.float)


    train_sentiment = torch.tensor(df_train['Sentiment_Score'].values, dtype=torch.float)
    test_sentiment = torch.tensor(df_test['Sentiment_Score'].values, dtype=torch.float)

    train_embed = torch.tensor(np.stack(df_train['Embedding'].values), dtype=torch.float)
    test_embed = torch.tensor(np.stack(df_test['Embedding'].values), dtype=torch.float)
    
    print(f"TFIDF Dim: {tfidf_dim}, Static Embed Dim: {static_embed_dim}")

    # TF-IDF normalization
    scaler_tfidf = StandardScaler()
    train_tfidf_padded = scaler_tfidf.fit_transform(train_tfidf_padded)
    test_tfidf_padded = scaler_tfidf.transform(test_tfidf_padded)

    # Embedding normalization
    scaler_embed = StandardScaler()
    train_embed_np = scaler_embed.fit_transform(np.stack(df_train['Embedding'].values))
    test_embed_np = scaler_embed.transform(np.stack(df_test['Embedding'].values))

    # Sentiment normalization (if needed)
    scaler_sent = StandardScaler()
    train_sentiment_np = scaler_sent.fit_transform(df_train['Sentiment_Score'].values.reshape(-1, 1)).flatten()
    test_sentiment_np = scaler_sent.transform(df_test['Sentiment_Score'].values.reshape(-1, 1)).flatten()
    # --- Label Encoding ---
    # ... (Label Encoding block remains the same) ...
    print("\n--- Label Encoding ---")
    label_encoder: LabelEncoder = LabelEncoder()
    all_labels_raw: List[str] = train_labels_raw + test_labels_raw
    label_encoder.fit(all_labels_raw)
    train_labels: torch.Tensor = torch.tensor(label_encoder.transform(train_labels_raw), dtype=torch.long)
    test_labels: torch.Tensor = torch.tensor(label_encoder.transform(test_labels_raw), dtype=torch.long)
    num_classes: int = len(label_encoder.classes_)
    print(f"Found {num_classes} emotion classes: {label_encoder.classes_}")


    # --- PyTorch Dataset and DataLoader (Updated) ---
    print("\n--- PyTorch Dataset and DataLoader (Updated) ---")

    train_dataset: TextFeatureDataset = TextFeatureDataset(
        train_padded, train_pos_padded, train_tfidf, train_sentiment, train_embed, train_labels)
    test_dataset: TextFeatureDataset = TextFeatureDataset(
        test_padded, test_pos_padded, test_tfidf, test_sentiment, test_embed, test_labels)

    train_loader: DataLoader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader: DataLoader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    print(f"Train DataLoader created with {len(train_loader)} batches.")
    print(f"Test DataLoader created with {len(test_loader)} batches.")

    # --- Model Definition (Updated) ---
    print("\n--- Model Definition (Updated) ---")
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    EMBED_DIM: int = 64
    POS_EMBED_DIM: int = 16 # New hyperparameter for POS embedding
    HIDDEN_DIM: int = 32

    model: MultiFeatureLSTMClassifier = MultiFeatureLSTMClassifier(
        vocab_size=vocab_size, 
        embed_dim=EMBED_DIM, 
        hidden_dim=HIDDEN_DIM, 
        num_classes=num_classes,
        pos_vocab_size=pos_vocab_size,
        pos_embed_dim=POS_EMBED_DIM,
        tfidf_dim=tfidf_dim,
        static_embed_dim=static_embed_dim
    ).to(device)
    criterion: nn.Module = nn.CrossEntropyLoss()
    optimizer: optim.Optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- Training and Validation Loop (Updated) ---
    # ... (Training loop logic updated to handle 5 inputs and 1 target) ...
    print("\n--- Starting Training Loop (Multi-Feature) ---")
    best_val_loss: float = float('inf')
    patience_counter: int = 0
    best_model_state: Dict[str, torch.Tensor] = model.state_dict()

    for epoch in range(NUM_EPOCHS):
        # --- Training Phase ---
        model.train()
        train_loss: float = 0.0
        
        for text_X, pos_X, tfidf_X, sentiment_X, embed_X, y_batch in train_loader:
            # Move all inputs and target to device
            text_X = text_X.to(device).long()
            pos_X = pos_X.to(device).long()
            tfidf_X = tfidf_X.to(device).float()
            sentiment_X = sentiment_X.to(device).float()
            embed_X = embed_X.to(device).float()
            y_batch = y_batch.to(device).long()
            
            optimizer.zero_grad()
            
            # Forward pass with 5 inputs
            outputs: torch.Tensor = model(text_X, pos_X, tfidf_X, sentiment_X, embed_X)
            
            loss: torch.Tensor = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * text_X.size(0)
        
        train_loss /= len(train_loader.dataset)

        # --- Validation Phase ---
        model.eval()
        val_loss: float = 0.0
        all_preds: List[int] = []
        all_targets: List[int] = []
        
        with torch.no_grad():
            for text_X, pos_X, tfidf_X, sentiment_X, embed_X, y_batch in test_loader:
                text_X = text_X.to(device).long()
                pos_X = pos_X.to(device).long()
                tfidf_X = tfidf_X.to(device).float()
                sentiment_X = sentiment_X.to(device).float()
                embed_X = embed_X.to(device).float()
                y_batch = y_batch.to(device).long()
                
                # Forward pass with 5 inputs
                outputs: torch.Tensor = model(text_X, pos_X, tfidf_X, sentiment_X, embed_X)
                
                loss: torch.Tensor = criterion(outputs, y_batch)
                val_loss += loss.item() * text_X.size(0)
                
                preds: np.ndarray = outputs.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(y_batch.cpu().numpy())
                
        val_loss /= len(test_loader.dataset)
        val_accuracy: float = accuracy_score(all_targets, all_preds)

        print(f"Epoch {epoch+1:02d}/{NUM_EPOCHS}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_accuracy:.4f}")

        # --- Early Stopping Logic ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
            print("   -> Best model saved.")
        else:
            patience_counter += 1
            print(f"   -> Validation loss did not improve. Patience: {patience_counter}/{PATIENCE}")
            if patience_counter >= PATIENCE:
                print(f"Early stopping triggered after {PATIENCE} epochs without improvement.")
                break

    # Load the weights from the epoch that yielded the best validation loss
    model.load_state_dict(best_model_state)

    # --- Evaluation (Updated) ---
    print("\n--- Final Evaluation on Test Set (Multi-Feature) ---")
    model.eval()
    all_preds_test: List[int] = []
    
    with torch.no_grad():
        for text_X, pos_X, tfidf_X, sentiment_X, embed_X, _ in test_loader:
            text_X = text_X.to(device).long()
            pos_X = pos_X.to(device).long()
            tfidf_X = tfidf_X.to(device).float()
            sentiment_X = sentiment_X.to(device).float()
            embed_X = embed_X.to(device).float()

            # Forward pass with 5 inputs
            outputs: torch.Tensor = model(text_X, pos_X, tfidf_X, sentiment_X, embed_X)
            
            preds: np.ndarray = outputs.argmax(dim=1).cpu().numpy()
            all_preds_test.extend(preds)

    # Calculate final metrics
    test_targets_np: np.ndarray = test_labels.numpy()
    all_preds_test_np: np.ndarray = np.array(all_preds_test)

    accuracy: float = accuracy_score(test_targets_np, all_preds_test_np)
    f1: float = f1_score(test_targets_np, all_preds_test_np, average='weighted')

    print("--- Results ---")
    print(f"Final Test Accuracy: {accuracy:.4f}")
    print(f"Final Test F1 Score (Weighted): {f1:.4f}")

    print("\nClassification Report:")
    report: str = classification_report(
        test_targets_np, 
        all_preds_test_np, 
        target_names=label_encoder.classes_,
        digits=4
    )
    print(report)

    print("--- Evaluation Complete ---")