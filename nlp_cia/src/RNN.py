import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score, accuracy_score, classification_report
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from collections import Counter
from itertools import chain

# --- Data Preparation (similar to your Keras code) ---

MAX_VOCAB_SIZE = 5000
MAX_LENGTH = 30
BATCH_SIZE = 32

# Load your data
df_train = pd.read_parquet(r'data/features/go_emotion_eng_features.parquet')
df_test = pd.read_parquet(r'data/features/test_improved_features.parquet')

train_texts = df_train['Sentence'].astype(str).tolist()
test_texts = df_test['Sentence'].astype(str).tolist()

# Build vocabulary
def build_vocab(texts, max_words=5000):
    counter = Counter(chain.from_iterable(t.split() for t in texts))
    most_common = counter.most_common(max_words)
    word2idx = {word: idx+2 for idx, (word, _) in enumerate(most_common)}
    word2idx["<PAD>"] = 0
    word2idx["<OOV>"] = 1
    return word2idx

word2idx = build_vocab(train_texts + test_texts, max_words=MAX_VOCAB_SIZE)

def texts_to_sequences(texts, word2idx):
    oov_token_idx = word2idx["<OOV>"]
    return [
        [word2idx.get(word, oov_token_idx) for word in text.split()]
        for text in texts
    ]

train_sequences = texts_to_sequences(train_texts, word2idx)
test_sequences = texts_to_sequences(test_texts, word2idx)

def pad_sequences_torch(sequences, maxlen, padding_value=0):
    padded_tensors = []
    for seq in sequences:
        seq_tensor = torch.tensor(seq, dtype=torch.long)
        if len(seq_tensor) > maxlen:
            seq_tensor = seq_tensor[:maxlen]
        padding_needed = maxlen - len(seq_tensor)
        if padding_needed > 0:
            padding = torch.full((padding_needed,), padding_value, dtype=torch.long)
            padded_seq = torch.cat([seq_tensor, padding])
        else:
            padded_seq = seq_tensor
        padded_tensors.append(padded_seq)
    return torch.stack(padded_tensors)

train_padded = pad_sequences_torch(train_sequences, MAX_LENGTH, padding_value=word2idx["<PAD>"])
test_padded = pad_sequences_torch(test_sequences, MAX_LENGTH, padding_value=word2idx["<PAD>"])

# --- POS Tag Processing ---
# Build POS vocab
all_pos_tags = list(chain.from_iterable(df_train['POS_Tags'].tolist() + df_test['POS_Tags'].tolist()))
pos_counter = Counter(all_pos_tags)
pos_word2idx = {tag: idx + 2 for idx, (tag, _) in enumerate(pos_counter.most_common())}
pos_word2idx["<PAD>"] = 0
pos_word2idx["<OOV>"] = 1
pos_vocab_size = len(pos_word2idx)

def pos_to_sequences(pos_lists, pos_word2idx):
    oov_idx = pos_word2idx["<OOV>"]
    return [[pos_word2idx.get(tag, oov_idx) for tag in pos_list] for pos_list in pos_lists]

train_pos_sequences = pos_to_sequences(df_train['POS_Tags'].tolist(), pos_word2idx)
test_pos_sequences = pos_to_sequences(df_test['POS_Tags'].tolist(), pos_word2idx)
train_pos_padded = pad_sequences_torch(train_pos_sequences, MAX_LENGTH, padding_value=pos_word2idx["<PAD>"])
test_pos_padded = pad_sequences_torch(test_pos_sequences, MAX_LENGTH, padding_value=pos_word2idx["<PAD>"])

# --- TF-IDF, Sentiment, and Static Embedding Tensors ---
def pad_feature_vectors(vectors, max_len=128):
    # If max_len is not given, use the length of the first vector
    if max_len is None:
        max_len = len(vectors[0])
    padded_vectors = []
    for vector in vectors:
        vector = np.array(vector)
        if len(vector) < max_len:
            padding_needed = max_len - len(vector)
            padded_vector = np.pad(vector, (0, padding_needed), 'constant', constant_values=(0.0, 0.0))
        else:
            padded_vector = vector
        padded_vectors.append(padded_vector)
    return np.stack(padded_vectors)

train_tfidf_padded = pad_feature_vectors(df_train['TF_IDF'].values)
test_tfidf_padded = pad_feature_vectors(df_test['TF_IDF'].values)
tfidf_dim = train_tfidf_padded.shape[1]

train_embed_np = np.stack(df_train['Embedding'].values)
test_embed_np = np.stack(df_test['Embedding'].values)
static_embed_dim = train_embed_np.shape[1]

# Normalize features
scaler_tfidf = StandardScaler()
train_tfidf_padded = scaler_tfidf.fit_transform(train_tfidf_padded)
test_tfidf_padded = scaler_tfidf.transform(test_tfidf_padded)

scaler_embed = StandardScaler()
train_embed_np = scaler_embed.fit_transform(train_embed_np)
test_embed_np = scaler_embed.transform(test_embed_np)

scaler_sent = StandardScaler()
train_sentiment_np = scaler_sent.fit_transform(df_train['Sentiment_Score'].values.reshape(-1, 1)).flatten()
test_sentiment_np = scaler_sent.transform(df_test['Sentiment_Score'].values.reshape(-1, 1)).flatten()

# Convert to tensors
train_tfidf = torch.tensor(train_tfidf_padded, dtype=torch.float)
test_tfidf = torch.tensor(test_tfidf_padded, dtype=torch.float)
train_sentiment = torch.tensor(train_sentiment_np, dtype=torch.float)
test_sentiment = torch.tensor(test_sentiment_np, dtype=torch.float)
train_embed = torch.tensor(train_embed_np, dtype=torch.float)
test_embed = torch.tensor(test_embed_np, dtype=torch.float)

# Labels
label_encoder = LabelEncoder()
label_encoder.fit(df_train['Emotion_core'].tolist() + df_test['Emotion_core'].tolist())
train_labels = torch.tensor(label_encoder.transform(df_train['Emotion_core'].tolist()), dtype=torch.long)
test_labels = torch.tensor(label_encoder.transform(df_test['Emotion_core'].tolist()), dtype=torch.long)
num_classes = len(label_encoder.classes_)

# --- PyTorch Dataset and DataLoader ---
class MultiFeatureTextDataset(Dataset):
    def __init__(self, X, pos, tfidf, sentiment, embed, y):
        self.X = X
        self.pos = pos
        self.tfidf = tfidf
        self.sentiment = sentiment
        self.embed = embed
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return (self.X[idx], self.pos[idx], self.tfidf[idx], self.sentiment[idx], self.embed[idx], self.y[idx])

train_dataset = MultiFeatureTextDataset(train_padded, train_pos_padded, train_tfidf, train_sentiment, train_embed, train_labels)
test_dataset = MultiFeatureTextDataset(test_padded, test_pos_padded, test_tfidf, test_sentiment, test_embed, test_labels)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# --- PyTorch Multi-Feature LSTM Model ---
class MultiFeatureLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes,
                 pos_vocab_size, pos_embed_dim, tfidf_dim, static_embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm1 = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(hidden_dim * 2, hidden_dim, batch_first=True, bidirectional=True)
        self.pos_embedding = nn.Embedding(pos_vocab_size, pos_embed_dim, padding_idx=0)
        self.final_input_dim = hidden_dim * 2 + pos_embed_dim + tfidf_dim + 1 + static_embed_dim
        self.fc = nn.Sequential(
            nn.Linear(self.final_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x, pos_tags, tfidf, sentiment, embedding):
        # Word embedding + LSTM
        x_embed = self.embedding(x)
        lstm_out1, _ = self.lstm1(x_embed)
        lstm_out2, (h_n, _) = self.lstm2(lstm_out1)
        lstm_out = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
        # POS embedding + mean pooling
        pos_embedded = self.pos_embedding(pos_tags)
        mask = (pos_tags != 0).unsqueeze(-1).float()
        pos_embedded_masked = pos_embedded * mask
        pos_sum = pos_embedded_masked.sum(dim=1)
        seq_len = mask.sum(dim=1).clamp(min=1)
        pos_mean_pooled = pos_sum / seq_len.float()
        # Sentiment reshape
        sentiment_reshaped = sentiment.float().unsqueeze(1)
        # Concatenate all features
        features = torch.cat([lstm_out, pos_mean_pooled, tfidf.float(), sentiment_reshaped, embedding.float()], dim=1)
        return self.fc(features)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMBED_DIM = 64
POS_EMBED_DIM = 16
HIDDEN_DIM = 32

model = MultiFeatureLSTMClassifier(
    vocab_size=len(word2idx),
    embed_dim=EMBED_DIM,
    hidden_dim=HIDDEN_DIM,
    num_classes=num_classes,
    pos_vocab_size=pos_vocab_size,
    pos_embed_dim=POS_EMBED_DIM,
    tfidf_dim=tfidf_dim,
    static_embed_dim=static_embed_dim
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# --- Training Loop ---
EPOCHS = 50
patience = 5
best_val_loss = float('inf')
patience_counter = 0
best_model_state = None

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for X_batch, pos_batch, tfidf_batch, sent_batch, embed_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        pos_batch = pos_batch.to(device)
        tfidf_batch = tfidf_batch.to(device)
        sent_batch = sent_batch.to(device)
        embed_batch = embed_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch, pos_batch, tfidf_batch, sent_batch, embed_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X_batch.size(0)
    avg_loss = total_loss / len(train_loader.dataset)

    # --- Validation Loss ---
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_batch, pos_batch, tfidf_batch, sent_batch, embed_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            pos_batch = pos_batch.to(device)
            tfidf_batch = tfidf_batch.to(device)
            sent_batch = sent_batch.to(device)
            embed_batch = embed_batch.to(device)
            y_batch = y_batch.to(device)
            outputs = model(X_batch, pos_batch, tfidf_batch, sent_batch, embed_batch)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item() * X_batch.size(0)
    avg_val_loss = val_loss / len(test_loader.dataset)
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f} - Val Loss: {avg_val_loss:.4f}")

    # --- Early Stopping Check ---
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        best_model_state = model.state_dict()
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

# Restore best model
if best_model_state is not None:
    model.load_state_dict(best_model_state)

# --- Evaluation ---
model.eval()
all_preds = []
all_targets = []
with torch.no_grad():
    for X_batch, pos_batch, tfidf_batch, sent_batch, embed_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        pos_batch = pos_batch.to(device)
        tfidf_batch = tfidf_batch.to(device)
        sent_batch = sent_batch.to(device)
        embed_batch = embed_batch.to(device)
        outputs = model(X_batch, pos_batch, tfidf_batch, sent_batch, embed_batch)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(y_batch.numpy())

accuracy = accuracy_score(all_targets, all_preds)
f1 = f1_score(all_targets, all_preds, average='weighted')

print(f"\nPyTorch MultiFeatureLSTM Results:")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print("\nClassification Report:")
print(classification_report(all_targets, all_preds, target_names=label_encoder.classes_))