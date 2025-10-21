import spacy
from spacy_ngram import NgramComponent
from spacytextblob.spacytextblob import SpacyTextBlob
import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import re
import torch
from transformers import BertTokenizer, BertForSequenceClassification, EvalPrediction, PreTrainedTokenizerFast
from sklearn.metrics import accuracy_score, f1_score
from bs4 import BeautifulSoup
from typing import Dict, List, Tuple
import os

def load_df(filepath: str, text_column: str = "Sentence") -> pd.DataFrame:
    """Function to load a pd.DataFrame from a given filepath"""
    # Try to read filepath as csv or excel dropping any empty sentence rows
    try:
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath).dropna(subset=[text_column]) 
            # Shuffle the DataFrame for randomness
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        elif filepath.endswith('.xlsx'):
            df = pd.read_excel(filepath).dropna(subset=[text_column])
            # Shuffle the DataFrame for randomness
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        else:
            raise ValueError("Unsupported file format. Please use .csv or .xlsx")
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")
    return df

def save_df_csv(df: pd.DataFrame, directory: str, filename: str):
    """"Function to save a Pandas dataframe as CSV"""

    # Check if output directory exists otherwise make it
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Create filepath
    filepath = os.path.join(directory, filename)
    
    # Save the DataFrame to a CSV file
    df.to_csv(filepath, index=False)
    print(f"Successfully processed {df.shape} and saved the output to {filepath}")

@spacy.language.Language.component('discard_stopwords')
def discard_stopwords(doc):
    stopwords = spacy.lang.nl.stop_words.STOP_WORDS
    filtered = filter(lambda x: x.text not in stopwords, doc)
    filtered = filter(lambda x: not x.is_punct, filtered)
    filtered = filter(lambda x: x.is_ascii, filtered)
    filtered = list(map(lambda x: x.text, filtered))
    doc = spacy.tokens.Doc(doc.vocab, words=filtered)
    return doc

def load_fit_tfidf(
    sentences: List[str], 
) -> Tuple[np.ndarray, List[str]]:
    """Load and fit a TF-IDF vectorizer on the provided sentences.
    
    Args:
        sentences (List[str]): List of sentences to fit the TF-IDF vectorizer.
    
    Authors: Koen Matthijssen, Nick Belterman
    """
    # Initialize TF-IDF vectorizer class
    vectorizer = TfidfVectorizer(
        max_features=2000,         
        min_df=1,                    
        max_df=1.0,                  
        lowercase=True,
        stop_words=None,             
        token_pattern=r'\b[a-záéíóúàèìòùâêîôûäëïöüç]+\b',  
    )
    
    print("Fitting TF-IDF...")
    tfidf_matrix = vectorizer.fit_transform(sentences)
    vocab = vectorizer.get_feature_names_out()
    print(f"Total vocabulary: {len(vocab)}")
    return tfidf_matrix, vocab

def create_tfidf_dataframe(
    df: pd.DataFrame,
    text_column: str = 'Sentence',
) -> pd.DataFrame:
    """Create a DataFrame from the TF-IDF matrix and vocabulary.
    
    Returns:
        pd.DataFrame: DataFrame with TF-IDF features.
    
    Authors: Koen Matthijssen, Nick Belterman
    """

    sentences = df[text_column]

    # Get TF-IDF matrix and vocabulary
    tfidf_matrix, vocab = load_fit_tfidf(sentences)

    # Convert to numerical lists format - only values > 0 in sentence order
    tfidf_vectors = []
    for idx, sentence in enumerate(sentences):
        vector = tfidf_matrix[idx].toarray()[0]
        sentence_words = []
        for word in sentence.lower().split():
            clean_word = re.sub(r'[^\wa-záéíóúàèìòùâêîôûäëïöüç]', '', word)
            if clean_word and clean_word in vocab:
                sentence_words.append(clean_word)
    
        sentence_order_values = []
        for word in sentence_words:
            if word in vocab:
                word_index = list(vocab).index(word)
                tfidf_score = vector[word_index]
                if tfidf_score > 0:
                    sentence_order_values.append(round(float(tfidf_score), 3))
    
        tfidf_vectors.append(sentence_order_values)

    print(f"{len(tfidf_vectors)} TF-IDF vectors created.")
    print(f"{df.shape[0]} rows in dataframe.")
    print(list(zip(sentence_words, sentence_order_values)))

    # Add to dataframe
    df['TF_IDF'] = tfidf_vectors

    # Dropna
    df.dropna(subset=[text_column], inplace=True)

    return df

# Load spaCy language model
nlp = spacy.load('nl_core_news_lg')
vec_size = nlp.vocab.vectors_length

# Add doc cleaning function to the pipeline
nlp.add_pipe('discard_stopwords', first=True)
nlp.add_pipe('spacy-ngram', config={'sentence_level': False, 'doc_level': True, 'ngrams': (2, 3)})
nlp.add_pipe('spacytextblob')

# Compute spaCy embeddings (from notebook)
def sent_embedding_spacy(text: str) -> np.ndarray:
    doc = nlp(text)
    vecs = [t.vector for t in doc if t.has_vector]
    return np.mean(vecs, axis=0) if vecs else np.zeros(vec_size, dtype=np.float32)
def get_nlp_features_spacy(df: pd.DataFrame, nlp):
    # Run the pipeline for all sentences
    sent_list = df['Sentence'].to_list()
    doc_list = list(nlp.pipe(sent_list))

    features = {
        'doc_entities': [],
        'doc_noun_chunks': [],
        'doc_2_grams': [],
        'doc_3_grams': [],
        # 'doc_sentiment': [],
        'doc_subjectivity': [],
        'POS_Tags': [],
        'token_tag': [],
        'token_lemmatized': [],
        'token_normalized': [],
        'token_dependancy': [],
        # 'token_polarity': [],
        # 'token_subjectivity': []
    }

    for doc in tqdm(doc_list, desc="spaCy features"):
        features['doc_entities'].append(list(map(lambda x: x.as_doc(), doc.ents)))
        features['doc_noun_chunks'].append(list(map(lambda x: x.as_doc(), doc.noun_chunks)))
        features['doc_2_grams'].append(doc._.ngram_2)
        features['doc_3_grams'].append(doc._.ngram_3)
        # features['doc_sentiment'].append(doc._.blob.polarity)
        features['doc_subjectivity'].append(doc._.blob.subjectivity)
        features['POS_Tags'].append(list(map(lambda x: x.pos_, doc)))
        features['token_tag'].append(list(map(lambda x: x.tag_, doc)))
        features['token_lemmatized'].append(list(map(lambda x: x.lemma_, doc)))
        features['token_normalized'].append(list(map(lambda x: x.norm_, doc)))
        features['token_dependancy'].append(list(map(lambda x: x.dep_, doc)))
        # features['token_polarity'].append(list(map(lambda x: x._.blob.polarity, doc)))
        # features['token_subjectivity'].append(list(map(lambda x: x._.blob.subjectivity, doc)))
        
    # Sanity check: make sure all lists in `features` are the same length
    lengths = {k: len(v) for k, v in features.items()}
    print("Feature lengths:", lengths)
    try:
        features_df = pd.DataFrame(features)
    except ValueError:
        expected_length = len(doc_list)
        for k, v in features.items():
            if len(v) != expected_length:
                print(f"⚠️ Feature '{k}' has length {len(v)} instead of {expected_length}")    
    features_df = pd.concat([features_df, pd.Series(map(lambda x: x.text, doc_list))], axis=1, ignore_index=True)
    col_names = list(features.keys())
    col_names.append('sentence_clean')
    features_df.columns = col_names

    full_df = pd.concat([df, features_df], axis=1)

    full_df = full_df.assign(
        simple_emotion = full_df['Emotion_core'].map(lambda x: 'happiness' if x == 'happiness' else 'other')
    )
    return full_df

def text_cleaning(text: str) -> str:
    """
    Cleans a text string by removing HTML tags, bracketed content,
    and non-alphanumeric/non-whitespace characters (except commas and apostrophes).

    Args:
        text: The input string to be cleaned.

    Returns:
        The cleaned string.
    """
    # Remove HTML tags using BeautifulSoup
    soup = BeautifulSoup(text, "html.parser")
    # Remove content within brackets, e.g., "[...]"
    text = re.sub(r'\[[^]]*\]', '', soup.get_text())
    # Define a regex pattern to remove unwanted characters, keeping only
    # letters, numbers, whitespace, commas, and apostrophes
    pattern = r"[^a-zA-Z0-9\s,']"
    text = re.sub(pattern, '', text)
    return text

def compute_metrics(p: EvalPrediction) -> dict:
    """
    Computes accuracy and F1 score for model evaluation.

    Args:
        p: An EvalPrediction object containing model predictions and labels.

    Returns:
        A dictionary with 'accuracy' and 'f1' scores.
    """
    # Get the predicted class by finding the index of the max logit
    preds = np.argmax(p.predictions, axis=1)
    print(f"{preds = }")
    print(f"{p.label_ids = }")
    # Calculate accuracy and F1 score
    return {
        'accuracy': accuracy_score(p.label_ids, preds),
        'f1': f1_score(p.label_ids, preds, average='weighted'),
    }

def encode_data(texts: list, tokenizer: PreTrainedTokenizerFast, labels: list = None, max_len: int = 128) -> dict:
    """
    Encodes a list of texts into token IDs, attention masks, and optionally labels.

    Args:
        texts: A list of text strings to be encoded.
        tokenizer: The tokenizer object (e.g., from Hugging Face Transformers).
        labels: An optional list of corresponding labels for the texts. Can be integers or strings.
        max_len: The maximum sequence length for tokenization.

    Returns:
        A dictionary containing PyTorch tensors for 'input_ids', 'attention_mask',
        and 'labels' (if provided).
    """

    # Tokenize the texts with truncation and padding
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_len)

    # If no labels are provided, return only the encoded text data
    if labels is None:
        return {
            'input_ids': torch.tensor(encodings['input_ids']),
            'attention_mask': torch.tensor(encodings['attention_mask'])
        }
    
    # Convert string labels (if any) to integers
    if isinstance(labels[0], str):
        label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
        labels = [label_map[label] for label in labels]

    # Return a dictionary of PyTorch tensors including labels
    return {
        'input_ids': torch.tensor(encodings['input_ids']),
        'attention_mask': torch.tensor(encodings['attention_mask']),
        'labels': torch.tensor(labels, dtype=torch.float)
    }

def list_to_string(col):
    return col.astype(str).map(lambda x: " ".join(x.replace(',', '').replace("'", "").split())[1:-1])

def load_model_and_tokenizer(model_dir: str):
    """
    Loads a fine-tuned BERT model and tokenizer from a specified directory.

    Args:
        model_dir (str): The directory where the model and tokenizer are saved.

    Returns:
        tuple: A tuple containing the loaded tokenizer and model.
    """
    print(f"Loading model and tokenizer from: {model_dir}")
    tokenizer = BertTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = BertForSequenceClassification.from_pretrained(model_dir)
    return tokenizer, model

def load_inference_data(tokenizer, inference_data: pd.DataFrame=None, file_path: str="data\\transcription\\csv\\transcription.csv") -> list:
	"""
	Loads inference data from a CSV file.

	Args:
		file_path (str): The path to the CSV file containing inference data.
	Returns:
		list: A list of sentences for inference.
	"""
	if inference_data is not None:
		df = inference_data
	else:
		df = pd.read_csv(file_path)
	sentences = df['Sentence'].apply(text_cleaning).tolist()
	encoded_sentences = encode_data(sentences, tokenizer)
	return encoded_sentences, df

def get_predictions(model: BertForSequenceClassification, encodings: Dict[str, torch.Tensor], batch_size: int = 32) -> List[int]:
    """
    Performs inference on a list of sentences using a fine-tuned BERT model in batches.

    Args:
        model (BertForSequenceClassification): The fine-tuned BERT model.
        encodings (Dict[str, torch.Tensor]): Encoded input data with 'input_ids' and 'attention_mask'.
        batch_size (int): The batch size for inference. Defaults to 32.

    Returns:
        List[int]: A list of predictions for each input sentence.
    """
    print("Tokenizing input data for inference...")

    model.eval()
    predictions = []

    input_ids = encodings['input_ids']
    attention_mask = encodings['attention_mask']

    print("Making predictions in batches...")
    with torch.no_grad():
        for i in tqdm(range(0, len(input_ids), batch_size), desc='Predicting sentiment'):
            batch_input_ids = input_ids[i:i + batch_size]
            batch_attention_mask = attention_mask[i:i + batch_size]

            # Pass the batch through the model
            outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
            logits = outputs.logits
            batch_predictions = torch.argmax(logits, dim=1).tolist()
            predictions.extend(batch_predictions)
    return predictions

def add_sentiment_scores(df: pd.DataFrame, model_dir: str):
    """
    Add sentiment scores to the DataFrame using a fine-tuned BERT model.

    Args:
        df (pd.DataFrame): The input DataFrame with a 'Sentence' column.
        model_dir (str): The directory where the fine-tuned BERT model is stored.
    """
    tokenizer, model = load_model_and_tokenizer(model_dir)
    sentences = df['Sentence'].tolist()
    encoded_sentences = encode_data(sentences, tokenizer)
    predictions = get_predictions(model, encoded_sentences)
    df['Sentiment_Score'] = predictions
# full_df = full_df.assign(
#     doc_entities = list_to_string(full_df['doc_entities']),
#     doc_2_grams = list_to_string(full_df['doc_2_grams']),
#     doc_3_grams = list_to_string(full_df['doc_3_grams']),
#     token_part_of_speech = list_to_string(full_df['token_part_of_speech']),
#     token_tag = list_to_string(full_df['token_tag']),
#     token_lemmatized = list_to_string(full_df['token_lemmatized']),
#     token_normalized = list_to_string(full_df['token_normalized']),
#     token_dependancy = list_to_string(full_df['token_dependancy'])
# )

# Function to map labels to emotion
def map_labels(labels_list):
    if len(labels_list) == 0:
        return 'neutral'
    return emotion_mapping[labels_list[0]]

if __name__ == "__main__":
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
    splits: Dict[str, str] = {
        'train': 'simplified/train-00000-of-00001.parquet',
        'validation': 'simplified/validation-00000-of-00001.parquet',
        'test': 'simplified/test-00000-of-00001.parquet'
    }
    # Base URL for the Hugging Face dataset
    HF_DATASET_BASE_URL: str = "hf://datasets/google-research-datasets/go_emotions/"
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
            # df = str_eval(df)

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