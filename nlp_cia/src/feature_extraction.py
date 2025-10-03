import spacy
from spacy_ngram import NgramComponent
from spacytextblob.spacytextblob import SpacyTextBlob
import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import re

@spacy.language.Language.component('discard_stopwords')
def discard_stopwords(doc):
    stopwords = spacy.lang.nl.stop_words.STOP_WORDS
    filtered = filter(lambda x: x.text not in stopwords, doc)
    filtered = filter(lambda x: not x.is_punct, filtered)
    filtered = filter(lambda x: x.is_ascii, filtered)
    filtered = list(map(lambda x: x.text, filtered))
    doc = spacy.tokens.Doc(doc.vocab, words=filtered)
    return doc

# Load the merged data
df = pd.read_csv('data/dataset/processed/go_emotion_dutch.csv')

# TF-IDF extraction (from notebook)
sentences = df['Sentence'].astype(str).tolist()
vectorizer = TfidfVectorizer(
    max_features=2000,
    min_df=1,
    max_df=1.0,
    lowercase=True,
    stop_words=None,
    token_pattern=r'\b[a-záéíóúàèìòùâêîôûäëïöüç]+\b',
)
tfidf_matrix = vectorizer.fit_transform(sentences)
vocab = vectorizer.get_feature_names_out()

tfidf_vectors = []
for idx, sentence in tqdm(enumerate(sentences), desc="TF-IDF"):
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

df['TF_IDF'] = tfidf_vectors

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

df['Embedding'] = df['Sentence'].apply(sent_embedding_spacy)
embeddings_array = np.stack(df['Embedding'].values)

# Run the pipeline for all sentences
sent_list = df['Sentence'].to_list()
doc_list = list(nlp.pipe(sent_list))

features = {
    'doc_entities': [],
    'doc_noun_chunks': [],
    'doc_2_grams': [],
    'doc_3_grams': [],
    'doc_sentiment': [],
    'doc_subjectivity': [],
    'doc_sentiment': [],
    'POS_Tags': [],
    'token_tag': [],
    'token_lemmatized': [],
    'token_normalized': [],
    'token_dependancy': [],
    'token_polarity': [],
    'token_subjectivity': []
}

for doc in tqdm(doc_list, desc="spaCy features"):
    features['doc_entities'].append(list(map(lambda x: x.as_doc(), doc.ents)))
    features['doc_noun_chunks'].append(list(map(lambda x: x.as_doc(), doc.noun_chunks)))
    features['doc_2_grams'].append(doc._.ngram_2)
    features['doc_3_grams'].append(doc._.ngram_3)
    features['doc_sentiment'].append(doc._.blob.polarity)
    features['doc_subjectivity'].append(doc._.blob.subjectivity)
    features['POS_Tags'].append(list(map(lambda x: x.pos_, doc)))
    features['token_tag'].append(list(map(lambda x: x.tag_, doc)))
    features['token_lemmatized'].append(list(map(lambda x: x.lemma_, doc)))
    features['token_normalized'].append(list(map(lambda x: x.norm_, doc)))
    features['token_dependancy'].append(list(map(lambda x: x.dep_, doc)))
    features['token_polarity'].append(list(map(lambda x: x._.blob.polarity, doc)))
    features['token_subjectivity'].append(list(map(lambda x: x._.blob.subjectivity, doc)))
    
features_df = pd.DataFrame(features)
features_df = pd.concat([features_df, pd.Series(map(lambda x: x.text, doc_list))], axis=1, ignore_index=True)
col_names = list(features.keys())
col_names.append('sentence_clean')
features_df.columns = col_names

full_df = pd.concat([df, features_df], axis=1)

full_df = full_df.assign(
    simple_emotion = full_df['Emotion_core'].map(lambda x: 'happiness' if x == 'happiness' else 'other')
)

def list_to_string(col):
    return col.astype(str).map(lambda x: " ".join(x.replace(',', '').replace("'", "").split())[1:-1])

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

full_df.to_csv('data/features/NLP_features.csv', index=False)