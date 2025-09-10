import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import re

# Loading Excel file
df = pd.read_excel(r"C:\Users\koenm\Documents\repositorys\year_2\fae2-nlpr-group-group-4-1\fae2-nlpr-group-group-4-1\Transcriptions\NLP_features_output.xlsx")
sentences = df['Sentence'].tolist()

# TF-IDF part. Finally working!!!!
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

# debugging part made by claude AI
test_words = ['penningmeesterschap', 'cécile', 'wateroppervlak', 'overhaal', 'stemt']
print("\nChecking for specific words in vocabulary:")
for word in test_words:
    if word.lower() in [v.lower() for v in vocab]:
        print(f"✅ Found: {word}")
    else:
        print(f"❌ Missing: {word}")

# Create readable format
tfidf_readable = []
for i in range(len(sentences)):
    sentence = sentences[i]
    vector = tfidf_matrix[i].toarray()[0]
    
    # Get ALL words with their scores
    all_scores = [(vocab[j], score) for j, score in enumerate(vector) if score > 0]
    all_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Format top words
    if all_scores:
        formatted = [f"{word}({score:.3f})" for word, score in all_scores[:8]]
        result = "; ".join(formatted)
    else:
        result = "EMPTY"
    
    tfidf_readable.append(result)

# Add to dataframe
df['TF_IDF'] = tfidf_readable

# Save
df.to_excel('output_final_tfidf.xlsx', index=False)

print(f"\nProcessed {len(df)} sentences")
print("\nFirst 8 examples:")
for i in range(min(8, len(df))):
    print(f"\n{i+1}. Sentence: {df.iloc[i]['Sentence']}")
    print(f"   TF-IDF: {df.iloc[i]['TF_IDF']}")
    
    # Check if we're getting the unique words
    sentence = df.iloc[i]['Sentence'].lower()
    tfidf_result = df.iloc[i]['TF_IDF']
    
    # Look for long/unique words in the sentence
    unique_words = re.findall(r'\b[a-záéíóúàèìòùâêîôûäëïöüç]{6,}\b', sentence)
    if unique_words:
        print(f"   Long words in sentence: {unique_words}")
        for word in unique_words:
            if word in tfidf_result.lower():
                print(f"   ✅ {word} found in TF-IDF")
            else:
                print(f"   ❌ {word} MISSING from TF-IDF")

# Final stats made by Claude AI
empty_count = sum(1 for x in tfidf_readable if x == "EMPTY")
print(f"\nFinal stats:")
print(f"Empty results: {empty_count}/{len(df)}")
print(f"Success rate: {(len(df)-empty_count)/len(df)*100:.1f}%")