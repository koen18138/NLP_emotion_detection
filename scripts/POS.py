import spacy
import pandas as pd
import numpy as np

# Configuration
CONFIG = {
    'spacy_model': 'nl_core_news_lg',
    'language': 'dutch',
    'text_column': 'Sentence',  
    'output_file': 'fae2-nlpr-group-group-4-1/Transcriptions/NLP_features_output.xlsx'
}

nlp = spacy.load(CONFIG['spacy_model'])
data = pd.read_excel("fae2-nlpr-group-group-4-1/Transcriptions/transcribed_data_assemblyAI.xlsx")

def extract_pos_tags(text: str, nlp_model) -> str:
    """Extract POS tags from text."""
    try:
        if not text or text.strip() == '':
            return ''
        
        doc = nlp_model(text.strip())
        pos_tags = [f"{token.text}_{token.pos_}" for token in doc]
        return ' '.join(pos_tags)
    
    except Exception as e:
        print(f"Error processing POS tags for text: '{text[:50]}...': {e}")
        return ''

print("Extracting POS tags for all sentences...")
data['POS_Tags'] = data[CONFIG['text_column']].apply(lambda x: extract_pos_tags(x, nlp))
print("POS tags extraction completed.")
print("Saving results to Excel...")
data.to_excel(CONFIG['output_file'], index=False)
print(f"Results saved to {CONFIG['output_file']}")
print("NLP feature extraction completed.")
