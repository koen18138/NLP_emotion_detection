import re
import pandas as pd

def split_into_sentences(text):
    """Split text into sentences using regex patterns."""
    
    # Remove the header information (everything before "Transcription:")
    if "Transcription:" in text:
        text = text.split("Transcription:", 1)[1].strip()
    
    # Split on sentence endings, but be careful with abbreviations and numbers
    # This regex looks for periods, exclamation marks, or question marks
    # followed by whitespace and a capital letter (or end of string)
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    
    # Keep all sentences, just strip whitespace
    cleaned_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence:  # Only remove completely empty strings
            cleaned_sentences.append(sentence)
    
    return cleaned_sentences

def create_excel_file(input_file, output_file):
    """Read transcription file and create Excel with sentences."""
    
    # Read the transcription file
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Split into sentences
    sentences = split_into_sentences(text)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Sentence_Number': range(1, len(sentences) + 1),
        'Sentence': sentences,
        'Speaker': [''] * len(sentences),  # Empty column for you to fill in
        'Timestamp': [''] * len(sentences),  # Empty column for timestamps if needed
        'Notes': [''] * len(sentences)  # Empty column for notes
    })
    
    # Save to Excel
    df.to_excel(output_file, index=False, engine='openpyxl')
    
    print(f"Created Excel file: {output_file}")
    print(f"Total sentences: {len(sentences)}")
    print("\nFirst 5 sentences:")
    for i in range(min(5, len(sentences))):
        print(f"{i+1}: {sentences[i][:100]}...")

# Usage
input_file = "/Users/florislokhorst/Desktop/transcript(cool).txt"
output_file = "/Users/florislokhorst/Desktop/transcription_sentences_whisper.xlsx"

create_excel_file(input_file, output_file)