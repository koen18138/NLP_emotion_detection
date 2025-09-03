import re
import pandas as pd

import whisper
import os

def transcribe_mp3(mp3_path):
    """
    Transcribe an MP3 file using Whisper
    """
    # Check if file exists
    if not os.path.exists(mp3_path):
        print(f"Error: File '{mp3_path}' not found!")
        return None
    
    print("Loading Whisper model...")
    # Load Whisper model (base is good balance of speed/accuracy)
    model = whisper.load_model("base")
    
    print(f"Transcribing: {os.path.basename(mp3_path)}")
    print("This may take a few minutes...")
    
    # Transcribe the audio
    result = model.transcribe(mp3_path)
    
    return result["text"]

# Main usage
if __name__ == "__main__":
    # Put your MP3 file path here
    mp3_file = r"C:\Users\koenm\Documents\test\Wie_is_de_Mol_(The Mole)_S16E05.mp3"
    
    # Transcribe
    transcript = transcribe_mp3(mp3_file)
    
    if transcript:
        # Print result
        print("\n" + "="*60)
        print("TRANSCRIPT:")
        print("="*60)
        print(transcript)
        print("="*60)
        
        # Save to file
        output_file = "transcript.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(transcript)
        
        print(f"\nTranscript saved to: {output_file}")
    else:
        print("Transcription failed!")

def split_into_sentences(text):
    """Split text into sentences using regex patterns."""
    
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

def transcript_to_excel(transcript_text, output_excel_path):
    """Convert transcript text to Excel with sentences."""
    
    print("Splitting transcript into sentences...")
    
    # Split into sentences
    sentences = split_into_sentences(transcript_text)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Sentence_Number': range(1, len(sentences) + 1),
        'Sentence': sentences,
        'Speaker': [''] * len(sentences),  # Empty column for you to fill in
        'Timestamp': [''] * len(sentences),  # Empty column for timestamps if needed
        'Notes': [''] * len(sentences)  # Empty column for notes
    })
    
    # Save to Excel
    df.to_excel(output_excel_path, index=False, engine='openpyxl')
    
    print(f"Excel file created: {output_excel_path}")
    print(f"Total sentences: {len(sentences)}")
    
    # Show first few sentences as preview
    print(f"\nFirst 3 sentences:")
    for i in range(min(3, len(sentences))):
        print(f"{i+1}: {sentences[i][:80]}...")
    
    return df

# Convert the transcript to Excel
if 'transcript' in locals() and transcript:
    excel_output_path = "transcript_sentences.xlsx"
    df_sentences = transcript_to_excel(transcript, excel_output_path)
    print(f"\nSuccess! Your transcript has been split into {len(df_sentences)} sentences.")
else:
    print("No transcript found. Please run the transcription cell first.")