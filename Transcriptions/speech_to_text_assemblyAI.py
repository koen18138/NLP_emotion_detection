import assemblyai as aai
import re
import pandas as pd
import os

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

def transcribe_and_create_excel(audio_file_path, api_key, output_excel_path, output_text_path=None):
    """Complete pipeline: transcribe audio and create Excel with sentences."""
    
    # Set up AssemblyAI
    aai.settings.api_key = api_key
    
    print("Starting transcription...")
    
    # Check if file exists
    if not os.path.exists(audio_file_path):
        print(f"Error: Audio file not found at {audio_file_path}")
        return
    
    # Configure transcription
    config = aai.TranscriptionConfig(
        speech_model=aai.SpeechModel.universal,
        language_code="nl",
    )
    
    # Transcribe
    transcript = aai.Transcriber(config=config).transcribe(audio_file_path)
    
    if transcript.status == "error":
        print(f"Transcription failed: {transcript.error}")
        return
    
    print(f"Transcription completed!")
    print(f"Duration: {transcript.audio_duration}ms")
    print(f"Confidence: {transcript.confidence}")
        
    # Save full transcription to text file if requested
    if output_text_path:
        with open(output_text_path, "w", encoding="utf-8") as f:
            f.write(f"Audio file: {audio_file_path}\n")
            f.write(f"Confidence: {transcript.confidence}\n")
            f.write(f"Duration: {transcript.audio_duration}ms\n\n")
            f.write("Transcription:\n")
            f.write(transcript.text)
        print(f"Full transcription saved to: {output_text_path}")
    
    # Split into sentences
    print("Splitting into sentences...")
    sentences = split_into_sentences(transcript.text)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Sentence_Number': range(1, len(sentences) + 1),
        'Sentence': sentences,
        'Speaker': [''] * len(sentences),  # Empty column for speaker
        'Timestamp': [''] * len(sentences),  # Empty column for timestamps
        'Notes': [''] * len(sentences)  # Empty column for notes
    })
    
    # Save to Excel
    df.to_excel(output_excel_path, index=False, engine='openpyxl')
    
    print(f"\nExcel file created: {output_excel_path}")
    print(f"Total sentences: {len(sentences)}")
    
    # Show first few sentences as preview
    print(f"\nFirst 3 sentences:")
    for i in range(min(3, len(sentences))):
        print(f"{i+1}: {sentences[i][:80]}...")
    
    return transcript, df

# Configuration
API_KEY = "ae9d26f45c504101be8279aebb10a8f7"
AUDIO_FILE = r"/Applications/Mp3/Wie is de Mol (The Mole) S16E05 with English subtitles.mp3"
OUTPUT_EXCEL = "/Users/florislokhorst/Desktop/transcribed_data_assemblyAI.xlsx"
OUTPUT_TEXT = "/Users/florislokhorst/Desktop/transcription_full.txt"

# Run the pipeline
if __name__ == "__main__":
    transcript, df = transcribe_and_create_excel(
        audio_file_path=AUDIO_FILE,
        api_key=API_KEY,
        output_excel_path=OUTPUT_EXCEL,
        output_text_path=OUTPUT_TEXT  # Set to None if you don't want the text file
    )