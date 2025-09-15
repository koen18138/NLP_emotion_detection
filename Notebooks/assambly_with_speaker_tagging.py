import assemblyai as aai
import re
import pandas as pd
import os

def split_into_sentences(text):
    """Split text into sentences using regex patterns."""
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    cleaned_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence:
            cleaned_sentences.append(sentence)
    return cleaned_sentences

def transcribe_and_create_excel(audio_file_path, api_key, output_excel_path, output_text_path=None):
    """Complete pipeline: transcribe audio with speaker diarization and create Excel."""
    
    # Set up AssemblyAI
    aai.settings.api_key = api_key
    
    print("Starting transcription with speaker detection...")
    
    # Check if file exists
    if not os.path.exists(audio_file_path):
        print(f"Error: Audio file not found at {audio_file_path}")
        return
    
    # Configure transcription WITH SPEAKER DIARIZATION
    config = aai.TranscriptionConfig(
        speech_model=aai.SpeechModel.universal,
        language_code="nl",
        speaker_labels=True  # This enables speaker diarization
    )
    
    # Transcribe
    transcript = aai.Transcriber(config=config).transcribe(audio_file_path)
    
    if transcript.status == "error":
        print(f"Transcription failed: {transcript.error}")
        return
    
    print(f"Transcription completed!")
    print(f"Duration: {transcript.audio_duration}ms")
    print(f"Confidence: {transcript.confidence}")
    
    # Process utterances with speaker labels
    sentences_with_speakers = []
    
    if transcript.utterances:  # Check if speaker diarization worked
        print(f"Detected {len(set(u.speaker for u in transcript.utterances))} different speakers")
        
        for utterance in transcript.utterances:
            # Split each utterance into sentences but keep speaker info
            utterance_sentences = split_into_sentences(utterance.text)
            for sentence in utterance_sentences:
                sentences_with_speakers.append({
                    'text': sentence,
                    'speaker': utterance.speaker,
                    'start_time': utterance.start / 1000,  # Convert to seconds
                    'end_time': utterance.end / 1000
                })
    else:
        # Fallback if no speaker diarization available
        print("No speaker diarization available, processing as single speaker")
        sentences = split_into_sentences(transcript.text)
        for sentence in sentences:
            sentences_with_speakers.append({
                'text': sentence,
                'speaker': 'Speaker A',
                'start_time': '',
                'end_time': ''
            })
    
    # Save full transcription with speakers to text file if requested
    if output_text_path:
        with open(output_text_path, "w", encoding="utf-8") as f:
            f.write(f"Audio file: {audio_file_path}\n")
            f.write(f"Confidence: {transcript.confidence}\n")
            f.write(f"Duration: {transcript.audio_duration}ms\n\n")
            f.write("Transcription with speakers:\n")
            f.write("-" * 50 + "\n")
            
            if transcript.utterances:
                for utterance in transcript.utterances:
                    f.write(f"\n{utterance.speaker} [{utterance.start/1000:.1f}s - {utterance.end/1000:.1f}s]:\n")
                    f.write(f"{utterance.text}\n")
            else:
                f.write(transcript.text)
        print(f"Full transcription saved to: {output_text_path}")
    
    # Create DataFrame with speaker information
    df = pd.DataFrame({
        'Sentence_Number': range(1, len(sentences_with_speakers) + 1),
        'Speaker': [s['speaker'] for s in sentences_with_speakers],
        'Sentence': [s['text'] for s in sentences_with_speakers],
        'Start_Time': [f"{s['start_time']:.1f}s" if s['start_time'] else '' for s in sentences_with_speakers],
        'End_Time': [f"{s['end_time']:.1f}s" if s['end_time'] else '' for s in sentences_with_speakers],
        'Notes': [''] * len(sentences_with_speakers)
    })
    
    # Save to Excel
    df.to_excel(output_excel_path, index=False, engine='openpyxl')
    
    print(f"\nExcel file created: {output_excel_path}")
    print(f"Total sentences: {len(sentences_with_speakers)}")
    
    # Show speaker distribution
    if transcript.utterances:
        speaker_counts = df['Speaker'].value_counts()
        print(f"\nSentences per speaker:")
        for speaker, count in speaker_counts.items():
            print(f"  {speaker}: {count} sentences")
    
    # Show first few sentences as preview
    print(f"\nFirst 3 sentences:")
    for i in range(min(3, len(sentences_with_speakers))):
        s = sentences_with_speakers[i]
        print(f"{i+1} [{s['speaker']}]: {s['text'][:80]}...")
    
    return transcript, df

# Configuration
API_KEY = "ae9d26f45c504101be8279aebb10a8f7"
AUDIO_FILE = r"C:\Users\koenm\Downloads\Wie is de Mol (The Mole) S16E05 with English subtitles.mp3"
OUTPUT_EXCEL = r"C:\Users\koenm\Documents\assebbly_test\trans_with_speaker.xlsx"
OUTPUT_TEXT = r"C:\Users\koenm\Documents\assebbly_test\trans_with_speaker_text"

# Run the pipeline
if __name__ == "__main__":
    transcript, df = transcribe_and_create_excel(
        audio_file_path=AUDIO_FILE,
        api_key=API_KEY,
        output_excel_path=OUTPUT_EXCEL,
        output_text_path=OUTPUT_TEXT
    )