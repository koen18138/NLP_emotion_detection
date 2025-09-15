import assemblyai as aai
import re
import pandas as pd
import os
from preprocessing import split_into_sentences

def transcribe_and_create_excel(
        api_key: str, 
        output_filepath: str = "data\\transcription\\csv\\transcription.csv", 
        audio_filepath: str = "data\\audio\\output.mp3", 
        output_text_path=None
):
    """Complete pipeline: transcribe audio with speaker diarization and create Excel."""
    
    # Set up AssemblyAI
    aai.settings.api_key = api_key
    
    print("Starting transcription with speaker detection...")
    
    # Check if file exists
    if not os.path.exists(audio_filepath):
        print(f"Error: Audio file not found at {audio_filepath}")
        return
    
    # Configure transcription WITH SPEAKER DIARIZATION
    config = aai.TranscriptionConfig(
        speech_model=aai.SpeechModel.universal,
        language_code="nl",
        speaker_labels=True  # This enables speaker diarization
    )
    
    # Transcribe
    transcript = aai.Transcriber(config=config).transcribe(audio_filepath)
    
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
            f.write(f"Audio file: {output_text_path}\n")
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
    
    if output_filepath.endswith('.xlsx'):
        # Save to Excel
        df.to_excel(output_filepath, index=False, engine='openpyxl')
    else:
        # Save to CSV
        df.to_csv(output_filepath, index=False)
        
    print(f"\nExcel file created: {output_filepath}")
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

if __name__ == "__main__":
    pass