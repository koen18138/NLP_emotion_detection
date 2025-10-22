import assemblyai as aai
import pandas as pd
import os
from preprocessing import split_into_sentences

from datetime import timedelta

def format_timestamp(seconds):
    """Convert seconds to HH:MM:SS format"""
    if seconds is None or seconds == '' or pd.isna(seconds):
        return None
    td = timedelta(seconds=float(seconds))
    # Convert to string and remove days if present
    time_str = str(td)
    if 'days' in time_str:
        time_str = time_str.split(', ')[1]
    return time_str


def transcribe_and_create_excel(
        api_key: str, 
        output_filepath: str = "data\\transcription\\csv\\transcription.csv", 
        audio_filepath: str = "data\\audio\\output.mp3", 
        output_text_path=None,
        get_speaker: bool = False,
        debug: bool = False
):
    """Complete pipeline: transcribe audio with speaker diarization and create Excel."""
    
    # Set up AssemblyAI
    aai.settings.api_key = api_key
    
    if get_speaker:
        print("Starting transcription with speaker detection...")
    if not get_speaker:
        print("Starting transcription without speaker detection...")


    # Check if output dir exist
    if not os.path.exists(os.path.dirname(output_filepath)):
        os.makedirs(os.path.dirname(output_filepath))

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
    data = []
    
    if transcript.utterances:
        print(f"Detected {len(set(u.speaker for u in transcript.utterances))} different speakers")
        for utterance in transcript.utterances:
            # Split each utterance into sentences but keep speaker info
            utterance_sentences = split_into_sentences(utterance.text)
            for sentence in utterance_sentences:
                data.append({
                    'Sentence': sentence,
                    'Speaker': utterance.speaker,
                    'Start Time': utterance.start / 1000,  # Convert ms to s
                    'End Time': utterance.end / 1000       # Convert ms to s
                })
                
    else:
        print('Failed to detect utterances')

    
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
    if not get_speaker:
        # Create DataFrame without speaker information
        df = pd.DataFrame({
            'Start Time': [format_timestamp(s['Start Time']) for s in data],
            'End Time': [format_timestamp(s['End Time']) for s in data],
            'Sentence': [s['Sentence'] for s in data],
        })
    if get_speaker:
        # Create DataFrame with speaker information
        df = pd.DataFrame({
            'Start Time': [format_timestamp(s['Start Time']) for s in data],
            'End Time': [format_timestamp(s['End Time']) for s in data],
            'Sentence': [s['Sentence'] for s in data],
            'Speaker': [s['Speaker'] for s in data],
        })
    
    if output_filepath.endswith('.xlsx'):
        # Save to Excel
        df.to_excel(output_filepath, index=False, engine='openpyxl')
    else:
        # Save to CSV
        df.to_csv(
            output_filepath, 
            index=False, 
        )   
        
    print(f"\nExcel file created: {output_filepath}")
    print(f"Total sentences: {len(data)}")
    if debug:
        # Show speaker distribution
        if transcript.utterances:
            speaker_counts = df['Speaker'].value_counts()
            print(f"\nSentences per speaker:")
            for speaker, count in speaker_counts.items():
                print(f"  {speaker}: {count} sentences")
        
        # Show first few sentences as preview
        print(f"\nFirst 3 sentences:")
        for i in range(min(3, len(data))):
            s = data[i]
            print(f"{i+1} [{s['speaker']}]: {s['text'][:80]}...")
    
    return transcript, df

# Configuration
API_KEY = "ae9d26f45c504101be8279aebb10a8f7"

if __name__ == "__main__":
    pass