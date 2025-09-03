# Install the assemblyai package by executing the command "pip install assemblyai"

import assemblyai as aai

aai.settings.api_key = "ae9d26f45c504101be8279aebb10a8f7"

# audio_file = "./local_file.mp3"
audio_file = r"/Applications/Mp3/Wie is de Mol (The Mole) S16E05 with English subtitles.mp3"

config = aai.TranscriptionConfig(
    speech_model=aai.SpeechModel.universal,
    language_code="nl",
)

transcript = aai.Transcriber(config=config).transcribe(audio_file)

if transcript.status == "error":
  raise RuntimeError(f"Transcription failed: {transcript.error}")

output_file = "/Users/florislokhorst/Desktop/transcription.txt"

with open(output_file, "w", encoding="utf-8") as f:
    f.write(f"Audio file: {audio_file}\n")
    f.write(f"Confidence: {transcript.confidence}\n")
    f.write(f"Duration: {transcript.audio_duration}ms\n\n")
    f.write("Transcription:\n")
    f.write(transcript.text)
    
print(f"Reported duration: {transcript.audio_duration}ms")
print(f"Text length: {len(transcript.text)} characters")
print(f"Word count: {len(transcript.text.split())} words")

if hasattr(transcript, 'words'):
    print(f"Number of word objects: {len(transcript.words)}")
print("Transcription saved to transcription.txt")