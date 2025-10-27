import re

# def process_videos_in_directory(input_dir: str="data\\video", output_dir: str="data\\audio", output_format: str = "mp3") -> None:
#     """
#     Processes all .mov and .mp4 files in the input directory, converting them to the specified audio format
#     and saving them in the output directory.

#     :param input_dir: Directory containing input video files.
#     :type input_dir: str
#     :param output_dir: Directory to save the converted audio files.
#     :type output_dir: str
#     :param output_format: Desired audio format, either 'wav' or 'mp3'. Default is 'wav'.
#     :type output_format: str
#     :return: None
#     """
#     if not os.path.exists(input_dir):
#         raise ValueError(f"Input directory does not exist: {input_dir}")
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)

#     for filename in os.listdir(input_dir):
#         if filename.lower().endswith((".mov", ".mp4")):
#             input_path = os.path.join(input_dir, filename)
#             base_name = os.path.splitext(filename)[0]
#             output_path = os.path.join(output_dir, f"{base_name}.{output_format}")

#             try:
#                 convert_video_to_audio(input_path, output_path, output_format)
#                 print(f"Converted {input_path} to {output_path}")
#             except Exception as e:
#                 print(f"Failed to convert {input_path}: {e}")
#         else:
#             print(f"No video files found in {input_dir}.")

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

# def parse_sentences_from_chunks(json_data: dict, output_dir: str = "data\\transcription\\csv") -> list[dict]:
#     """
#     Parses the text from chunks into sentences and assigns start and end times.
#     """
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)

#     chunks = json_data.get("chunks", [])

#     if not chunks:
#         raise ValueError("JSON data must contain 'chunks'.")

#     result = []
#     current_sentence_words = []
#     start_time = None

#     sentence_end_pattern = re.compile(r'[.?!]$')
    
#     for chunk in chunks:
#         word = chunk["text"]
#         timestamp = chunk["timestamp"]

#         if start_time is None:
#             start_time = timestamp[0]

#         current_sentence_words.append(word)

#         # Check for sentence-ending punctuation using a regex pattern
#         if sentence_end_pattern.search(word.strip()):
#             sentence_text = "".join(current_sentence_words).strip()
            
#             # Use regex to replace extra spaces with single spaces,
#             # but preserve spaces for normal words
#             sentence_text = re.sub(r'\s([.,?!])', r'\1', sentence_text)
#             sentence_text = re.sub(r'\s+', ' ', sentence_text).strip()
            
#             result.append({
#                 "Sentence": sentence_text,
#                 "Start_time": start_time,
#                 "End_time": timestamp[1]
#             })
            
#             # Reset for the next sentence
#             current_sentence_words = []
#             start_time = None

#     # Handle any remaining text as a final sentence
#     if current_sentence_words:
#         sentence_text = "".join(current_sentence_words).strip()
#         sentence_text = re.sub(r'\s([.,?!])', r'\1', sentence_text)
#         sentence_text = re.sub(r'\s+', ' ', sentence_text).strip()
        
#         result.append({
#             "Sentence": sentence_text,
#             "Start_time": start_time,
#             "End_time": chunks[-1]["timestamp"][1]
#         })

#     # Save to CSV
#     output_filepath = os.path.join(output_dir, "transcription.csv")
#     os.makedirs(output_dir, exist_ok=True)
    
#     with open(output_filepath, "w", newline="", encoding="utf-8") as csvfile:
#         writer = csv.DictWriter(csvfile, fieldnames=["Sentence", "Start_time", "End_time"])
#         writer.writeheader()
#         writer.writerows(result)
    
#     print(f"Results saved to {output_filepath}")
#     return result

dutch_short_words_with_punctuation = {
		"z'n": "zijn",
        "zo'n":"zo een",
		"d'r": "haar",
		"'k": "ik",
		"'t": "het",
		"'m": "hem",
		"'n": "een",
		"'r": "haar",
		"m'n": "mijn",
		"z'n'n": "zijn een",
		"m'n'n": "mijn een",
		"'n": "een",
		"'t": "het",
		"'s": "des",
		"'s": "des",
		"'k": "ik",
		"'ns": "eens",
		"'es": "eens",
		"'is": "eens",
		"'ie": "hij",
		"'em": "hem",
		"'twas": "het was"
	}

# def map_chunks_to_json(text, chunks):
#     """
#     Maps chunks and updates the sentence based on short-word mapping.

#     Returns a dictionary with:
#       - original text
#       - mapped text (with words replaced)
#       - chunks (with optional mapping info)
#     """
#     processed_chunks = []
#     mapped_text = text

#     # Create a mapping of original words to replacements for sentence replacement
#     replacements = {}

#     for chunk in chunks:
#         raw_text = chunk["text"]
#         word = raw_text
#         mapped_word = dutch_short_words_with_punctuation.get(word, None)

#         # If mapped, prepare for sentence substitution (preserving spacing and punctuation)
#         if mapped_word:
#             # Escape characters for safe regex substitution
#             escaped_word = re.escape(word)
#             # Use word boundaries where applicable to avoid partial replacements
#             pattern = rf'\b{escaped_word}\b'
#             mapped_text = re.sub(pattern, mapped_word, mapped_text)

#         # Add to processed chunks
#         processed_chunks.append({
#             "text": mapped_word if mapped_word else raw_text,
#             "timestamp": chunk["timestamp"],
#         })

#     return {
#         "text": mapped_text,
#         "chunks": processed_chunks
#     }

# def transcribe_and_process_audio_directory(audio_directory: str="data\\audio", output_dir: str="data\\transcription") -> None:
#     """
#     Transcribes the given audio files using Whisper and saves the result to a JSON file.

#     :param audio_path: Path to the input audio file.
#     :type audio_path: str
#     :param output_json_path: Path to save the transcription result in JSON format.
#     :type output_json_path: str
#     :return: None
#     """
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#     if not os.path.exists(os.path.join(output_dir, "json")):
#         os.makedirs(os.path.join(output_dir, "json"))
#     if not os.path.exists(audio_directory):
#         os.makedirs(audio_directory)

#     for file_name in os.listdir(audio_directory):
#         if file_name.endswith("mp3"):
#             base_filename = os.path.splitext(file_name)[0]
#             filepath = os.path.join(audio_directory, base_filename + ".mp3")
#             print(f"transcribing: {file_name}")
#             try:
#                 result = pipe(
#                     filepath, 
#                     return_timestamps="word",
#                     generate_kwargs={"language": "Dutch", "task": "transcribe"}
#                 )
#             except Exception as e:
#                 print(f"Transcription failed due: {e}")

#             print(f"{datetime.datetime.now()}: Transcription result:", result)
#             output_filepath = os.path.join(output_dir, "json", f"{base_filename}.json")
            
#             with open(output_filepath, "w", encoding="utf-8") as f:
#                 json.dump(result, f, ensure_ascii=False, indent=2)
#             print(f"Transcription saved to {output_filepath}")

#             mapped_result = map_chunks_to_json(result["text"], result["chunks"])
#             output_filepath = os.path.join(output_dir, "json", f"{base_filename}_mapped.json")

#             with open(output_filepath, "w", encoding="utf-8") as f:
#                 json.dump(mapped_result, f, ensure_ascii=False, indent=2)
#             print(f"Mapped transcription saved to {output_filepath}")

#             parse_sentences_from_chunks(mapped_result)
#         else:
#             print(f"No audio files found in {audio_directory}.")

if __name__ == "__main__":
    pass
