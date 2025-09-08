import os
from typing import Literal
import json
import csv
from moviepy.editor import VideoFileClip # moviepy==1.0.3
import re

def convert_video_to_audio(
	input_path: str,
	output_path: str,
	output_format: Literal["wav", "mp3"] = "wav"
) -> None:
	"""
	Converts a .mov or .mp4 video file to a .wav or .mp3 audio file.

	:param input_path: Path to the input video file (.mov or .mp4).
	:type input_path: str
	:param output_path: Path to the output audio file (.wav or .mp3).
	:type output_path: str
	:param output_format: Desired audio format, either 'wav' or 'mp3'. Default is 'wav'.
	:type output_format: Literal["wav", "mp3"]
	:raises ValueError: If the input or output format is not supported, or if no audio stream is found.
	:raises FileNotFoundError: If the input file does not exist.
	:return: None
	"""
	if not os.path.isfile(input_path):
		raise FileNotFoundError(f"Input file not found: {input_path}")
	if not input_path.lower().endswith((".mov", ".mp4")):
		raise ValueError("Input file must be a .mov or .mp4 video.")
	if output_format not in ("wav", "mp3"):
		raise ValueError("Output format must be 'wav' or 'mp3'.")
	if not output_path.lower().endswith(f".{output_format}"):
		raise ValueError(f"Output file must end with .{output_format}")

	with VideoFileClip(input_path) as video:
		print(f"Processing video file: {input_path}")
		audio = video.audio
		if audio is None:
			raise ValueError("No audio stream found in the video file.")
		audio.write_audiofile(output_path, codec="libmp3lame" if output_format == "mp3" else None)
		
def split_mp4(input_file: str, num_parts: int, output_dir: str="data") -> list[str]:
	"""Splits an MP4 video file into a specified number of equal-duration parts.

	This function uses the moviepy library to load a video, calculate the duration
	for each segment, and export the video into multiple smaller files. The new
	files are saved in the same directory as the input file.

	:param input_file: The path to the input MP4 video file.
	:type input_file: str
	:param num_parts: The number of equal parts to split the video into.
	:type num_parts: int
	:raises ValueError: If the input file does not exist or if num_parts is not
				a positive integer.
	:return: A list of file paths to the newly created video parts.
	:rtype: list[str]
	"""
	if not os.path.exists(input_file):
		raise ValueError(f"Input file not found: {input_file}")

	if not isinstance(num_parts, int) or num_parts <= 0:
		raise ValueError("Number of parts must be a positive integer.")

	print(f"Loading video: {input_file}...")
	video_clip = VideoFileClip(input_file)
	total_duration = video_clip.duration
	part_duration = total_duration / num_parts
	output_files = []

	# Get the base name and extension to create new filenames
	file_name, file_ext = os.path.splitext(input_file)
	print(f"{file_name = }, {file_ext = }")

	for i in range(num_parts):
		start_time = i * part_duration
		# Ensure the last part goes to the very end of the video
		end_time = (i + 1) * part_duration if i < num_parts - 1 else total_duration

		print(f"Creating part {i + 1}/{num_parts} from {start_time:.2f}s to {end_time:.2f}s...")
		subclip = video_clip.subclip(start_time, end_time)

		output_file = f"{file_name}_part_{i + 1}{file_ext}"
		output_filepath = os.path.join(output_dir, output_file)

		print(f"Exporting to {output_filepath}...")

		# Write the new video file
		# The 'codec' parameter is set to 'libx264' for better compatibility
		subclip.write_videofile(output_filepath, codec="libx264", audio_codec="aac")

		output_files.append(output_file)
		print(f"Part {i + 1} saved as {output_file}.")

	video_clip.close()
	print("Video splitting complete.")
	return output_files

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



def parse_sentences_with_timestamps(json_data: dict, output_csv: str="sentences.csv") -> list[dict]:
    """
    Parses the text in the JSON data into sentences and assigns start and end times
    to each sentence based on the chunks' timestamps.

    :param json_data: The JSON data containing "text" and "chunks".
    :type json_data: dict
    :return: A list of dictionaries, each containing a sentence, start time, and end time.
    :rtype: list[dict]
    """
    text = json_data.get("text", "")
    chunks = json_data.get("chunks", [])

    if not text or not chunks:
        raise ValueError("JSON data must contain 'text' and 'chunks'.")

    sentences = split_into_sentences(text)
    result = []
    chunk_index = 0  # Track the current chunk being processed

    for sentence in sentences:
        sentence = sentence.strip()
        start_time = None
        end_time = None
        accumulated_text = ""

        while chunk_index < len(chunks):
            chunk = chunks[chunk_index]
            chunk_text = chunk["text"].strip()

            # Add the current chunk text to the accumulated text
            if accumulated_text and not chunk_text.startswith(("'", "-")):
                accumulated_text += " "  # Add a space between chunks
            accumulated_text += chunk_text

            # Set the start time if it's the first chunk for this sentence
            if start_time is None:
                start_time = chunk["timestamp"][0]

            # Update the end time to the current chunk's end time
            end_time = chunk["timestamp"][1]

            # Check if the accumulated text matches the sentence
            if accumulated_text == sentence:
                chunk_index += 1  # Move to the next chunk for the next sentence
                break

            # If the accumulated text exceeds the sentence, something is wrong
            if len(accumulated_text) > len(sentence):
                raise ValueError(f"Mismatch between sentence and chunks: '{sentence}' vs '{accumulated_text}'")

            chunk_index += 1  # Move to the next chunk

        result.append({
            "sentence": sentence,
            "start_time": start_time,
            "end_time": end_time
        })

    # Save the result to a CSV file
    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["sentence", "start_time", "end_time"])
        writer.writeheader()
        writer.writerows(result)

    print(f"Results saved to {output_csv}")

dutch_short_words_with_punctuation = {
		"z'n": "zijn",
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

def map_chunks_to_json(text, chunks):
    """
    Maps chunks and updates the sentence based on short-word mapping.

    Returns a dictionary with:
      - original text
      - mapped text (with words replaced)
      - chunks (with optional mapping info)
    """
    processed_chunks = []
    mapped_text = text

    # Create a mapping of original words to replacements for sentence replacement
    replacements = {}

    for chunk in chunks:
        raw_text = chunk["text"]
        word = raw_text
        mapped_word = dutch_short_words_with_punctuation.get(word, None)

        # If mapped, prepare for sentence substitution (preserving spacing and punctuation)
        if mapped_word:
            # Escape characters for safe regex substitution
            escaped_word = re.escape(word)
            # Use word boundaries where applicable to avoid partial replacements
            pattern = rf'\b{escaped_word}\b'
            mapped_text = re.sub(pattern, mapped_word, mapped_text)

        # Add to processed chunks
        processed_chunks.append({
            "text": mapped_word if mapped_word else raw_text,
            "timestamp": chunk["timestamp"],
        })

    return {
        "text": text,
        "chunks": processed_chunks
    }

def merge_transcripts(directory: str) -> None:
	"""
	Merges multiple transcript JSON objects into one.
	Assumes each JSON file in the directory has "text" and "chunks" keys.
	:param directory: Directory containing JSON transcript files.
	:type directory: str
	:return: None
	:rtype: None
	"""
	merged_text = ""
	merged_chunks = []
	time_offset = 0.0

	for filename in os.listdir(directory):
		if filename.endswith(".json"):
			print(f"Processing file: {filename}")
			filepath = os.path.join(directory, filename)
			with open(filepath, "r", encoding="utf-8") as f:
				transcript = json.load(f)
			
			text = transcript.get("text", "")
			chunks = transcript.get("chunks", [])

			# Append text with a space between segments
			if merged_text and not merged_text.endswith(" "):
				merged_text += " "
			merged_text += text.strip()

			for chunk in chunks:
				chunk_text = chunk["text"]
				start, end = chunk["timestamp"]

				# Offset timestamps
				new_start = round(start + time_offset, 2)
				new_end = round(end + time_offset, 2)

				merged_chunks.append({
				"text": chunk_text,
				"timestamp": [new_start, new_end]
				})

			# Update time offset to the end of the last chunk
			if chunks:
				time_offset = merged_chunks[-1]["timestamp"][1]

	# Save merged transcript as JSON
	output_path = os.path.join(directory, "merged_transcript.json")
	with open(output_path, "w", encoding="utf-8") as out_file:
		json.dump({
			"text": merged_text,
			"chunks": merged_chunks
		}, out_file, ensure_ascii=False, indent=2)
	print(f"Merged transcript saved to {output_path}")

if __name__ == "__main__":

	merge_transcripts("data/output")
	json_path = "data/output/merged_transcript.json"
	with open(json_path, "r", encoding="utf-8") as f:
		json_data = json.load(f)


	json_data_mapped = map_chunks_to_json(json_data["text"], json_data["chunks"])
	print(json_data_mapped)
	parse_sentences_with_timestamps(json_data)
	import pandas as pd
	df = pd.read_csv("sentences.csv")
	print(df["sentence"][2])


