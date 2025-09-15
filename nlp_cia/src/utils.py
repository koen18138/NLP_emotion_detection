import os
from typing import Literal
import json
from moviepy.editor import VideoFileClip # moviepy==1.0.3
from transformers import BertTokenizer, BertForSequenceClassification
from sentiment_analysis_train import text_cleaning, encode_data
import pandas as pd

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

def load_model_and_tokenizer(model_dir: str):
    """
    Loads a fine-tuned BERT model and tokenizer from a specified directory.

    Args:
        model_dir (str): The directory where the model and tokenizer are saved.

    Returns:
        tuple: A tuple containing the loaded tokenizer and model.
    """
    print(f"Loading model and tokenizer from: {model_dir}")
    tokenizer = BertTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = BertForSequenceClassification.from_pretrained(model_dir)
    return tokenizer, model

def load_inference_data(tokenizer, inference_data: pd.DataFrame=None, file_path: str="data\\transcription\\csv\\transcription.csv") -> list:
	"""
	Loads inference data from a CSV file.

	Args:
		file_path (str): The path to the CSV file containing inference data.
	Returns:
		list: A list of sentences for inference.
	"""
	print(os.getcwd())
	if inference_data is not None:
		df = inference_data
	else:
		df = pd.read_csv(file_path)
	sentences = df['sentence'].apply(text_cleaning).tolist()
	encoded_sentences = encode_data(sentences, tokenizer)
	return encoded_sentences, df

def url_to_mp3(url: str, output_dir: str = r"data\\audio") -> None:
	# importing packages
	from pytubefix import YouTube
	import os

	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	
	# url input from youtube
	yt = YouTube(url)

	# extract only audio
	video = yt.streams.filter(only_audio=True).first()

	# download the file
	out_file = video.download(output_path=output_dir, filename="output.mp4")
	print(f"Downloaded to {out_file}")

	# save as mp3
	base, ext = os.path.splitext(out_file)
	new_file = base + '.mp3'
	try:
		os.rename(out_file, new_file)
	except FileExistsError:
		os.remove(new_file)
		os.rename(out_file, new_file)

if __name__ == "__main__":
	pass

