
import os
from typing import Literal
from moviepy.editor import VideoFileClip # moviepy==1.0.3

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
		audio = video.audio
		if audio is None:
			raise ValueError("No audio stream found in the video file.")
		audio.write_audiofile(output_path, codec="libmp3lame" if output_format == "mp3" else None)
		
	
if __name__ == "__main__":
	# Example usage
	try:
		convert_video_to_audio("example.mov", "output.wav", "wav")
		print("Conversion successful!")
	except Exception as e:
		print(f"An error occurred: {e}")