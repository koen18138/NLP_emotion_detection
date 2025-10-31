def url_to_mp3(url: str, output_dir: str = r"data\\audio") -> str:
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
	
	return new_file

if __name__ == "__main__":
	pass

