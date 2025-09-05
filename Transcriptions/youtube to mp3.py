# importing packages
from pytubefix import YouTube
import os

# url input from youtube
yt = YouTube("https://www.youtube.com/watch?v=gO8N3L_aERg")

# extract only audio
video = yt.streams.filter(only_audio=True).first()

# set destination to save file
destination = (r"C:\Users\koenm\Documents\test\fire")

# download the file
out_file = video.download(output_path=destination)

# save the file
base, ext = os.path.splitext(out_file)
new_file = base + '.mp3'
os.rename(out_file, new_file)

