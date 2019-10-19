from pydub import AudioSegment
import os
import sys
import ffmpeg
from pyAudioAnalysis.pyAudioAnalysis.utilities import load_file

file_path = "/Users/gillian/PycharmProjects/voice_mining/mp3_douyin_hot"
file_list = load_file(file_path)
for i, file in enumerate(file_list):
    sound = AudioSegment.from_mp3(file)
    names_index = file.find(",")
    name = file[0:names_index]
    sound.export ("%s.wav" % name, format = "wav")