from pyAudioAnalysis.pyAudioAnalysis import audioBasicIO
import os
import subprocess as sp
import itertools

class Dataset:

	# Dataset object is composed of:
	# data 
	# targets
	# train and test sets for cross validation
	# classes dictionary to map classes to numbers

	def __init__(self,root_path,type,decode):
		self.type = type
		if  type == "casia":
			self.classes = {0:'angry', 1:'fear', 2:'happy', 3:'neutral', 4:'sad', 5:'surprise'}
			self.get_casia_dataset(root_path)

	def get_casia_dataset(self, root_path):
		speakers = []
		classes = {v: k for k, v in self.classes.items() }
		print (classes)
		self.targets = []; self.data = []; self.train_sets = []; self.test_sets =[]; get_data = True;audio_files_number = 0;
		for speaker in os.listdir(root_path):
			if speaker[0] == ".":
				continue
			speakers.append(speaker)
			test = []
			print ("speakers",speakers)
			for emotion in os.listdir(os.path.join(root_path, speaker)):
				if emotion[0] =="." :
					continue
				print ("emotion",emotion)
				if os.path.isdir(os.path.join(root_path, speaker, emotion)):
					for audio_files in os.listdir(os.path.join(root_path, speaker, emotion)):
						if audio_files[0] == ".":
							continue
						if audio_files[-3: ] == 'wav':
							audio_files_path = os.path.join(root_path, speaker, emotion, audio_files)
							# print audio_files
							[Fs, x] = audioBasicIO.readAudioFile(audio_files_path)
							self.data.append((x, Fs))
							self.targets.append(classes[emotion])
							test.append(audio_files_number)
							audio_files_number += 1
			self.test_sets.append(test)
		for speaker in range(len(speakers)):
			train = []
			for i in range(audio_files_number):
				if i not in self.test_sets[speaker]:
					train.append(i)
					self.train_sets.append(train)
