from pyAudioAnalysis.utilities import load_file
from pyAudioAnalysis.audioFeatureExtraction import stFeatureExtraction,mtFeatureExtraction
from pyAudioAnalysis.audioBasicIO import  readAudioFile
#mtFeatureExtraction(signal, fs, mt_win, mt_step, st_win, st_step):


file_path = "/Volumes/[C] Windows 10/Projects/pyAudioAnalysis/1.music_wav"
file_list = load_file(file_path)
features_st = list()
features_mt = list()


for i, file in enumerate(file_list):
    [Fs, signal] = readAudioFile(file)
    st_features = stFeatureExtraction(signal, Fs, 0.05*Fs,0.25*Fs )
    mt_features = stFeatureExtraction(signal, Fs, 0.5 * Fs, 2.5 * Fs)
    features_st.append(st_features)
    features_mt.append(mt_features)


