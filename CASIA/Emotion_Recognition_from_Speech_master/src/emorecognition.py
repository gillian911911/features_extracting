# -*- coding:utf-8 -*-
from pyAudioAnalysis.pyAudioAnalysis import audioFeatureExtraction
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from sklearn import svm
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from optparse import OptionParser
import numpy as np
from scipy import spatial
from scipy.stats import entropy
import scipy.io as sio
import numpy as np
import pickle
from time import sleep
import sys
from CASIA.Emotion_Recognition_from_Speech_master.src.dataset import Dataset
from CASIA.Emotion_Recognition_from_Speech_master.src.preprocessing import Eigenspectrum, Preprocessor
#from adv_statistics import negentropy, differences_entropy

if __name__ == '__main__':
	#warnings.filterwarnings('ignore')
	parser = OptionParser()
	parser.add_option("-d", "--dataset", dest="db_type", default="casia")
	parser.add_option("-p", "--dataset_path", dest="path", default="/Users/gillian/PycharmProjects/voice_mining/voice_ming20191018/CASIA/CASIA database")
	parser.add_option("-l", "--load_data", action="store_true", dest="load_data",default="True")
	parser.add_option("-e", "--extract_features", action="store_true", dest="extract_features",default="True")
	parser.add_option("-s", "--speaker_indipendence", action="store_true", dest="speaker_indipendence",default="True")
	parser.add_option("-i", "--plot_eigenspectrum", action="store_true", dest="plot_eigenspectrum",default="True")
	(options, args) = parser.parse_args(sys.argv)
	load_data = options.load_data
	extract_features = options.extract_features
	db_type = options.db_type
	speaker_indipendence = options.speaker_indipendence
	path = options.path
	plot_eigenspectrum = options.plot_eigenspectrum

	if load_data:
		print ("Loading data from " + db_type + " dataset...")
		if db_type not in ('casia'):
			sys.exit("Dataset not registered. Please create a method to read it")

		db = Dataset(path,db_type,decode=False)

		print( "Saving " + db_type + " dataset info to file...")
		pickle.dump(db, open(db_type + '_db.p', 'wb')) # https://blog.csdn.net/u010402786/article/details/51161828
		# cPickle used to save and load data of fixed structure
	else:
		print( "Getting data from " + db_type + " dataset...")
		db = pickle.load(open(db_type + '_db.p', 'rb'))

	n_samples = len(db.targets)
	print( "Number of dataset samples: " + str(n_samples))

	if extract_features:
		win_size = 0.04
		step = 0.01
		Fglobal = []
		i = 0
		for (x,Fs) in db.data:
			F = audioFeatureExtraction.stFeatureExtraction(x, Fs, win_size*Fs, step*Fs)
			Fglobal.append( np.concatenate((	np.mean(F[0], axis=1), # axis =1 ：压缩列，对各行求均值，返回 m *1 矩阵
												np.std(F[0], axis=1)))) # 计算每一行的标准差
                            # 数据拼接
			sys.stdout.write("\033[F") # cursor up one line
			i = i+1; print ("Extracting features " + str(i) + '/' + str(n_samples) + " from data...")

		print ("Saving features to file...")
		pickle.dump(Fglobal, open(db_type + '_features.p', 'wb'))
	else:
		print ("Getting features from files...")
		Fglobal = pickle.load(open(db_type + '_features.p', 'rb')) # Fglobal: features

	Fglobal = np.array(Fglobal)
	y = np.array(db.targets)

	# evaluating SVM using cross validation
	print ("Evaluating model with cross validation...")

	if speaker_indipendence:
		k_folds = len(db.test_sets)
		splits = zip(db.train_sets,db.test_sets)
	else:
		k_folds = 10 # cross_validation
		sss = StratifiedShuffleSplit(n_splits=k_folds, test_size=0.2, random_state=1)
		splits = sss.split(Fglobal, y)

	# setting preprocessing
	pp = Preprocessor('standard',n_components=50)
	n_classes = len(db.classes)
	clf = OneVsRestClassifier(svm.SVC(kernel='rbf',C=10, gamma=0.01))

	prfs = []; scores = []; acc = np.zeros(n_classes)
	mi_threshold = 0.0
	for (train,test) in splits:
		# selecting features using mutual information
		Ftrain = Fglobal[train]; Ftest = Fglobal[test]
		f_subset = pp.mutual_info_select(Ftrain,y[train],mi_threshold)
		# choose traits

		Ftrain = Ftrain[:,f_subset]; Ftest = Ftest[:,f_subset]
		
		#standard transformation
		(Ftrain,Ftest) = pp.standardize(Ftrain,Ftest)
		
		# eigenspectrum over all data
		if plot_eigenspectrum:
			es = Eigenspectrum(Ftrain)
			es.show()
		
		(Ftrain,Ftest) = pp.project_on_pc(Ftrain,Ftest)

		clf.fit(Ftrain, y[train]) # train the PCA model by X
		ypred = clf.predict(Ftest)

		#print clf.score(Ftest, y[test])
		scores.append(clf.score(Ftest, y[test])) # Returns the mean accuracy on the given test data and labels
		print (precision_recall_fscore_support(y[test], ypred))
		prfs.append(precision_recall_fscore_support(y[test], ypred))


	# mean total accuracy
	print("\nAccuracy =  %0.2f (%0.2f)\n" % (np.mean(scores), np.std(scores)))

	#mean per class precision and recall 
	mean_prec = np.zeros((1,n_classes))
	mean_recall = np.zeros((1,n_classes))
	precs = []; recalls = []
	for mat in prfs:
		precs.append(mat[0])
		recalls.append(mat[1])
		mean_prec = mean_prec + mat[0]
		mean_recall = mean_recall + mat[1]
	mean_prec = mean_prec[0] / k_folds
	mean_recall = mean_recall[0] / k_folds

	#mean total recall and precision
	precs = np.array(precs)
	recalls = np.array(recalls)
	prec_mean = np.mean(precs,axis=0)
	prec_std = np.std(precs,axis=0)
	recall_mean = np.mean(recalls,axis=0)
	recall_std = np.std(recalls,axis=0)
	print ("Recall %0.2f (%0.2f)" % (np.mean(recall_mean), np.std(recall_mean)))
	print ("Precision: %0.2f (%0.2f)\n" % (np.mean(prec_mean), np.std(prec_mean)))

	for i in range(0,n_classes):
		print (db.classes[i] + " precision = %0.2f (%0.2f)" % (prec_mean[i],prec_std[i]))
		print (db.classes[i] + " recall = %0.2f (%0.2f)\n" % (recall_mean[i],recall_std[i]))
