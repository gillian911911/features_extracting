# Voice  features_extracting

1. There are two classes voice analysia code, 1)voice features extracting and 2)voice emotion prediction.
2. The voice features extracting is in pyAudioAnalysis packages.
3. The emotion analysis is in CASIA packages.


## pyAudioAnalysis is a Python library covering a wide range of audio analysis tasks. Through pyAudioAnalysis you can:

Extract audio features and representations (e.g. mfccs, spectrogram, chromagram)
Classify unknown sounds
Train, parameter tune and evaluate classifiers of audio segments
Detect audio events and exclude silence periods from long recordings
Perform supervised segmentation (joint segmentation - classification)
Perform unsupervised segmentation (e.g. speaker diarization)
Extract audio thumbnails
Train and use audio regression models (example application: emotion recognition)
Apply dimensionality reduction to visualize audio data and content similarities


#

# 语音特征分析及情感预测
## 数据集说明
本次建模采用的是中国科学院自动化所发布的中文语音情感数据库casia（http://www.chineseldc.org/resource_info.php?rid=76），共包括四个专业发音人，6种情绪，生气（angry）、高兴（happy）、害怕（fear）、悲伤（sad）、惊讶（surprise）和中性（neutral），共1200句发音。采用16kHz采样，16bit量化，并以WAV格式保存。

语音数据储存于CASIA database，使用Emotion_Recognition_from_Speech_master/src中的emorecognition.py即可完成语音特征分析及情感预测。

# 1.在emorecognition.py中导入相关模块，设定基本参数

# 2. emorecognition.py 调用dataset获取语料
emorecognition.py 调用dataset获取语料，并将音频数据处理后保存为casia_db.p文件。

在dataset中将对数据进行分组（train and test sets），并调研pyaudioanalysis文件中audioBasicIO将音频文件转化为音频流数组，即fs :采样频率；x:音频样本。
数据标签集合={0:'angry', 1:'fear', 2:'happy', 3:'neutral', 4:'sad', 5:'surprise'}。

# 3. emorecognition.py 进行语音特征提取
在emorecognition.py 使用extract_features调用pyAudioAnalysis包进行语音特征提取。将得到的34个特征数组做均值mean和方差std,得到每个样本68个特征，总共1200个样本，将获取的特征保存为casia_features.p文件。


（1）	Pyaudioanalysis包

Pyaudioanalysis包中核心语音分析代码见_pycache_文件夹。单独对语料库进行分析，并导出excel见test文件夹中的voice_test1.py.


（2）voice_test1.py单独分析语音特征并储存为excel文件。

使用pyAudioAnalysis包里面的audioFeatureExtraction.stFeatureExtraction(signal, Fs, win, step)函数，输入参数signal,为前面所提取的音频流，Fs为采样率，对音频进行特征提取，提取的特征F为34*176（176与音频长度有关）矩阵，其中每一行代表一个不同的特征，共有34个不同特征，每个特征长度为176，以下为34个特征说明：

•	1-Zero Crossing Rate：短时平均过零率，即每帧信号内，信号过零点的次数，体现的是频率特性
•	2-Energy：短时能量，即每帧信号的平方和，体现的是信号能量的强弱
•	3-Entropy of Energy：能量熵，跟频谱的谱熵（Spectral Entropy）有点类似，不过它描述的是信号的时域分布情况，体现的是连续性
•	4-Spectral Centroid：频谱中心又称为频谱一阶距，频谱中心的值越小，表明越多的频谱能量集中在低频范围内，如：voice与music相比，通常spectral centroid较低
•	5-Spectral Spread：频谱延展度，又称为频谱二阶中心矩，它描述了信号在频谱中心周围的分布状况
•	6-Spectral Entropy：谱熵，根据熵的特性可以知道，分布越均匀，熵越大，能量熵反应了每一帧信号的均匀程度，如说话人频谱由于共振峰存在显得不均匀，而白噪声的频谱就更加均匀，借此进行VAD便是应用之一
•	7-Spectral Flux：频谱通量，描述的是相邻帧频谱的变化情况
•	8-Spectral Rolloff：频谱滚降点
•	9~21-MFCCs：9-21为梅尔倒谱系数，非常重要的音频特征。
•	22~33-Chroma Vector：这个有12个参数，对应就是12级音阶
•	34-Chroma Deviation：这个就是Chroma Vector的标准方差。

(3)34个特征的提取公式见audioFeatureExtraction.py文件。


# 4. 在emorecognition.py 中使用SVM对结构化语音数据的情感类型进行预测
使用sklearn包多分类multiclass的OneVsRestClassifier，SVM算法，进行多分类，其中kernel为采用的核函数，取rbf，C为目标函数的惩罚系数，用来平衡分类间隔margin和错分样本的，gamma为核函数的系数，probablity使用可能性估计。
对输入的特征进行标准化StandardScaler()处理和PCA主成分分析降维处理，保留50维最终模型入围，对数据集进行十折交叉验证。


预测结果分类为1，即为fear害怕
## 评价标准
十折交叉验证后
	平均准确率accuracy为0.75，标准差为0.02
	平均召回率recall为0.75，标准差为0.06
	平均精确率Precision为0.75，标准差为0.05
Accuracy（准确率——分类正确的概率）：（正样本分对的量+负样本分对的量）/总样本量
recall（召回率——正样本分类正确的概率）：正样本分对的量/正样本的样本量
Precision（精确率——预测正样本中，分类正确的概率）：正样本分对的量/预测的正样本数量

其中每个分类中的precison和recall为：

Label	Precision(mean/std)	Recall(mean/std)
Angry（200个）	0.73/0.05	0.72/0.06
fear（200个）	0.73/0.04	0.67/0.07
happy（200个）	0.72/0.06	0.73/0.06
neutral（200个）	0.84/0.05	0.87/0.06
sad（200个）	0.68/0.05	0.74/0.05
surprise（200个）	0.79/0.07	0.75/0.07

其中：中性情绪预测较准，而伤心情绪预测效果最差。

## 二分类
仅使用生气（angry）和开心（happy）做正负情绪分类，特征提取方法类似，最终结果为：

	平均准确率auc为0.81，标准差为0.03
	平均召回率recall为0.81，标准差为0.02
	平均精确率Precision为0.82，标准差为0.01

Label	Precision(mean/std)	Recall(mean/std)
angry	0.83/0.05	0.80/0.08
happy	0.81/0.06	0.83/0.07



ffmpeg下载地址https://www.ffmpeg.org/download.html#build-mac
