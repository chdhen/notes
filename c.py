# cnn lstm model
from numpy import mean
from numpy import std
import seaborn as sns
from numpy import dstack
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import metrics
from sklearn.metrics import classification_report
from keras.layers import Permute, Reshape
from sklearn import metrics
from matplotlib import pyplot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from torch.utils.tensorboard import SummaryWriter

#-----DATA EXPLORATION
features=list()
with open('UCI HAR Dataset/features.txt') as feat:
    features=[l.split()[1] for l in feat.readlines()]
print("Number of features = {}".format(len(features)))
for f in features:
    print(f,end=" | ")

#---Exploring Train Data
train = pd.read_csv('UCI HAR Dataset/train/X_train.txt', delim_whitespace=True, header=None, encoding='UTF-8')
train.columns = features
train['subject'] = pd.read_csv('UCI HAR dataset/train/subject_train.txt', header=None, squeeze=True)
train['Activity'] = pd.read_csv('UCI HAR dataset/train/y_train.txt', names=['Activity'], squeeze=True)
train['ActivityName'] = train['Activity'].map({1: 'WALKING', 2:'WALKING_UPSTAIRS',3:'WALKING_DOWNSTAIRS',4:'SITTING', 5:'STANDING',6:'LAYING'})
print(train.sample())
print(train.shape)
#---test
test = pd.read_csv('UCI HAR Dataset/test/X_test.txt', delim_whitespace=True, header=None, encoding='UTF-8')
test.columns = features
test['subject'] = pd.read_csv('UCI HAR dataset/test/subject_test.txt', header=None, squeeze=True)
test['Activity'] = pd.read_csv('UCI HAR dataset/test/y_test.txt', names=['Activity'], squeeze=True)
test['ActivityName'] = train['Activity'].map({1: 'WALKING', 2:'WALKING_UPSTAIRS',3:'WALKING_DOWNSTAIRS',4:'SITTING', 5:'STANDING',6:'LAYING'})
print(test.sample())
print(test.shape)
#----查看空项
print('Duplicates in train = {}'.format(sum(train.duplicated())))
print('Duplicates in test = {}'.format(sum(test.duplicated())))
print('Invalid values in train = {}'.format(train.isnull().values.sum()))
print('Invalid values in test = {}'.format(test.isnull().values.sum()))


#----CNN-LSTM Model
def file_load(filepath):
    df = read_csv(filepath, header=None, delim_whitespace=True)
    return df.values
def train_test_append(filenames, append_before=''):
    datalist = list()
    for name in filenames:
        data = file_load(append_before + name)
        datalist.append(data)
    datalist = dstack(datalist)
    return datalist

def inertial_signals_load(group, append_before=''):
    filepath = append_before + group + '/Inertial Signals/'
    filenames = list()
    filenames += ['total_acc_x_' + group + '.txt', 'total_acc_y_' + group + '.txt', 'total_acc_z_' + group + '.txt']
    filenames += ['body_acc_x_' + group + '.txt', 'body_acc_y_' + group + '.txt', 'body_acc_z_' + group + '.txt']
    filenames += ['body_gyro_x_' + group + '.txt', 'body_gyro_y_' + group + '.txt', 'body_gyro_z_' + group + '.txt']
    X = train_test_append(filenames, filepath)
    y = file_load(append_before + group + '/y_' + group + '.txt')
    return X, y


def load_dataset(append_before=''):
    trainX, trainy = inertial_signals_load('train', append_before + 'UCI HAR Dataset/')
    testX, testy = inertial_signals_load('test', append_before + 'UCI HAR Dataset/')
    trainy = trainy - 1
    testy = testy - 1
    trainy = to_categorical(trainy)
    testy = to_categorical(testy)
    print(trainX.shape, trainy.shape, testX.shape, testy.shape)
    return trainX, trainy, testX, testy


trainX, trainy, testX, testy = load_dataset()
from PIL import Image
import numpy as np
import os




#testX = np.interp(testX, (testX.min(), testX.max()), (0, 255))

for i in range(testX.shape[0]):
    image_data = testX[i]  # 取出第 i 个样本的所有特征值
    image = Image.fromarray(image_data, mode='F')  # 将二维数组转换成 PIL.Image 对象
    image_dir = './A/images/test'  # 图片保存路径
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    if image.mode == "F":
        image = image.convert('RGB')

    image_path = os.path.join(image_dir, f'{i}.png')  # 构造图片保存路径
    image.save(image_path)  # 保存图片到指定路径
# verbose, epochs, batch_size = 0, 400, 150
# n_timesteps = trainX.shape[A]#n_timesteps = 128
# n_features = trainX.shape[2]# n_features = 9
# n_outputs = trainy.shape[A]# n_outputs = 6
# print("trainX:",trainX.shape)

# n_steps = 4
# n_length = 32
# trainX = trainX.reshape((trainX.shape[0], n_steps, n_length, n_features))
# testX = testX.reshape((testX.shape[0], n_steps, n_length, n_features))
# print(trainX.shape, testX.shape)
