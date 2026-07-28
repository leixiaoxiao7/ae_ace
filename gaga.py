import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from funcs.tools import *
from funcs.nns import *
from funcs.dataCook import *

# 1. 模拟数据 (你可以替换为真实数据)
#X = np.random.rand(360, 15, 5).astype(np.float32)      # 输入：360 个样本，形状是 15x5
#Y = np.random.randint(0, 2, (360, 1)).astype(np.float32) # 标签：360 个样本，值为 0 或 1

outSavePkl='./data/gmwm3D'
# load 3D gmwm data
data = loadPkl_file(outSavePkl)

gmwm3D = data['gmwm3D']
gmwm3D_extend=data['gmwm3D_extend']
gmwmDF = data['df'] 
gmwmDim = data['gmwmDimensions']
uniqueHeaders = data['uniqueHeaders']
otherFeaturesH = data['otherFeatureHeaders']
siteList = data['siteList']
X=gmwm3D
y=gmwmDF['ace']
trainX, y_train, testX, y_test=train_test_splitProp3D(X, y, 0.8)

data_dim=4
agentStructureName='get_stcdcnn_model'

# # # 2. 构建 CNN 模型
# model = Sequential([
#     Conv1D(filters=32, kernel_size=5, activation='relu',padding='Same', input_shape=trainX.shape[1:]),
#     MaxPooling1D(pool_size=1, strides=1),
#     Dropout(0.1),
#
#     Conv1D(filters=64, kernel_size=3, activation='relu',padding='Same'),
#     MaxPooling1D(pool_size=1, strides=1),
#     Dropout(0.1),
#
#     Flatten(),
#     Dense(200, activation='relu'),
#     Dense(64, activation='relu'),
#     Dropout(0.1),
#     Dense(1, activation='softmax')  # 二分类输出
# ])
#
# # 3. 编译模型
# model.compile(optimizer=Adam(learning_rate=0.001),
#               loss='binary_crossentropy',
#               metrics=['accuracy'])

subjNum,feaNum,_=trainX.shape
if data_dim==4:
    trainX=trainX.reshape(trainX.shape[0],trainX.shape[1],trainX.shape[2],1)
    testX=testX.reshape(testX.shape[0],testX.shape[1],testX.shape[2],1)

in_shape=trainX.shape[1:]
model=eval(f'{agentStructureName}(in_shape,feaNum)')

# 4. 模型训练
model.fit(trainX, y_train, epochs=100, batch_size=16, validation_split=0.1)

# 5. 模型评估
loss, mae,acc = model.evaluate(trainX, y_train)
print(f'Model Accuracy: {acc:.4f}')

# 6. 模型预测（示例）
#testX = np.random.rand(5, 15, 5).astype(np.float32)
Y_pred_prob = model.predict(testX)
Y_pred_class = (Y_pred_prob > 0.5).astype(int)

accuracy = accuracy_score(y_test, Y_pred_class)
print(f"准确率: {accuracy:.2f}")  # 输出: 0.80

cm = confusion_matrix(y_test, Y_pred_class)
print("混淆矩阵:\n", cm)

report = classification_report(y_test, Y_pred_class)
print("分类报告:\n", report)
# print("预测概率：", Y_pred_prob.ravel())

'''
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

# 加载数据并训练模型
digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.3)
mlp = MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu')
mlp.fit(X_train, y_train)

# 评估模型
y_pred = mlp.predict(X_test)
print("准确率:", accuracy_score(y_test, y_pred))
print("混淆矩阵:\n", confusion_matrix(y_test, y_pred))
print("分类报告:\n", classification_report(y_test, y_pred))

'''
