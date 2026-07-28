import json
import pickle
import argparse
import os
import sys
#import tensorflow as tf
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
# plot associated
import seaborn as sns
import matplotlib.pyplot as plt
import pylab
import math
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import time
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle

# import datetime
# datetime.datetime.now()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 查看numpy array里面是否有浮点数
# output specific


def npArr_contains_float(npArr):
    #lst=npArr.tolist()
    # for item in npArr:
    #     print(f"------------------type: {item.dtype}")
    return ('float' in str(npArr.dtype))

# 查看list里面是否有float的值


def list_contains_float(lst):
    return any(isinstance(item, float) for item in lst)

# 拿到list一个value的index array


def find_indices(lst, value):
    return [i for i, x in enumerate(lst) if x == value]


# 从list拿到一个sublist的所有index array
def find_items_indices(lst, itemList):
    return [i for i, x in enumerate(lst) if x in itemList]


# grab multiple-items absed on indicies
def indexItems(list, indices):
    return [list[j] for j in indices]


# check if a list has duplicates item(s)
def listHasDuplicates(lst):
    return len(lst) != len(set(lst))

# check if all items from a list/array are the SAME


def allSame(items):
  it = iter(items)
  for first in it:
    break
  else:
    return True  # empty case, note all([]) == True
  return all(x == first for x in it)


# read from a json file
def readFromJson(dirFile):
    with open(dirFile, 'r') as file:
        content = file.read()
    try:
        data = json.loads(content)
        return data
    except json.JSONDecodeError as err:
        raise Exception('树形结构JSON读取错误，请检查-----------: ', err)


# save a dictionary to json-txt
def saveDicToJsonTxt(dict, dictFile):
    try:
        with open(dictFile, 'w') as fp:
            json.dump(dict, fp)
            return True
    except Exception as e:
        return False


# save json to directory
def saveJson(jsonF, dictFile):
    try:
        with open(dictFile, 'w') as f:
            json.dump(jsonF, f)
            return True
    except Exception as err:
        print("save-JSON-FALSE: ", err)
        return False

# read dic from json-txt


def readDicFromJsonTxt(dictFile):

    outDict = None
    with open(dictFile, 'r') as fp:
        outDict = json.load(dictFile)

    return outDict


# read from TXT to jsonArray:
def readTxtAsJsonArr(file):

    # read processed json-file-data
    with open(file, 'r') as file:
        content = file.read()
    data = json.loads(content)

    return data


# 存储txt文件
# save the txt file into directory
def saveTxt(dir, outName, info, tsStruct=None):
    # if NO-directory, create one
    if os.path.exists(f'./{dir}') == False:
        os.makedirs(f'./{dir}')
    # mange text outputs
    savedStdout = sys.stdout
    outTxt = './'+dir+'/'+outName+'.txt'
    with open(outTxt, 'wt') as file:
        sys.stdout = file
        if tsStruct is not None:
            print("\n[built-TS-modelStruct]:\n", tsStruct.summary(), "\n\n")
        print("[info] --------------------------------------------: \n\n\n", info)

    sys.stdout = savedStdout


# check空、null或者不存在
def checkVoid(item):

    if item is None:
        return True

    if len(item) == 0:
        return True

    return False


#  存储PKL经典模型
def savePklMode(file, clf):
    try:
        with open(file, 'wb') as f:
            pickle.dump(clf, f)
            return True
    except:
        return False

# 读取经典PKL形式存储的模型信息


def readPklMode(file):
    try:
        with open(file, 'rb') as f:
            loadedCf = pickle.load(f)
            return loadedCf
    except Exception as err:
        print('**********************【读取模型数据失败】：', err)
        return None

# 存储tensorflow深度学习模型


def saveTensorModel(file, model):
    try:
        model.save(file)
        return True
    except:
        return False

# 读取tensorflow深度学习模型


def readTensorModel(file, model=None):
    try:
        model = load_model(file)
    except Exception as err:
        print('*******************【读取深度学习模型错误】：', err)
    return model


# 针对【字典查询】，简化列名-复原到-->原始复杂列名
def simplifiedColToOrgName(simplifiedName, tarColumnsInfo):

    for key, value in tarColumnsInfo.items():
        if (value == simplifiedName):
            return key

    return None


# 检查信息是否为None或空
def checkVoid(item):

    if item == None:
        return True
    if len(str(item).replace(' ', '')) == 0:
        return True

    return False

# get the proportion of rate for each class


def get_classRate(classArr):
    classRateArr = []
    for var in set(classArr):
        rate = (classArr == var).sum()/len(classArr)
        classRateArr = np.append(classRateArr, rate)

    return classRateArr

# proportionally split the data based on the requirement
# [time-series-data] manipulation
def listItemVoid(item):
    if isinstance(item, str):
        return empty(item)
    elif item is None:
        return True
    else:
        return math.isnan(item)
''


# save pkl file
def savePkl_file(data,outName):
   with open(outName+'.pkl', 'wb') as f:
      pickle.dump(data, f)
      return "success"
   
   return None



# load pkl file
def loadPkl_file(fileName):
   with open(fileName+'.pkl', 'rb') as f:
      data = pickle.load(f)

   return data