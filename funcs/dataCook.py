'''
Programmer: Xiaoxiao Lei
Purpose: Compile various functions for data-organize/preprocessing & other conveniences
'''
'''
[pandas]--"read_csv"
https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html
'''
# coding=utf-8
import argparse, os
#import tensorflow as tf
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
# plot associated
import seaborn as sns
import matplotlib.pyplot as plt
import pylab
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import time
# import datetime
# datetime.datetime.now()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# get the proportion of rate for each class
def get_classRate(classArr):
    classRateArr=[];
    for var in set(classArr):
        rate=(classArr==var).sum()/len(classArr)
        classRateArr=np.append(classRateArr,rate)

    return classRateArr

# change the rates of certain class to average
# scal how much proportion(s)/times would be the major-VERSUS-minor
def resolve_imb_data(X,targets,scale):
    classRateArr=get_classRate(targets)
    minRate=np.amin(classRateArr)
    subjN,feaNum=X.shape
    # get the proportion for each class
    prop=[]
    for r in classRateArr:
        if r==minRate:
            proportion=1
        else:
            proportion=minRate*scale/r
        prop=np.append(prop,proportion)
        #print("prop: ",prop)
    # for each class process starting idex
    classes=set(targets)

    endIdx=[]
    newX=np.array([], dtype=np.int64).reshape(0,feaNum)
    newTargets=[]
    for k,label in enumerate(classes):
        cookX=X
        classIdx=np.where(targets==label)[0]
        np.random.shuffle(classIdx)
        targetEnd=round(len(classIdx)*prop[k])
        newClassIdx=classIdx[:targetEnd]
        #newX = np.vstack([newX,X.iloc[newClassIdx,:]])
        newX = np.vstack([newX,X[newClassIdx]])
        newTargets=np.append(newTargets,targets[newClassIdx])

    # random shuffle the ouput
    newRandIdx=[i for i in range(len(newTargets))]
    np.random.shuffle(newRandIdx)
    newX=newX[newRandIdx]
    newTargets=newTargets[newRandIdx]

    return newX,newTargets


# proportionally split the data based on the requirement
def train_test_splitProp(X,targets,trainProp):
    classes=set(targets)
    #print("classes: ",classes)
    trainSampleNum,feaNum=X.shape
    idx=[i for i in range(len(targets))]
    x=X
    y=targets
    subjN,feaNum=x.shape
    x_train=np.array([],dtype=np.float).reshape(0,feaNum)
    x_test=np.array([],dtype=np.float).reshape(0,feaNum)
    y_train=[]
    y_test=[]
    for cc in classes:
        idxCat=np.where(y==cc)[0]
        newEnd=round(len(idxCat)*trainProp)
        '''
        print("idxCat:\n",idxCat.shape)
        print("newEnd:\n",newEnd)
        print("y[idxCat]: \n",y[idxCat])
        '''
        # feature-spec
        if str(type(x))=="<class 'numpy.ndarray'>":
            x_train=np.vstack([x_train,x[idxCat[:newEnd],:]])
            x_test=np.vstack([x_test,x[idxCat[newEnd:],:]])
        else:
            x_train=np.vstack([x_train,x.iloc[idxCat[:newEnd],:]])
            x_test=np.vstack([x_test,x.iloc[idxCat[newEnd:],:]])
        
        # label-spec
        y_train=np.append(y_train,y[idxCat[:newEnd]])
        y_test=np.append(y_test,y[idxCat[newEnd:]])

    # random-shuffle [training-dataset]
    newRandIdx1=[i for i in range(len(y_train))]
    np.random.shuffle(newRandIdx1)
    x_train=x_train[newRandIdx1]
    y_train=y_train[newRandIdx1]

    # random-shuffle [testing-dataset]
    newRandIdx2=[i for i in range(len(y_test))]
    np.random.shuffle(newRandIdx2)
    x_test=x_test[newRandIdx2]
    y_test=y_test[newRandIdx2]

    return x_train,y_train,x_test,y_test


def train_test_splitProp3D(X,targets,trainProp):
    classes=set(targets)
    #print("classes: ",classes)
    idx=[i for i in range(len(targets))]
    x=X
    y=targets
    subjN,feaNum0,feaNum1=x.shape
    x_train=np.array([],dtype=np.float64).reshape(0,feaNum0,feaNum1)
    x_test=np.array([],dtype=np.float64).reshape(0,feaNum0,feaNum1)
    y_train=[]
    y_test=[]
    for cc in classes:
        idxCat=np.where(y==cc)[0]
        newEnd=round(len(idxCat)*trainProp)
        '''
        print("idxCat:\n",idxCat.shape)
        print("newEnd:\n",newEnd)
        print("y[idxCat]: \n",y[idxCat])
        '''
        # feature-spec
        x_train=np.vstack([x_train,x[idxCat[:newEnd]]])
        x_test=np.vstack([x_test,x[idxCat[newEnd:]]])
        # label-spec
        y_train=np.append(y_train,y[idxCat[:newEnd]])
        y_test=np.append(y_test,y[idxCat[newEnd:]])

    # random-shuffle [training-dataset]
    newRandIdx1=[i for i in range(len(y_train))]
    np.random.shuffle(newRandIdx1)
    x_train=x_train[newRandIdx1]
    y_train=y_train[newRandIdx1]

    # random-shuffle [testing-dataset]
    newRandIdx2=[i for i in range(len(y_test))]
    np.random.shuffle(newRandIdx2)
    x_test=x_test[newRandIdx2]
    y_test=y_test[newRandIdx2]

    return x_train,y_train,x_test,y_test


# 按照比例cut-off
def propSplitXY3D(trainX, trainy, trainCutProp):
    trainCutSize = int(len(trainy) * trainCutProp)
    trainX, trainy = trainX[0:trainCutSize, :, :], trainy[0:trainCutSize]
    return trainX, trainy


def propSplitXY(trainX, trainy, trainCutProp):
    trainCutSize = int(len(trainy) * trainCutProp)
    trainX, trainy = trainX[0:trainCutSize, :], trainy[0:trainCutSize]
    return trainX, trainy


# proportionally split the data based on the requirement
# [time-series-data] manipulation
def train_test_splitProp_period1(X,targets,trainProp):
    classes=set(targets)
    print("classes: ",classes)
    subjNum,feaNum=X.shape
    #splitIdx=round(subjNum*trainProp)
    x=X
    y=targets
    x_train=np.array([],dtype=np.float).reshape(0,feaNum)
    x_test=np.array([],dtype=np.float).reshape(0,feaNum)
    y_train=[]
    y_test=[]
    for cc in classes:
        idxCat=np.where(y==cc)[0]
        newEnd=round(len(idxCat)*trainProp)
        print("[cc]: ",cc)
        print("idxCat:\n",idxCat.shape)
        print("newEnd:\n",newEnd)
        print("y[idxCat]: \n",len(y[idxCat]))
        # feature-spec
        x_train=np.vstack([x_train,x[idxCat[:newEnd]]])
        x_test=np.vstack([x_test,x[idxCat[newEnd:]]])
        # label-spec
        y_train=np.append(y_train,y[idxCat[:newEnd]])
        y_test=np.append(y_test,y[idxCat[newEnd:]])

    return x_train,y_train,x_test,y_test


def train_test_splitProp_period(X,targets,trainProp):
    if X.ndim==2:
        subjNum,_=X.shape
    elif X.ndim==3:
        subjNum,_,_=X.shape
    splitIdx=round(subjNum*trainProp)
    # specify X and Y
    x_train=X[:splitIdx]
    x_test=X[splitIdx:]
    y_train=targets[:splitIdx]
    y_test=targets[splitIdx:]

    return x_train,y_train,x_test,y_test

# graph [heatmap] plot for [confusion-matrix]
def draw_confusionMatrix(y_true,y_pred,outName,inputFile):
    # get the current time
    now=time.strftime("%Y-%m-%d_%H-%M-%S")
    # figure-plot
    sns.set()
    f,ax=pylab.subplots()
    cm=confusion_matrix(y_true,y_pred)
    sns.heatmap(cm,annot=True,ax=ax,cmap="tab10",fmt='.5g', annot_kws={"fontsize":22}) #"mako"
    #sns.set(font_scale=10)
    ax.set_title("confusion matrix")
    ax.set_xlabel("predict")
    ax.set_ylabel("true")
    '''
    SPECIFY & SAVE the [image-name]
    '''
    outFigName=outName+'_'+now+'.png'
    pylab.savefig(outFigName)
    '''
    Save the output report
    '''
    import sys
    savedStdout = sys.stdout  #保存标准输出流
    outTxt='./'+outName+'_'+now+'.txt'
    with open(outTxt, 'wt') as file:
        sys.stdout = file  #标准输出重定向至文件
        if empty(inputFile)==False:
            print("\n",inputFile)
        print("\n",cm)
        print(classification_report(y_true, y_pred))
    sys.stdout = savedStdout  #恢复标准输出流


# check whether a string is empty or not
# True --> if empty; False --> if NOT empty
def empty(mystring):
    assert isinstance(mystring, str)
    mystring=mystring.replace(' ', '')
    if len(mystring) == 0:
        return True #string is empty
    else:
        return False #string is not empty

# define the data-extraction associated information
