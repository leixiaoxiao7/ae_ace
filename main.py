from scipy.io import loadmat
import scipy.io as sio
import json
import pandas as pd
import numpy as np
import pickle
import time
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

import sys
sys.path.append('./funcs')
sys.path.append('./data')
from funcs.tools import *
from funcs.brainCook import *
from funcs.nns import *
from funcs.dataCook import *
from funcs.graphPlot import *

'''
##### PURPOSE:
    - organize GMWM data to 3D formates

'''


def all_unique_typeItems(t):
    return len(t) == len(set(t))



if __name__ == "__main__":

    timeStamp = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())

    # parametric initializations ****************** GMWM info *****************************************************************
    # gmwm info specifications 
    inExcel = 'subjBrainDataNarrGen_360plain.xlsx'
    headerIdxs = [0, 1, 2, 3, 4]
    gmwmDimensions = ('area', 'thickness', 'cortical_curvature',
                        'volume', 'connectivity_density')
    uniqueHeaders = ('Site_ID', 'TICV')
    otherFeatureHeaders=('TICV',)
    # input
    demoExcel = 'XXL_ACEcohort395.xlsx'
    # output --> LOADE data
    outSavePkl='./data/gmwm3D'

    print("demo-process-end")


    # ******* ACQUIRE associated GMWM data -> get 3D data from the originals ************************************************
    # brainData = loadmat('./data/dataOrg.mat', squeeze_me=True)
    # haha=loadmat1('./data/dataOrg.mat')
    # ------> (important!) cook gmwm data with 3D & save them **********************
    # data=cookGMWM_withDemo(inExcel,headerIdxs,gmwmDimensions, uniqueHeaders, demoExcel, otherFeatureHeaders, outSavePkl) #

    ###########################################################################################################################

    # load 3D gmwm data --------------------------------------------------------------
    data = loadPkl_file(outSavePkl)

    gmwm3D = data['gmwm3D']
    gmwm3D_extend=data['gmwm3D_extend']
    gmwmDF = data['df']
    gmwmDim = data['gmwmDimensions']
    uniqueHeaders = data['uniqueHeaders']
    otherFeaturesH = data['otherFeatureHeaders']
    siteList = data['siteList']

    # specify X and y
    X=gmwm3D
    y=gmwmDF['ace']

    # ******* process assocaited data with Neural-network *******************************************************************
    # train specifications ----------------------------------------------------------
    data_dim=3
    #agentStructureName='CNN6'
    agentStructureName='CNN4'
    #agentStructureName='brainSpatialAwareBinary'
    #agentStructureName='get_image_model'


    epochNum=100


    # model cook ---------------------------------------------------------------------------------

    modelDir="./models/"
    # prepare 3D train-vs-test datasets 
    trainX, y_train, testX, y_test=train_test_splitProp3D(X, y, 0.85)

    # dimension cook
    subjNum,feaNum,_=trainX.shape
    if data_dim==4:
        trainX=trainX.reshape(trainX.shape[0],trainX.shape[1],trainX.shape[2],1)
        testX=testX.reshape(testX.shape[0],testX.shape[1],testX.shape[2],1)

    in_shape=trainX.shape[1:]

    # get model
    model=eval(f'{agentStructureName}(in_shape,feaNum)')

    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss', # val_los
        patience=10,
        restore_best_weights=True
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.8,
        patience=10,
        min_lr=1e-6
    )

    checkpoint = ModelCheckpoint(
        f'{agentStructureName}_best.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )

    # 4. model-train
    history = model.fit(trainX, y_train, epochs=epochNum, batch_size=16, 
                       validation_split=0.2, verbose=1, 
                       callbacks=[early_stopping, reduce_lr, checkpoint])

    # Plot training history
    plot_training_history(history, save_path=f'{modelDir}{agentStructureName}_training_history_{timeStamp}.png')

    # 5. model-evaluate
    (loss, acc) =model.evaluate(trainX,y_train) 
    #(loss, mae, acc, _) = model.evaluate(trainX, y_train)
    # if data_dim == 4 or agentStructureName =='brainClassifier1':
    #     (loss,mae,acc) =model.evaluate(trainX,y_train)

    # if data_dim == 3:
    #     (loss, mae, acc, _) = model.evaluate(trainX, y_train)

    
    print(f'Model Accuracy: {acc:.4f}')

    # 6. model-prediction
    Y_pred_prob = model.predict(testX)
   #Y_pred_class = (Y_pred_prob > 0.5).astype(int)
    Y_pred_class = np.argmax(Y_pred_prob,axis=1)

    accuracy = accuracy_score(y_test, Y_pred_class)
    print(f"Accuracy: {accuracy:.2f}")  # 输出: 0.80

    cm = confusion_matrix(y_test, Y_pred_class)
    print("Confusion matrix:\n", cm)

    report = classification_report(y_test, Y_pred_class)
    print("Classification report:\n", report)

    print("haha")