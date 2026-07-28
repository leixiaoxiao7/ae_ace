# coding=utf-8
import argparse
import os
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
import numpy as np
from numpy import zeros, newaxis
from tensorflow.keras.optimizers import Adam
from keras.models import Model
from sklearn import preprocessing
from rl.core import Processor
from rl.processors import MultiInputProcessor
import pandas as pd
import pymysql
from datetime import datetime
import sys
sys.path.append('./funcs')
sys.path.append('./data')
from funcs.dataFuncs import *
from funcs.dataCook import *
from funcs.tools import *
from funcs.ICMDP_Env_xxl_multiClass import ClassifyEnv
from funcs.get_agentModel import *
from funcs.agentSectionSpec import getAgentFactors, buildDQN

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["SM_FRAMEWORK"] = "tf.keras"


class ClassifyProcessor(Processor):
    def process_observation(self, observation):
        img = observation.reshape(INPUT_SHAPE)
        processed_observation = np.array(img)
        return processed_observation

    def process_state_batch(self, batch):
        batch = batch.reshape((-1,) + INPUT_SHAPE)
        processed_batch = batch.astype('float32') / 1.
        return processed_batch

    def process_reward(self, reward):
        #prefent [reward/punishment outcrash], control the reward/punishment within certain arrange
        return np.clip(reward, -30000., 30000.)


if __name__ == '__main__':

    begin_time = datetime.now()
    outSavePkl='./data/gmwm3D'
    replicate = 4

    # -------------------------[parametric specifications]-----------------
    #data=readPklMode('./data/reducedGMWM.pkl')
    data=loadPkl_file(outSavePkl)

    #gmwm3D=data['gmwm3D_reduced']
    gmwm3D=data['gmwm3D']
    gmwmDF=data['df']
    gmwmDim=data['gmwmDimensions']
    uniqueHeaders=data['uniqueHeaders']
    siteList=data['siteList']
    data_dim=3
    agentStructureName='CNN'

    X = gmwm3D
    targets = gmwmDF['ace']
    # manage dataset replication if necessary
    if replicate>1:
        # manage replicate
        X=np.repeat(X,replicate,axis=0)
        targets=np.repeat(targets.values,replicate,axis=0)
        # manage randomization
        randIdx=[i for i in range(len(targets))]
        np.random.shuffle(randIdx)
        X=X[randIdx]
        targets=targets[randIdx]
        print("haha --- mark over here")

    trainProp = 0.8
    printAndSave = True
    '''
    [IMPORTANT] parametric setting
    [@@@]
    '''
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    #parser = argparse.ArgumentParser()
    training_steps = 100000 # the # of steps during training
    data_dim = 3
    condition = 'brain_data_classify'
    printInfo = condition+'\n'
    #agentModeName = 'get_image_model2'
    agentModeName='CNN1'

    GAMMA = 0.3
    BATCH_SIZE = 16
    MEMORY_LIMIT = 500000
    VALUE_TEST = .1

    '''
    [reinforcement-agents]
    (2):
    get_simpleMLP

    (3):
    get_simpleMLP3D
    get_simpleLSTM
    basicCNN
    CNN1
    get_convClassify_modelDraft1D_1

    (4):
    get_image_model
    get_image_model2
    '''

    # test model & output
    '''
    /////////////////////////////////////////////////////////////////////////////////////////////////////

    '''
    # output-mode-save
    outName = condition+'_advancedModelRI_'+agentModeName+'_'+now
    checkpoint_path = "train_"+outName
    checkpoint_path += "/cp-{epoch:04d}.ckpt"

    #[1] preprocessing
    printInfo = "lxx"
    # get the data & do associated cook
    #X, targets, printInfo = preProcessData()
    #printInfo += f'variables:{X.columns}\n'
    X = np.array(X)
    #targets = np.array(targets)-1

    # cooking assocaited data
    x_train, y_train, x_test, y_test = train_test_splitProp3D(
        X, targets, trainProp)
    '''
    x_train = np.array(x_train).astype('float64')
    x_test = np.array(x_test).astype('float64')
    y_train = np.array(y_train).astype('int32')
    y_test = np.array(y_test).astype('int32')
    '''
    # scale the test data to a balanced version
    #xT1, yT1 = resolve_imb_data(x_test, y_test, 1)

    subjNum, feaNum, _ = X.shape
    printInfo += f'sampleSize:{subjNum}\n'
    #feaNum += 2
    # split the train & test data
    '''
    if [CONVOLUTION] is necessary --> do something
    '''
    # parameter specification
    num_classes = len(set(targets))
    classRate = get_classRate(targets)
    printInfo += f'condition:[f{condition}{num_classes}_classify]\n'

    if data_dim == 4:
        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_train.shape[2], 1)

    in_shape = x_train.shape[1:]

    print("set_classes: ", set(targets))
    print("num_classes: ", num_classes)

    '''
    [3] specify agent associated parameters
    '''
    mode = 'train'
    model = eval(f'{agentModeName}(in_shape,feaNum,num_classes)')
    env = ClassifyEnv(mode, classRate, x_train, y_train, '', '', '')
    INPUT_SHAPE = in_shape
    print("in_shape: ", in_shape)
    print("[model_summary]: \n", model.summary())
    # build associated agent(s)
    memory, policy = getAgentFactors(MEMORY_LIMIT, VALUE_TEST)
    processor = ClassifyProcessor()
    '''
    [agents overview] keras-rl.readthedocs.io/en/latest/agents/overview/
    '''
    dqn = buildDQN(model, num_classes, policy, memory,
                   processor, GAMMA, BATCH_SIZE)

    # if need to save --> then save
    if printAndSave:
        # save the associated data-structure
        # have 'epoch' included within files (use `str.format`)
        checkpoint_dir = os.path.dirname(checkpoint_path)
        # create a [feedback-adjustment-loop], for every 'period' of iterations, manage the running
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            verbose=1,
            save_weights_only=True,
            period=10)
        dqn.save_weights(checkpoint_path.format(epoch=0), overwrite=True)

    # train model
    dqn.fit(env, nb_steps=training_steps, log_interval=80000, verbose=1)

    # test model & output
    # add print info/output
    printInfo += f'[agentModelName]: {agentModeName}\n[MEMORY_LIMIT]: {MEMORY_LIMIT}\n[VALUE_TEST]: {VALUE_TEST}\n[training_steps]: {training_steps}\n[GAMMA]: {GAMMA}\n[BATCH_SIZE]: {BATCH_SIZE}\n[running-condition]:{condition}'
    printInfo += '\n[dataset]:'+outSavePkl
    end_time = datetime.now()

    # print("[测试--on-test-BALANCED_1]")
    # env = ClassifyEnv(mode, classRate, x_test, y_test, model,
    #                   printInfo+f'\n[testmode]:testBal', (outName+'testBal') if printAndSave else '')
    # env.mode = 'testBal'
    # dqn.test(env, nb_episodes=1, visualize=False)


    print("[测试-on-TEST]")
    env = ClassifyEnv(mode, classRate, x_test, y_test,
                      model, printInfo+f'\n[testmode]:test', (outName+'test') if printAndSave else '')
    env.mode = 'test'
    dqn.test(env, nb_episodes=1, visualize=False)


    print("run_time: ", (end_time-begin_time))
