# encoding=utf-8
import numpy as np
import os
import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding
import seaborn as sns
import matplotlib.pyplot as plt
import pylab
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import time  # time-marker
import pymysql


class ClassifyEnv(gym.Env):

    # mode means training or testing
    def __init__(self, mode, class_rate, trainx, trainy, model, printInfo, outName):

        self.mode = mode
        self.class_rate = class_rate
        self.model = model

        self.Env_data = trainx
        self.Answer = trainy
        self.id = np.arange(trainx.shape[0])

        self.game_len = self.Env_data.shape[0]

        self.num_classes = len(set(self.Answer))
        self.action_space = spaces.Discrete(self.num_classes)
        print(self.action_space)
        self.step_ind = 0
        self.y_pred = []

        self.printInfo = printInfo
        self.outName = outName

        #print("self:\n",self)
        #breakpoint()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, a):
        '''
        self.Answer[self.id]==y_train
        '''
        np.seterr(divide='ignore', invalid='ignore')
        self.y_pred.append(a)  # here is the predicted class
        cm = confusion_matrix(
            self.Answer[self.id][:len(self.y_pred)], self.y_pred)
        # Now the normalize the diagonal entries
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # The diagonal entries are the accuracies of each class
        percentPosArr = np.array(cm.diagonal())
        percentNegArr = 1.-np.array(cm.diagonal())
        y_true_cur = []
        info = {}
        terminal = False
        rate = self.class_rate[int(self.Answer[self.id[self.step_ind]])]
        if len(percentNegArr) == len(self.class_rate):
            perIncorr = percentNegArr[int(self.Answer[self.id[self.step_ind]])]
        '''
        assign reward/punishment
        '''
        if a == self.Answer[self.id[self.step_ind]]:  # if the answer is correct
            if len(percentNegArr) == len(self.class_rate):
                reward = 1. / rate + perIncorr / rate
            else:
                reward = 1. / rate
        else:
            if len(percentNegArr) == len(self.class_rate):  # if the answer is incorrect
                reward = -1. / rate - perIncorr / rate
                if self.mode == 'train':
                    terminal = True
            else:
                reward = -1./rate

        self.step_ind += 1

        if self.step_ind == self.game_len - 1:

            y_true_cur = self.Answer[self.id]
            info['gmean'], info['fmeasure'] = self.My_metrics(np.array(self.y_pred),
                                                              np.array(y_true_cur[:self.step_ind]))
            terminal = True

        #print("self.y_pred:\n",self.y_pred)
        #print("info:\n",info)
        #accTrain=accuracy_score(np.array(self.Answer[self.id]),np.array(self.y_pred))
        #print("[trainAcc]: ",accTrain)
        #print("[self]: ",self)

        return self.Env_data[self.id[self.step_ind]], reward, terminal, info

    def My_metrics(self, y_pre, y_true):
        confusion_mat = confusion_matrix(y_true, y_pre)
        np.seterr(divide='ignore', invalid='ignore')

        FP = confusion_mat.sum(axis=0) - np.diag(confusion_mat)
        FN = confusion_mat.sum(axis=1) - np.diag(confusion_mat)
        TP = np.diag(confusion_mat)
        TN = confusion_mat.sum() - (FP + FN + TP)

        FP = FP.astype(float)
        FN = FN.astype(float)
        TP = TP.astype(float)
        TN = TN.astype(float)
        '''
        '''
        TPrate = TP / (TP + FN)  # 真阳性率
        TNrate = TN / (TN + FP)  # 真阴性率[*]
        FPrate = FP / (TN + FP)  # 假阳性率[*]
        FNrate = FN / (TP + FN)  # 假阴性率
        #PPvalue = TP / (TP + FP)  # 阳性预测值[*]
        #NPvalue = TN / (TN + FN)  # 假性预测值

        G_mean = np.sqrt(TPrate * TNrate)

        #Recall = TPrate = TP / (TP + FN)
        Recall = np.diag(confusion_mat)/np.sum(confusion_mat, axis=1)
        Precision = np.diag(confusion_mat) / np.sum(confusion_mat, axis=0)
        F_measure = 2 * Recall * Precision / (Recall + Precision)
        print("\n", confusion_mat)
        print(classification_report(y_true, y_pre))
        # extract time & save associated output(s)
        # now = time.strftime("%Y-%m-%d_%H-%M-%S")
        now = time.strftime("%Y-%m-%d_%H_%M_%S")
        print(now)
        if self.mode == 'test':
            #modeNum=self.mode.replace('testBal', '')
            #outName = 'testOut'+now
            '''
            # get the associated outputs
            # connect database
            dbconn=pymysql.connect(
              host="localhost",
              database="test",
              user="root",
              password="leixiaoxiao",
              port=3306,
              charset='utf8'
            )
            # initiate cursor execution
            cursor=dbconn.cursor()
            # get the command
            sqlUpdate1=f'INSERT INTO `freqT` (`var1`,`var2`,`var3`,`var4`,`time`) VALUES ({y_true[0]},{y_true[1]},{y_pre[0]},{y_pre[1]},\'{now}\')'
            cursor.execute(sqlUpdate1)
            dbconn.commit()
            cursor.close()
            dbconn.close()

            # manage the output Predict_Test_L2_B71M_CUN_M_RL1
            dbconn=pymssql.connect('(local)','sa','huarun','master')
            cursor=dbconn.cursor()
            sqlUpdate1="""INSERT INTO dbo.Predict_Test_"""+targetPredVar+"""RL1 (pred_start,pred_end,curr_time,action) VALUES (%s,%s,%s,%s)"""
            #print(sqlUpdate1)
            cursor.execute(sqlUpdate1,(foreStart,foreEnd,currTime,actionVal))
            dbconn.commit()
            cursor.close()
            dbconn.close()
            '''
            # try to plot the associated confusion matrix
            sns.set()
            f, ax = plt.subplots()
            c2 = confusion_matrix(y_true, y_pre)
            sns.heatmap(c2, annot=True, ax=ax, cmap="mako", fmt='.5g', annot_kws={
                        "fontsize": round(72/len(self.class_rate))})
            ax.set_title("confusion matrix")
            ax.set_xlabel("predict")
            ax.set_ylabel("true")
            #ax.set_xticklabels(fontsize=round(72/len(self.class_rate)))
            #ax.set_yticklabels(fontsize=round(72/len(self.class_rate)))
            outFigName = f"./models/rlSave/{self.outName}.png"
            pylab.savefig(outFigName)
            pylab.show()
            # try to through the associated output
            import sys
            savedStdout = sys.stdout  # 保存标准输出流
            outTxt = f"./models/rlSave/{self.outName}.txt"
            with open(outTxt, 'wt') as file:
                sys.stdout = file  # 标准输出重定向至文件
                print("\n", self.model.summary())
                print("\n", self.printInfo)
                print("\n", confusion_mat)
                print(classification_report(y_true, y_pre))

            sys.stdout = savedStdout  # 恢复标准输出流

        res = 'G-mean:{}, F_measure:{}\n'.format(G_mean, F_measure)

        acc = accuracy_score(y_true, y_pre)
        G_mean = np.nan_to_num(G_mean, copy=True, nan=0.0,
                               posinf=None, neginf=None)
        F_measure = np.nan_to_num(
            F_measure, copy=True, nan=0.0, posinf=None, neginf=None)
        G_mean = G_mean.all()/len(G_mean)
        #G_mean=G_mean.all()
        F_measure = F_measure.all()/len(F_measure)
        #F_measure=F_measure.all()
        # if this is testing example --> plot the confusion matrix
        return G_mean, F_measure

    def reset(self):
        if self.mode == 'train':
            np.random.shuffle(self.id)
        self.step_ind = 0
        self.y_pred = []
        return self.Env_data[self.id[self.step_ind]]
        self.y_pred = []
        return self.Env_data[self.id[self.step_ind]]
