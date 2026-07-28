from scipy.io import loadmat
import scipy.io as sio
import json
import pandas as pd
import numpy as np
import pickle

import sys
sys.path.append('./funcs')
sys.path.append('./data')
from funcs.tools import *
from funcs.brainCook import *

'''
##### PURPOSE:
    - organize GMWM data to 3D formates

'''



def all_unique_typeItems(t):
    return len(t) == len(set(t))


if __name__ == "__main__":

   # parametric initializations ****************** GMWM info *****************************************************************
   inExcel = 'subjBrainDataNarrGen_360plain.xlsx'
   headerIdxs = [0, 1, 2, 3, 4]
   gmwmDimensions = ('area', 'thickness', 'cortical_curvature',
                     'volume', 'connectivity_density')
   uniqueHeaders = ('Site_ID', 'TICV')
   otherFeatureHeaders=('TICV',)

   demoExcel = 'XXL_ACEcohort395.xlsx'
   
   outSavePkl='./data/gmwm3D'

   print("demo-process-end")


   # ******* ACQUIRE associated GMWM data ************************************************
   # brainData = loadmat('./data/dataOrg.mat', squeeze_me=True)
   # haha=loadmat1('./data/dataOrg.mat')
   # ------> (important!) cook gmwm data with 3D & save them **********************
   # data=cookGMWM_withDemo(inExcel,headerIdxs,gmwmDimensions, uniqueHeaders, demoExcel, otherFeatureHeaders, outSavePkl) #

   #########################################################
   
   # load 3D gmwm data
   data = loadPkl_file(outSavePkl)

   gmwm3D = data['gmwm3D']
   gmwm3D_extend=data['gmwm3D_extend']
   gmwmDF = data['df']
   gmwmDim = data['gmwmDimensions']
   uniqueHeaders = data['uniqueHeaders']
   otherFeaturesH = data['otherFeatureHeaders']
   siteList = data['siteList']

   print("haha")
