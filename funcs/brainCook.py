import pandas as pd
import numpy as np
from tools import *

# get complete list of GMWM features
def getGMWM_featureName(region, headers, dim):

   outList = []
   dimList = list(dim)
   dimTarget = dimList.pop(0)

   for header in headers:
      pian = header[0]
      dimI = header[1]
      side = header[2]
      chunk = header[3]
      fea = header[4]
      if pian == region[0] and side == region[1] and chunk == region[2]:
         if dimI == dimTarget:
            outList.append(fea)
            if len(dimList) == 0:
               return outList
            else:
               dimTarget = dimList.pop(0)
               continue

   return outList


# process GMWM features with demographic data
def cookGMWM_withDemo(inExcel,headerIdxs,gmwmDimensions, uniqueHeaders, demoExcel, otherFeatureHeaders,outSavePkl=None):
   # *******************************************************************************************************************************************
   #brainData = loadmat('./data/dataOrg.mat', squeeze_me=True)
   #haha=loadmat1('./data/dataOrg.mat')
   dfOrg = pd.read_excel(inExcel, header=headerIdxs)
   dfDemo = pd.read_excel(demoExcel, header=[0])

   # extract header
   headersOrg = dfOrg.columns

   headerLen = len(headersOrg[0])
   headersReplace = {}
   oneRowHeaderList = []
   columnIdxInfo = {}

   for orgHeader in headersOrg:
      condenseHeader = (orgHeader[headerLen-1])
      headersReplace[orgHeader] = condenseHeader
      oneRowHeaderList.append(condenseHeader)

   # simplify column-index
   df = dfOrg
   df.rename(columns=headersReplace, inplace=True)
   df.columns = oneRowHeaderList

   # manage site-list + column alignments btw df-vs-dfDemo
   siteList = df[uniqueHeaders[0]]
   # sort df-demo same as dfgmwm
   dfDemo = dfDemo.set_index("newids").loc[siteList].reset_index()
   asdClass = [(1 if cond == 'ASD' else 0) for cond in dfDemo['Cohort']] # ASD - 1; others - 0
   genderClass = [(1 if cond == 'M' else 0) for cond in dfDemo['Gender']] # male -1; female - 0
   df['ace'] = asdClass
   df['gender'] = genderClass
   uniqueHeaders += ('ace',)
   uniqueHeaders += ('gender',)
   # add demographic features
   for col in dfDemo.columns:
      if 'DAS-' in col:
         df[col]=dfDemo[col]
         uniqueHeaders+=(col,)
         otherFeatureHeaders+=(col,)

   # get values
   # values = np.array(df.values)

   # data = json.dumps(brainData, indent=2)
   # manage 3D ******************************************
   # oragnize brain regions
   regionList = []
   baba = []
   headers = headersOrg
   for idx, header in enumerate(headersOrg):

      #if all_unique_typeItems(header):
      if oneRowHeaderList[idx] not in uniqueHeaders:
         region = header[0]
         if 'Unnamed' in header[2]:
            headerList = list(header)
            headerList[2] = 'middle'
            newHeader = tuple(headerList)
            headers = pd.MultiIndex.from_tuples(
                [newHeader if t == header else t for t in headers])
            side = 'middle'
         else:
            side = header[2]
         chunk = header[3]
         pile = (region, side, chunk)
         if pile not in regionList:
            regionList.append(pile)

   # compile 3D data -------- GMWM only
   ticv = df[uniqueHeaders[1]]
   matrix_3Dgmwm = np.empty((0, len(regionList), len(gmwmDimensions)))
   matrix_3Dgmwm_demo = np.empty((0,len(regionList)+len(otherFeatureHeaders),len(gmwmDimensions)))

   for siteId in siteList:
      subjMatrix = np.empty((0, len(gmwmDimensions)))
      
      # gmwm regions
      for region in regionList:
         col_feaIdxs = getGMWM_featureName(region, headers, gmwmDimensions)
         col_feaIdxs.append(uniqueHeaders[0])
         region2D = df[col_feaIdxs]
         subjRegion2D_df = region2D[region2D[uniqueHeaders[0]] == siteId]
         subjRegion2D_df = subjRegion2D_df.drop(
             subjRegion2D_df.columns[-1], axis=1)
         subRegion2D = np.array(subjRegion2D_df)
         subjMatrix = np.vstack((subjMatrix, subRegion2D))

      subMat3D = subjMatrix[np.newaxis, :, :]
      matrix_3Dgmwm = np.vstack((matrix_3Dgmwm, subMat3D))

      # other features
      subMat_extend=subjMatrix
      for col in otherFeatureHeaders:
         col_Idxs=[col,uniqueHeaders[0]]
         xFeas=df[col_Idxs]
         xF=xFeas[xFeas[uniqueHeaders[0]] == siteId]
         xF=xF.drop(xF.columns[-1],axis=1)
         xF_val=xF.values
         xF_valRep=np.full((1,len(gmwmDimensions)),xF_val)
         subMat_extend=np.vstack((subMat_extend,xF_valRep))
      
      subMat3D_extend=subMat_extend[np.newaxis,:,:]
      matrix_3Dgmwm_demo=np.vstack((matrix_3Dgmwm_demo,subMat3D_extend))


      print("subSiteId_print: ", siteId)

   # compile 3D data ---------- GMWM + ticv/demographic features (5-times duplication)
   #matrix_3Dgmwm_complete=np.empty((0, len(regionList)+, len(gmwmDimensions))

   # loop through values
   #for idx, value in enumerate(headersOrg):

   # compile data
   data = {
      'gmwm3D': matrix_3Dgmwm,
      'gmwm3D_extend':matrix_3Dgmwm_demo,
      'df': df,
      'gmwmDimensions': gmwmDimensions,
      'uniqueHeaders': uniqueHeaders,
      'otherFeatureHeaders': otherFeatureHeaders,
      'siteList': siteList
   }

   # save data
   if outSavePkl != None:
      savePkl_file(data,outSavePkl)


   return data