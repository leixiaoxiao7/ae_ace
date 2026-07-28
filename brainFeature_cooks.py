

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
