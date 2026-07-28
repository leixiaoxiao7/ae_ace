from scipy.io import loadmat
import scipy.io as sio
import json

import sys
#sys.path.append('./agents')
#sys.path.append('./funcs')
sys.path.append('./data')

def _check_keys( dict):
   """
   checks if entries in dictionary are mat-objects. If yes
   todict is called to change them to nested dictionaries
   """
   for key in dict:
      if isinstance(dict[key], sio.matlab.mio5_params.mat_struct):
         dict[key] = _todict(dict[key])
   return dict


def _todict(matobj):
    """
    A recursive function which constructs from matobjects nested dictionaries
    """
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, sio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict


def loadmat1(filename):
    """
    this function should be called instead of direct scipy.io .loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    """
    data = loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


if __name__ == "__main__":

   #brainData = loadmat('./data/dataOrg.mat', squeeze_me=True)
   haha=loadmat1('./data/dataOrg.mat')

   #data = json.dumps(brainData, indent=2)
   
   print("haha")
