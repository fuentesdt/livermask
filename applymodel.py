# usage:
# 
#   python ./applymodel.py --predictimage=./Ven.raw.nii.gz --segmentation=./onehot.nii.gz 
#   python ./applymodel.py --predictimage=./Ven.raw.nii.gz --maskimage=./onehot.nii.gz --segmentation=./tumor.nii.gz  --crcmodel
#   python ./applymodel.py --predictimage=./Ven.raw.nii.gz --maskimage=./onehot.nii.gz --segmentation=./tumor.nii.gz  --hccmodel
# 

import numpy as np
import os

# raw dicom data is usually short int (2bytes) datatype
# labels are usually uchar (1byte)
IMG_DTYPE = np.int16
SEG_DTYPE = np.uint8

# setup command line parser to control execution
from optparse import OptionParser
parser = OptionParser()
parser.add_option( "--modelzoo",
                  action="store", dest="modelzoo", default='/rsrch1/ip/dtfuentes/github/livermask/ModelZoo/',
                  help="model zoo location", metavar="Path")
parser.add_option( "--modelpath",
                  action="store", dest="modelpath", default=None,
                  help="model location", metavar="Path")
parser.add_option( "--crcmodel",
                  action="store_true", dest="crcmodel", default=False,
                  help="use crc model", metavar="FILE")
parser.add_option( "--hccmodel",
                  action="store_true", dest="hccmodel", default=False,
                  help="use hcc model", metavar="FILE")
parser.add_option( "--c3dexe",
                  action="store", dest="c3dexe", default='/usr/local/bin/c3d',
                  help="c3d executable", metavar="Path")
parser.add_option( "--predictimage",
                  action="store", dest="predictimage", default=None,
                  help="apply model to image", metavar="Path")
parser.add_option( "--maskimage",
                  action="store", dest="maskimage", default=None,
                  help="image mask", metavar="Path")
parser.add_option( "--segmentation",
                  action="store", dest="segmentation", default=None,
                  help="model output ", metavar="Path")
parser.add_option( "--trainingresample",
                  type="int", dest="trainingresample", default=256,
                  help="setup info", metavar="int")
(options, args) = parser.parse_args()

if (options.modelpath != None ):
  modelpath= options.modelpath
else:
  modelpath= "%s/livermask/tumormodelunet.json" % options.modelzoo
  if (options.crcmodel):
    modelpath= "%s/crcmets/tumormodelunet.json" % options.modelzoo
  elif (options.hccmodel):
    modelpath= "%s/hcclesion/tumormodelunet.json" % options.modelzoo

print("using %s " % modelpath)

assert os.path.isfile(modelpath)

# FIXME:  @jonasactor - is there a better software/programming practice to keep track  of the global variables?
_globalexpectedpixel=512

##########################
# apply model to new data
##########################
if (options.predictimage != None and options.segmentation != None and options.c3dexe != None ):
  import json
  import nibabel as nib
  import skimage.transform
  # force cpu for debug
  import os
  os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
  # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

  import keras
  import tensorflow as tf
  print("keras version: ",keras.__version__, 'TF version:',tf.__version__)

  from keras.models import model_from_json
  # load json and create model
  _glexpx = _globalexpectedpixel
  json_file = open(modelpath, 'r')
  loaded_model_json = json_file.read()
  json_file.close()
  loaded_model = model_from_json(loaded_model_json)
  # load weights into new model
  weightsfile= '.'.join(modelpath.split('.')[0:-1]) + '.h5'
  loaded_model.load_weights(weightsfile)
  print("Loaded model from disk")

  imagepredict = nib.load(options.predictimage)
  imageheader  = imagepredict.header
  numpypredict = imagepredict.get_data().astype(IMG_DTYPE )
  # error check
  assert numpypredict.shape[0:2] == (_glexpx,_glexpx)
  nslice = numpypredict.shape[2]
  print('nslice = %d' % nslice)
  resizepredict = skimage.transform.resize(numpypredict,(options.trainingresample,options.trainingresample,nslice ),order=0,preserve_range=True,mode='constant').astype(IMG_DTYPE).transpose(2,1,0)
  
  if (options.maskimage != None):
     imagemask = nib.load(options.maskimage)
     maskheader  = imagemask.header
     numpymask = imagemask.get_data().astype(SEG_DTYPE )
     # error check
     assert numpymask.shape[0:2] == (_glexpx,_glexpx)
     masknslice = numpymask.shape[2]
     assert masknslice  ==  nslice
     resize_mask = skimage.transform.resize(numpymask,(options.trainingresample,options.trainingresample,nslice ),order=0,preserve_range=True,mode='constant').astype(IMG_DTYPE).transpose(2,1,0)

     # bind the image and mask
     predict_vector = np.repeat(resizepredict [:,:,:,np.newaxis],2,axis=3)
     predict_vector [:,:,:,1]=resize_mask 

     # apply 2d model to all slices
     segout = loaded_model.predict(predict_vector[slice(0,nslice),:,:,:]  )

     # post processing
     postprocessingcmd = '%s -verbose %s  -scale .5  %s  -scale .5 %s  -vote -o %s' % (options.c3dexe,  options.segmentation.replace('.nii.gz', '-0.nii.gz' ),  options.segmentation.replace('.nii.gz', '-1.nii.gz' ), options.segmentation.replace('.nii.gz','-[23456789].nii.gz'), options.segmentation)

  else:
     # apply 2d model to all slices
     segout = loaded_model.predict(resizepredict[:,:,:,np.newaxis] )

     # post processing
     postprocessingcmd = '%s -verbose %s -vote  -binarize -o %s -comp -thresh 1 1 1 0 -o %s' % (options.c3dexe, options.segmentation.replace('.nii.gz', '-?.nii.gz' ), options.segmentation.replace('.nii.gz', 'binarize.nii.gz' ), options.segmentation)

  # make sure directory exists
  mkdircmd = "mkdir -p %s" % "/".join(options.segmentation.split("/")[:-1])
  print(mkdircmd )
  os.system (mkdircmd )

  # write out each one-hot image
  numlabel = segout.shape[-1]
  for jjj in range(numlabel):
      segout_resize = skimage.transform.resize(segout[...,jjj],(nslice,_glexpx,_glexpx),order=0,preserve_range=True,mode='constant').transpose(2,1,0)
      segout_img = nib.Nifti1Image(segout_resize, None, header=imageheader)
      segout_img.to_filename( options.segmentation.replace('.nii.gz', '-%d.nii.gz' % jjj) )

  print(postprocessingcmd)
  os.system (postprocessingcmd )

#########################
# print help
#########################
else:
  parser.print_help()
