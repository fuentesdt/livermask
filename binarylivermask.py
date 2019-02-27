# usage:
# 
#   python ./binarylivermask.py --predictimage=./Ven.raw.nii.gz --segmentation=./onehot.nii.gz 
# 

import numpy as np

# raw dicom data is usually short int (2bytes) datatype
# labels are usually uchar (1byte)
IMG_DTYPE = np.int16
SEG_DTYPE = np.uint8

# setup command line parser to control execution
from optparse import OptionParser
parser = OptionParser()
parser.add_option( "--predictmodel",
                  action="store", dest="predictmodel", default='/rsrch1/ip/dtfuentes/github/livermask/modelout/TrainingData/001/000/tumormodelunet.json',
                  help="apply model to image", metavar="Path")
parser.add_option( "--c3dexe",
                  action="store", dest="c3dexe", default='/usr/local/bin/c3d',
                  help="c3d executable", metavar="Path")
parser.add_option( "--predictimage",
                  action="store", dest="predictimage", default=None,
                  help="apply model to image", metavar="Path")
parser.add_option( "--segmentation",
                  action="store", dest="segmentation", default=None,
                  help="model output ", metavar="Path")
parser.add_option( "--trainingresample",
                  type="int", dest="trainingresample", default=256,
                  help="setup info", metavar="int")
(options, args) = parser.parse_args()


# FIXME:  @jonasactor - is there a better software/programming practice to keep track  of the global variables?
_globalexpectedpixel=512

##########################
# apply model to new data
##########################
if (options.predictmodel != None and options.predictimage != None and options.segmentation != None and options.c3dexe != None ):
  import json
  import nibabel as nib
  import skimage.transform
  # force cpu for debug
  import os
  os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
  # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
  from keras.models import model_from_json
  # load json and create model
  _glexpx = _globalexpectedpixel
  json_file = open(options.predictmodel, 'r')
  loaded_model_json = json_file.read()
  json_file.close()
  loaded_model = model_from_json(loaded_model_json)
  # load weights into new model
  weightsfile= '.'.join(options.predictmodel.split('.')[0:-1]) + '.h5'
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
  
  # apply 2d model to all slices
  segout = loaded_model.predict(resizepredict[:,:,:,np.newaxis] )

  # write out each one-hot image
  numlabel = segout.shape[-1]
  for jjj in range(numlabel):
      segout_resize = skimage.transform.resize(segout[...,jjj],(nslice,_glexpx,_glexpx),order=0,preserve_range=True,mode='constant').transpose(2,1,0)
      segout_img = nib.Nifti1Image(segout_resize, None, header=imageheader)
      segout_img.to_filename( options.segmentation.replace('.nii.gz', '-%d.nii.gz' % jjj) )

  # post processing
  postprocessingcmd = '%s %s -vote  -binarize -o %s -comp -thresh 1 1 1 0 -o %s' % (options.c3dexe, options.segmentation.replace('.nii.gz', '-?.nii.gz' ), options.segmentation.replace('.nii.gz', 'binarize.nii.gz' ), options.segmentation)

  print(postprocessingcmd)
  os.system (postprocessingcmd )

#########################
# print help
#########################
else:
  parser.print_help()
