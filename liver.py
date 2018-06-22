import numpy as np

IMG_DTYPE = np.float32
SEG_DTYPE = np.uint8
# run liver.py --builddb
# run liver.py --buildmodel=0 --kfolds=5
# run liver.py --buildmodel=1 --kfolds=5
# run liver.py --buildmodel=2 --kfolds=5
# run liver.py --buildmodel=3 --kfolds=5
# run liver.py --buildmodel=4 --kfolds=5
# for idbatch in 10 100 ; do for idres in 128 256 ; do for idtr in run_a run_b run_c run_d run_e run_f run_g run_h ; do python Code/loadtraining.py --buildmodel=UID/anonymizetrain.npy --trainingid=$idtr  --trainingresample=$idres --trainingbatch=$idbatch;done; done; done
# run Code/loadtraining.py --predictimage=UID/Art.raw.nii.gz --predictmodel=UID/tumormodelunet.json --segmentation=test.nii.gz
# setup command line parser to control execution
from optparse import OptionParser
parser = OptionParser()
parser.add_option( "--builddb",
                  action="store_true", dest="builddb", default=False,
                  help="FILE containing image info", metavar="FILE")
parser.add_option( "--debug",
                  action="store_true", dest="debug", default=False,
                  help="compare tutorial dtype", metavar="Bool")
parser.add_option( "--ModelID",
                  action="store", dest="modelid", default=None,
                  help="model id", metavar="FILE")
parser.add_option( "--outputModelBase",
                  action="store", dest="outputModelBase", default=None,
                  help="output location ", metavar="Path")
parser.add_option( "--inputModelData",
                  action="store", dest="inputModelData", default="datalocation/TrainingImages.npy",
                  help="database file", metavar="Path")
parser.add_option( "--predictmodel",
                  action="store", dest="predictmodel", default=None,
                  help="apply model to image", metavar="Path")
parser.add_option( "--predictimage",
                  action="store", dest="predictimage", default=None,
                  help="apply model to image", metavar="Path")
parser.add_option( "--segmentation",
                  action="store", dest="segmentation", default=None,
                  help="model output ", metavar="Path")
parser.add_option( "--anonymize",
                  action="store", dest="anonymize", default=None,
                  help="setup info", metavar="Path")
parser.add_option( "--trainingid",
                  action="store", dest="trainingid", default='run_a',
                  help="setup info", metavar="Path")
parser.add_option( "--trainingmodel",
                  action="store", dest="trainingmodel", default='half',
                  help="setup info", metavar="string")
parser.add_option( "--trainingloss",
                  action="store", dest="trainingloss", default='dscimg',
                  help="setup info", metavar="string")
parser.add_option( "--trainingsolver",
                  action="store", dest="trainingsolver", default='adadelta',
                  help="setup info", metavar="string")
parser.add_option( "--trainingresample",
                  type="int", dest="trainingresample", default=128,
                  help="setup info", metavar="int")
parser.add_option( "--trainingbatch",
                  type="int", dest="trainingbatch", default=10,
                  help="setup info", metavar="int")
parser.add_option( "--buildmodel",
                  action="store", dest="buildmodel", default=None,
                  help="setup info", metavar="Path")
(options, args) = parser.parse_args()


##########################
# preprocess database and store to disk
##########################
if (options.builddb):
  import csv
  import nibabel as nib  
  from scipy import ndimage

  from sklearn.model_selection import KFold
  X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
  y = np.array([1, 2, 3, 4])
  kf = KFold(n_splits=2)
  kf.get_n_splits(X)
  print(kf)  
  for train_index, test_index in kf.split(X):
     print("TRAIN:", train_index, "TEST:", test_index)
     X_train, X_test = X[train_index], X[test_index]
     y_train, y_test = y[train_index], y[test_index]

  # create  custom data frame database type
  globalexpectedpixel=512
  mydatabasetype = [('dataid', int), ('axialliverbounds',bool), ('axialtumorbounds',bool), ('imagepath','S128'),('imagedata','(%d,%d)float32' %(globalexpectedpixel,globalexpectedpixel)),('truthpath','S128'),('truthdata','(%d,%d)uint8' % (globalexpectedpixel,globalexpectedpixel))]

  # initialize empty dataframe
  numpydatabase = np.empty(0, dtype=mydatabasetype  )

  # load all data from csv
  dbfile="/rsrch1/ip/dtfuentes/SegmentationTrainingData/LiTS2017/LITS/trainingdata.csv"
  rootlocation="/rsrch1/ip/dtfuentes/SegmentationTrainingData/LiTS2017/LITS"
  totalnslice = 0 
  with open(dbfile, 'r') as csvfile:
    myreader = csv.DictReader(csvfile, delimiter=',')
    for row in myreader:
      imagelocation = '%s/%s' % (rootlocation,row['image'])
      truthlocation = '%s/%s' % (rootlocation,row['label'])
      print(imagelocation,truthlocation )

      # load nifti file
      imagedata = nib.load(imagelocation )
      numpyimage= imagedata.get_data().astype(IMG_DTYPE )
      # error check
      assert numpyimage.shape[0:2] == (globalexpectedpixel,globalexpectedpixel)
      nslice = numpyimage.shape[2]

      # load nifti file
      truthdata = nib.load(truthlocation )
      numpytruth= truthdata.get_data().astype(SEG_DTYPE)
      # error check
      assert numpytruth.shape[0:2] == (globalexpectedpixel,globalexpectedpixel)

      # bounding box for each label
      if( np.max(numpytruth) ==1 ) :
        liverboundingbox  = ndimage.find_objects(numpytruth)
        tumorboundingbox  = None
      else:
        (liverboundingbox,tumorboundingbox                      )  = ndimage.find_objects(numpytruth)

      # error check
      if( nslice  == numpytruth.shape[2]):
        # custom data type to subset  
        datamatrix = np.zeros(nslice  , dtype=mydatabasetype )
        
        # custom data type to subset  
        datamatrix ['dataid']          = np.repeat(row['dataid']    ,nslice  ) 
        #datamatrix ['xbounds']      = np.repeat(boundingbox[0],nslice  ) 
        #datamatrix ['ybounds']      = np.repeat(boundingbox[1],nslice  ) 
        #datamatrix ['zbounds']      = np.repeat(boundingbox[2],nslice  ) 
        #datamatrix ['nslice' ]      = np.repeat(nslice,nslice  ) 
        # id the slices within the bounding box
        axialliverbounds                              = np.repeat(False,nslice  ) 
        axialtumorbounds                             = np.repeat(False,nslice  ) 
        axialliverbounds[liverboundingbox[2]]         = True
        if (tumorboundingbox != None):
          axialtumorbounds[tumorboundingbox[2]]       = True
        datamatrix ['axialliverbounds'   ]            = axialliverbounds
        datamatrix ['axialtumorbounds'  ]             = axialtumorbounds
        datamatrix ['imagepath']                      = np.repeat(imagelocation ,nslice  ) 
        datamatrix ['truthpath']                      = np.repeat(truthlocation ,nslice  ) 
        datamatrix ['imagedata']                      = numpyimage.transpose(2,1,0) 
        datamatrix ['truthdata']                      = numpytruth.transpose(2,1,0)  
        numpydatabase = np.hstack((numpydatabase,datamatrix))
        # count total slice for QA
        totalnslice = totalnslice + nslice 
      else:
        print('training data error image[2] = %d , truth[2] = %d ' % (nslice,numpytruth.shape[2]))

  # save numpy array to disk
  np.save('trainingdata.npy', numpydatabase )

##########################
# build NN model
##########################
elif (options.anonymize != None):
  import json
  # load database
  numpydatabase = np.load(options.inputModelData)

  # get setup info
  with open(options.anonymize) as json_data:
      setupdata = json.load(json_data)
      mrn = setupdata[0]['mrn']  
      print(setupdata)

  # get subset
  trainingsubset =  numpydatabase[numpydatabase['mrn'] != mrn ]

  # ensure we get the same results each time we run the code
  np.random.seed(seed=0) 
  np.random.shuffle(trainingsubset )

  # subset within bounding box that has both viable and necrosis tissue
  trainingsubsetmasked    =  trainingsubset[trainingsubset['axialnecrosisbounds'] == True  ] 
  trainingsubsetmasked    =  trainingsubsetmasked[trainingsubsetmasked['axialviablebounds'] == True  ] 

  # add images outside bounding box to training data ? 
  trainingsubsetnotmasked =  trainingsubset[trainingsubset['axialliverbounds'] == False ]

  # copy pointers
  anonnumpydb =trainingsubsetmasked[['imagedata','truthdata']]
  x_train=anonnumpydb['imagedata']  
  y_train=trainingsubsetmasked['truthdata'] 

  # output location
  anonymizeoutputlocation =  '/'.join(options.anonymize.split('/')[:-1])
  np.save("%s/anonymizetrain.npy" % anonymizeoutputlocation,anonnumpydb )
  
  import nibabel as nib  
  print ( "writing training data for reference " ) 
  imgnii = nib.Nifti1Image(x_train[: ,:,:] , None )
  imgnii.to_filename( '%s/trainingimg.nii.gz' % anonymizeoutputlocation )
  segnii = nib.Nifti1Image(y_train[: ,:,:] , None )
  segnii.to_filename( '%s/trainingseg.nii.gz' % anonymizeoutputlocation )

##########################
# build NN model from anonymized data
##########################
elif (options.buildmodel!= None):
  from keras.layers import InputLayer, Conv2D, MaxPool2D, Flatten, Dense, UpSampling2D, LocallyConnected2D
  from keras.models import Model, Sequential

  # ## Training
  # * As the arrays we created before are 3-dimensional (no channel for grey images), we have to add one dimension to make it compatible with the ConvNet.
  # * Also, keep some slices for testing. 
  #     * Training converges with about 200 slices.
  #     * The initial results with 700 slices are terrible.
  #     * Explain!
  # * About 100 epochs lead to a pretty well-performing net. On an average CPU, one iteration takes about 10-15 sec. On GPU, this is much faster (increase the batch size also, to avoid unneccessary GPU memory transfers)
  # * With Batch Normalisation and PReLU, the number of parameters gets much larger, and training takes much longer. 
  #     * Does the result warrant the wait?
  #     * Explain!
  # * Callbacks enable better logging. 
  #     * We can add the TensorBoard logging mechanism. 
  #     * TensorBoard needs to be started externally, pointing to the log directory, which defaults to `./logs`.
  
  # In[16]:
  
  # load database
  anontrainingdata = np.load(options.buildmodel)
  totnslice = len(anontrainingdata)

  # load training data
  import skimage.transform
  x_train=skimage.transform.resize(anontrainingdata['imagedata'],(totnslice, options.trainingresample,options.trainingresample),order=0,preserve_range=True).astype(IMG_DTYPE)
  y_train=skimage.transform.resize(anontrainingdata['truthdata'],(totnslice, options.trainingresample,options.trainingresample),order=0,preserve_range=True).astype(SEG_DTYPE)

  # need parameter study on best way to split
  studydict = {'run_a':slice(0  ,300), 'run_b':slice(0  ,500), 'run_c':slice(0  ,700), 'run_d':slice(0  ,900), 'run_e':slice(0  ,1100),'run_f':slice(0  ,1300),'run_g':slice(0  ,1500) ,'run_h':slice(0  ,1700)  }

  TRAINING_SLICES      = studydict[options.trainingid]
  VALIDATION_SLICES    = slice(1700,1800)
  

  # DOC - Conv2D trainable parametes should be kernelsize_x * kernelsize_y * input_channels * output_channels
  # DOC -  2d convolution over each input channel is summed and provides one output channel.  each output channel has a independent set of kernel weights to train.
  # https://stackoverflow.com/questions/43306323/keras-conv2d-and-input-channels
  # https://github.com/petewarden/tensorflow_makefile/blob/master/tensorflow/g3doc/api_docs/python/functions_and_classes/tf.nn.conv2d.md
  # http://cs231n.github.io/convolutional-networks/
  # http://machinelearninguru.com/computer_vision/basics/convolution/convolution_layer.html
  # https://petewarden.com/2015/04/20/why-gemm-is-at-the-heart-of-deep-learning/

  #
  # We will use this to generate the regularisation block for the sequential model.
  from keras.layers import UpSampling2D
  from keras.layers import BatchNormalization,SpatialDropout2D
  from keras.layers.advanced_activations import LeakyReLU, PReLU
  def addConvBNSequential(model, filters=32, kernel_size=(3,3), batch_norm=True, activation='prelu', padding='same', kernel_regularizer=None,dropout=0.):
      if batch_norm:
          model = BatchNormalization()(model)
      if dropout>0.:
          model = SpatialDropout2D(dropout)(model)
      if activation == 'prelu':
          model = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding, activation='linear', kernel_regularizer=kernel_regularizer)(model)
          model = PReLU()(model)
      elif activation == 'lrelu':
          model = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding, activation='linear', kernel_regularizer=kernel_regularizer)(model)
          model = LeakyReLU()(model)
      else:
          model = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding, activation=activation, kernel_regularizer=kernel_regularizer)(model)
      return model
  
  
  # In[ ]:
  
  
  # Creates a small U-Net.
  from keras.layers import Input, concatenate
  def get_batchnorm_unet(_filters=32, _filters_add=0, _kernel_size=(3,3), _padding='same', _activation='prelu', _kernel_regularizer=None, _final_layer_nonlinearity='sigmoid', _batch_norm=True, _num_classes=1):
      # FIXME - HACK image size
      crop_size = options.trainingresample
      if _padding == 'valid':
          input_layer = Input(shape=(crop_size+40,crop_size+40,1))
      elif _padding == 'same':
          input_layer = Input(shape=(crop_size,crop_size,1))
  
      x0 = addConvBNSequential(input_layer, filters=_filters, kernel_size=_kernel_size, padding=_padding, activation=_activation, kernel_regularizer=_kernel_regularizer, batch_norm=_batch_norm)
      x0 = addConvBNSequential(x0,          filters=_filters, kernel_size=_kernel_size, padding=_padding, activation=_activation, kernel_regularizer=_kernel_regularizer, batch_norm=_batch_norm)
      x1 = MaxPool2D()(x0)
      
      x1 = addConvBNSequential(x1,          filters=_filters+_filters_add, kernel_size=_kernel_size, padding=_padding, activation=_activation, kernel_regularizer=_kernel_regularizer, batch_norm=_batch_norm)
      x1 = addConvBNSequential(x1,          filters=_filters+_filters_add, kernel_size=_kernel_size, padding=_padding, activation=_activation, kernel_regularizer=_kernel_regularizer, batch_norm=_batch_norm)
      x2 = MaxPool2D()(x1)
      
      x2 = addConvBNSequential(x2,          filters=_filters+2*_filters_add, kernel_size=_kernel_size, padding=_padding, activation=_activation, kernel_regularizer=_kernel_regularizer, batch_norm=_batch_norm)
      x2 = addConvBNSequential(x2,          filters=_filters+2*_filters_add, kernel_size=_kernel_size, padding=_padding, activation=_activation, kernel_regularizer=_kernel_regularizer, batch_norm=_batch_norm)
      x3 = UpSampling2D()(x2)
      
      x3 = concatenate([x1,x3])
      x3 = addConvBNSequential(x3,          filters=_filters+_filters_add, kernel_size=_kernel_size, padding=_padding, activation=_activation, kernel_regularizer=_kernel_regularizer, batch_norm=_batch_norm,dropout=.5)
      x3 = addConvBNSequential(x3,          filters=_filters+_filters_add, kernel_size=_kernel_size, padding=_padding, activation=_activation, kernel_regularizer=_kernel_regularizer, batch_norm=_batch_norm,dropout=.5)
      x4 = UpSampling2D()(x3)
      
      x4 = concatenate([x0,x4])
      x4 = addConvBNSequential(x4,          filters=_filters, kernel_size=_kernel_size, padding=_padding, activation=_activation, kernel_regularizer=_kernel_regularizer, batch_norm=_batch_norm,dropout=.5)
      x4 = addConvBNSequential(x4,          filters=_filters, kernel_size=_kernel_size, padding=_padding, activation=_activation, kernel_regularizer=_kernel_regularizer, batch_norm=_batch_norm,dropout=.5)
  
      # FIXME - need for arbitrary output
      output_layer = Conv2D(_num_classes, kernel_size=(1,1), activation=_final_layer_nonlinearity)(x4)
      
      model = Model(inputs=input_layer, outputs=output_layer)
      return model
  
  def get_bnormfull_unet(_filters=32, _filters_add=0, _kernel_size=(3,3), _padding='same', _activation='prelu', _kernel_regularizer=None, _final_layer_nonlinearity='sigmoid', _batch_norm=True, _num_classes=1):
      # FIXME - HACK image size
      crop_size = options.trainingresample
      if _padding == 'valid':
          input_layer = Input(shape=(crop_size+40,crop_size+40,1))
      elif _padding == 'same':
          input_layer = Input(shape=(crop_size,crop_size,1))
  
      x0 = addConvBNSequential(input_layer, filters=_filters, kernel_size=_kernel_size, padding=_padding, activation=_activation, kernel_regularizer=_kernel_regularizer, batch_norm=_batch_norm)
      x0 = addConvBNSequential(x0,          filters=_filters, kernel_size=_kernel_size, padding=_padding, activation=_activation, kernel_regularizer=_kernel_regularizer, batch_norm=_batch_norm)
      x1 = MaxPool2D()(x0)
      
      x1 = addConvBNSequential(x1,          filters=_filters+_filters_add, kernel_size=_kernel_size, padding=_padding, activation=_activation, kernel_regularizer=_kernel_regularizer, batch_norm=_batch_norm)
      x1 = addConvBNSequential(x1,          filters=_filters+_filters_add, kernel_size=_kernel_size, padding=_padding, activation=_activation, kernel_regularizer=_kernel_regularizer, batch_norm=_batch_norm)
      x2 = MaxPool2D()(x1)
      
      x2 = addConvBNSequential(x2,          filters=2*(_filters+_filters_add), kernel_size=_kernel_size, padding=_padding, activation=_activation, kernel_regularizer=_kernel_regularizer, batch_norm=_batch_norm)
      x2 = addConvBNSequential(x2,          filters=2*(_filters+_filters_add), kernel_size=_kernel_size, padding=_padding, activation=_activation, kernel_regularizer=_kernel_regularizer, batch_norm=_batch_norm)
      x3 = MaxPool2D()(x2)
      
      x3 = addConvBNSequential(x3,          filters=4*(_filters+_filters_add), kernel_size=_kernel_size, padding=_padding, activation=_activation, kernel_regularizer=_kernel_regularizer, batch_norm=_batch_norm)
      x3 = addConvBNSequential(x3,          filters=4*(_filters+_filters_add), kernel_size=_kernel_size, padding=_padding, activation=_activation, kernel_regularizer=_kernel_regularizer, batch_norm=_batch_norm)
      x4 = MaxPool2D()(x3)
      
      x4 = addConvBNSequential(x4,          filters=8*(_filters+_filters_add), kernel_size=_kernel_size, padding=_padding, activation=_activation, kernel_regularizer=_kernel_regularizer, batch_norm=_batch_norm)
      x4 = addConvBNSequential(x4,          filters=8*(_filters+_filters_add), kernel_size=_kernel_size, padding=_padding, activation=_activation, kernel_regularizer=_kernel_regularizer, batch_norm=_batch_norm)
      x5 = UpSampling2D()(x4)
      
      x5 = concatenate([x3,x5])
      x5 = addConvBNSequential(x5,          filters=4*(_filters+_filters_add), kernel_size=_kernel_size, padding=_padding, activation=_activation, kernel_regularizer=_kernel_regularizer, batch_norm=_batch_norm,dropout=.5)
      x5 = addConvBNSequential(x5,          filters=4*(_filters+_filters_add), kernel_size=_kernel_size, padding=_padding, activation=_activation, kernel_regularizer=_kernel_regularizer, batch_norm=_batch_norm,dropout=.5)
      x6 = UpSampling2D()(x5)
      
      x6 = concatenate([x2,x6])
      x6 = addConvBNSequential(x6,          filters=2*(_filters+_filters_add), kernel_size=_kernel_size, padding=_padding, activation=_activation, kernel_regularizer=_kernel_regularizer, batch_norm=_batch_norm,dropout=.5)
      x6 = addConvBNSequential(x6,          filters=2*(_filters+_filters_add), kernel_size=_kernel_size, padding=_padding, activation=_activation, kernel_regularizer=_kernel_regularizer, batch_norm=_batch_norm,dropout=.5)
      x7 = UpSampling2D()(x6)
      
      x7 = concatenate([x1,x7])
      x7 = addConvBNSequential(x7,          filters=_filters+_filters_add, kernel_size=_kernel_size, padding=_padding, activation=_activation, kernel_regularizer=_kernel_regularizer, batch_norm=_batch_norm,dropout=.5)
      x7 = addConvBNSequential(x7,          filters=_filters+_filters_add, kernel_size=_kernel_size, padding=_padding, activation=_activation, kernel_regularizer=_kernel_regularizer, batch_norm=_batch_norm,dropout=.5)
      x8 = UpSampling2D()(x7)
      
      x8 = concatenate([x0,x8])
      x8 = addConvBNSequential(x8,          filters=_filters, kernel_size=_kernel_size, padding=_padding, activation=_activation, kernel_regularizer=_kernel_regularizer, batch_norm=_batch_norm,dropout=.5)
      x8 = addConvBNSequential(x8,          filters=_filters, kernel_size=_kernel_size, padding=_padding, activation=_activation, kernel_regularizer=_kernel_regularizer, batch_norm=_batch_norm,dropout=.5)
  
      # FIXME - need for arbitrary output
      output_layer = Conv2D(_num_classes, kernel_size=(1,1), activation=_final_layer_nonlinearity)(x8)
      
      model = Model(inputs=input_layer, outputs=output_layer)
      return model
  
## ipdb> bt
##  /opt/apps/miniconda/miniconda3/lib/python3.6/bdb.py(434)run()
##    432         sys.settrace(self.trace_dispatch)
##    433         try:
##--> 434             exec(cmd, globals, locals)
##    435         except BdbQuit:
##    436             pass
##
##  <string>(1)<module>()
##
##  /opt/apps/miniconda/miniconda3/lib/python3.6/site-packages/IPython/core/interactiveshell.py(2527)safe_execfile()
##   2525                 py3compat.execfile(
##   2526                     fname, glob, loc,
##-> 2527                     self.compile if shell_futures else None)
##   2528             except SystemExit as status:
##   2529                 # If the call was made with 0 or None exit status (sys.exit(0)
##
##  /opt/apps/miniconda/miniconda3/lib/python3.6/site-packages/IPython/utils/py3compat.py(188)execfile()
##    186     with open(fname, 'rb') as f:
##    187         compiler = compiler or compile
##--> 188         exec(compiler(f.read(), fname, 'exec'), glob, loc)
##    189
##    190 # Refactor print statements in doctests.
##
##  /rsrch1/ip/dtfuentes/github/RandomForestHCCResponse/Code/loadtraining.py(357)<module>()
##    355   #             def weighted(y_true, y_pred, weights, mask=None):
##    356   #model.compile(loss='categorical_crossentropy',optimizer='adadelta')
##--> 357   model.compile(loss=dice_coef_loss,optimizer=options.trainingsolver)
##    358   print("Model parameters: {0:,}".format(model.count_params()))
##    359   # FIXME - better to use more epochs on a single one-hot model? or break up into multiple models steps?
##
##  /opt/apps/miniconda/miniconda3/lib/python3.6/site-packages/keras/engine/training.py(830)compile()
##    828                 with K.name_scope(self.output_names[i] + '_loss'):
##    829                     output_loss = weighted_loss(y_true, y_pred,
##--> 830                                                 sample_weight, mask)
##    831                 if len(self.outputs) > 1:
##    832                     self.metrics_tensors.append(output_loss)
##
##  /opt/apps/miniconda/miniconda3/lib/python3.6/site-packages/keras/engine/training.py(429)weighted()
##    427         """
##    428         # score_array has ndim >= 2
##--> 429         score_array = fn(y_true, y_pred)
##    430         if mask is not None:
##    431             # Cast the mask to floatX to avoid float64 upcasting in Theano
##
##  /rsrch1/ip/dtfuentes/github/RandomForestHCCResponse/Code/loadtraining.py(315)dice_coef_loss()
##    313       # FIXME HACK need for arbitrary length
##    314       #lossweight= [.1,.1,1.,1.]
##--> 315       return 1-dice_coef(y_true, y_pred)
##    316       #lossweight= [1.,1.,1.,1.]
##    317       #totalloss = lossweight[0]*( 1-dice_coef(y_true[:,:,:,0], y_pred[:,:,:,0])) +lossweight[1]*( 1-dice_coef(y_true[:,:,:,1], y_pred[:,:,:,1])) +lossweight[2]*( 1-dice_coef(y_true[:,:,:,2], y_pred[:,:,:,2])) +lossweight[3]*( 1-dice_coef(y_true[:,:,:,3], y_pred[:,:,:,3]))
##
##> /rsrch1/ip/dtfuentes/github/RandomForestHCCResponse/Code/loadtraining.py(310)dice_coef()
##    308       """
##    309       intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
##2-> 310       return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)
##    311
##    312   def dice_coef_loss(y_true, y_pred)
  
  
  ### Train model with Dice loss
  import keras.backend as K
  def dice_coef(y_true, y_pred, smooth=1):
      """
      Dice = (2*|X & Y|)/ (|X|+ |Y|)
           =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
      ref: https://arxiv.org/pdf/1606.04797v1.pdf
      @url: https://gist.github.com/wassname/7793e2058c5c9dacb5212c0ac0b18a8a
      @author: wassname
      """
      intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
      # BUG - implicit reduce mean -   /opt/apps/miniconda/miniconda3/lib/python3.6/site-packages/keras/engine/training.py(447)weighted()
      #return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)
      # BUG - implicit reduce mean  will normalize to batch not all pixel
      npixel  =  K.cast(K.prod(K.shape(y_true)[1:]),np.float32)
      # BUG - note thise sum is implicitly over the batch.... thus the DSC is the average across the batch
      return npixel *(2. * intersection + smooth) / (K.sum(K.square(y_true),axis=None) + K.sum(K.square(y_pred),axis=None) + smooth)
  
  def dice_coef_loss(y_true, y_pred):
      # FIXME HACK need for arbitrary length
      #lossweight= [.1,.1,1.,1.]
      #return 1-dice_coef(y_true, y_pred)
      return -dice_coef(y_true, y_pred)
      #lossweight= [1.,1.,1.,1.]
      #totalloss = lossweight[0]*( 1-dice_coef(y_true[:,:,:,0], y_pred[:,:,:,0])) +lossweight[1]*( 1-dice_coef(y_true[:,:,:,1], y_pred[:,:,:,1])) +lossweight[2]*( 1-dice_coef(y_true[:,:,:,2], y_pred[:,:,:,2])) +lossweight[3]*( 1-dice_coef(y_true[:,:,:,3], y_pred[:,:,:,3]))
      #return totalloss 
  
  #  NOTE - intuition for array sum
  #  xxx = np.array([[[[0., 8., 3., 0.], [2., 6., 4., 3.]], [[2., 8., 4., 2.], [8., 1., 3., 7.]]], [[[4., 2., 1., 3.], [3., 8., 2., 5.]], [[6., 0., 1., 6.], [1., 6., 8., 2.]]], [[[4., 6., 3., 7.], [4., 1., 0., 2.]], [[3., 3., 8., 4.], [9., 1., 2., 1.]]]])
  #  kxx = K.variable(value=xxx )
  #  kxx.shape = K.variable(value=xxx )
  #  xxx.shape = (3, 2, 2, 4)
  #  intersection =  np.sum(xxx , axis = -1)
  #  intersection.shape = (3, 2, 2)
  #  imagesum     =   np.sum(xxx , axis = (1,2)) #  imagesum.shape = (3, 4)
  #  kimagesum    =    K.sum(kxx , axis = (1,2)) 
  #  zzz = xxx / imagesum[:,np.newaxis,np.newaxis,:] 
  #  kzz = kxx / K.expand_dims(K.expand_dims(kimagesum,axis=1),axis=2)
  def dice_imageloss(y_true, y_pred, smooth=0):
      """
      Dice = \sum_Nbatch \sum_Nonehot (2*|X & Y|)/ (|X|+ |Y|)
           = \sum_Nbatch \sum_Nonehot  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
      return negative dice value for minimization. one dsc per one hot image for each batch. Nbatch * Nonehot total images. 
      objective function has implicit reduce mean -  /opt/apps/miniconda/miniconda3/lib/python3.6/site-packages/keras/engine/training.py(447)weighted()
      """
      # DSC = DSC_image1 +  DSC_image2 + DSC_image3 + ...
      intersection = 2. *K.abs(y_true * y_pred) + smooth
      # FIXME - hard code sum over 2d image
      sumunion = K.sum(K.square(y_true),axis=(1,2)) + K.sum(K.square(y_pred),axis=(1,2)) + smooth
      dicevalues= K.sum(intersection / K.expand_dims(K.expand_dims(sumunion,axis=1),axis=2), axis=(1,2))
      return -dicevalues
  
  def dice_metric_zero(y_true, y_pred):
      batchdiceloss =  dice_imageloss(y_true, y_pred)
      return -batchdiceloss[:,0]
  def dice_metric_one(y_true, y_pred):
      batchdiceloss =  dice_imageloss(y_true, y_pred)
      return -batchdiceloss[:,1]
  def dice_metric_two(y_true, y_pred):
      batchdiceloss =  dice_imageloss(y_true, y_pred)
      return -batchdiceloss[:,2]
  def dice_metric_three(y_true, y_pred):
      batchdiceloss =  dice_imageloss(y_true, y_pred)
      return -batchdiceloss[:,3]

  # Convert the labels into a one-hot representation
  from keras.utils.np_utils import to_categorical
  
  # Convert to uint8 data and find out how many labels.
  t=y_train.astype(np.uint8)
  t_max=np.max(t)
  print("Range of values: [0, {}]".format(t_max))
  y_train_one_hot = to_categorical(t, num_classes=t_max+1).reshape((y_train.shape)+(t_max+1,))
  print("Shape before: {}; Shape after: {}".format(y_train.shape, y_train_one_hot.shape))
  # The liver neuron should also be active for lesions within the liver
  liver = np.max(y_train_one_hot[:,:,:,1:], axis=3)
  y_train_one_hot[:,:,:,1]=liver
  

  # output location
  outputlocation =  '/'.join(options.buildmodel.split('/')[:-1])
  logfileoutputdir= '%s/tblog/%s/%s/%s/%d/%s/%03d' % (outputlocation,options.trainingloss,options.trainingmodel,options.trainingsolver,options.trainingresample,options.trainingid,options.trainingbatch)
  # ensure directory exists
  import os
  os.system ('mkdir -p %s' % logfileoutputdir)

  
  from keras.callbacks import TensorBoard
  tensorboard = TensorBoard(log_dir=logfileoutputdir, histogram_freq=0, write_graph=True, write_images=False)
  
  # dictionary of models to evaluate
  modeldict = {'half': get_batchnorm_unet(_activation='relu', _batch_norm=True,_filters=64, _filters_add=64,_num_classes=4),'full': get_bnormfull_unet(_activation='relu', _batch_norm=True,_filters=64, _filters_add=64,_num_classes=4)}
  model = modeldict[options.trainingmodel] 

  lossdict = {'dscvec': dice_coef_loss,'dscimg': dice_imageloss}
  # FIXME - dice applied to each class separately, and weight each class
  # 
  # ojective function is summed
  #f    weighted          /opt/apps/miniconda/miniconda3/lib/python3.6/site-packages/keras/engine/training.py
  #             function:_weighted_masked_objective
  #             def weighted(y_true, y_pred, weights, mask=None):
  #model.compile(loss='categorical_crossentropy',optimizer='adadelta')
  model.compile(loss=lossdict[options.trainingloss],metrics=[dice_metric_zero,dice_metric_one,dice_metric_two,dice_metric_three],optimizer=options.trainingsolver)
  print("Model parameters: {0:,}".format(model.count_params()))
  # FIXME - better to use more epochs on a single one-hot model? or break up into multiple models steps?
  # FIXME -  IE liver mask first then resize to the liver for viable/necrosis ? 
  history = model.fit(x_train[TRAINING_SLICES ,:,:,np.newaxis],
                      y_train_one_hot[TRAINING_SLICES ],
                      validation_data=(x_train[VALIDATION_SLICES,:,:,np.newaxis],y_train_one_hot[VALIDATION_SLICES]),
                      callbacks = [tensorboard],
                      batch_size=options.trainingbatch, epochs=1000)
                      #batch_size=10, epochs=300
  
  
  # ### Assignment: Extend the plot function to handle multiple classes.
  # Then, activate the visualization callback in the training again. Try to find a slice with more than one output class to see the success.
  

  # output predictions
  if (options.trainingid == 'run_a'):
    import nibabel as nib  
    validationimgnii = nib.Nifti1Image(x_train[VALIDATION_SLICES,:,:] , None )
    validationimgnii.to_filename( '%s/validationimg.nii.gz' % logfileoutputdir )
    validationonehotnii = nib.Nifti1Image(y_train[VALIDATION_SLICES  ,:,:] , None )
    validationonehotnii.to_filename( '%s/validationseg.nii.gz' % logfileoutputdir )
    y_predicted = model.predict(x_train[VALIDATION_SLICES,:,:,np.newaxis])
    y_segmentation = np.argmax(y_predicted , axis=-1)
    validationprediction = nib.Nifti1Image(y_predicted [:,:,:] , None )
    validationprediction.to_filename( '%s/validationpredict.nii.gz' % logfileoutputdir )
    validationoutput     = nib.Nifti1Image( y_segmentation.astype(np.uint8), None )
    validationoutput.to_filename( '%s/validationoutput.nii.gz' % logfileoutputdir )
  
  # https://machinelearningmastery.com/save-load-keras-deep-learning-models/
  # serialize model to JSON
  model_json = model.to_json()
  with open("%s/tumormodelunet.json" % logfileoutputdir , "w") as json_file:
      json_file.write(model_json)
  # serialize weights to HDF5
  model.save_weights("%s/tumormodelunet.h5" % logfileoutputdir )
  print("Saved model to disk")
##########################
# debugging
##########################
elif (options.predictmodel != None and options.predictimage != None and options.segmentation != None ):
  import json
  import nibabel as nib  
  import skimage.transform
  # force cpu for debug
  import os
  os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
  os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
  from keras.models import model_from_json
  # load json and create model
  json_file = open(options.predictmodel, 'r')
  loaded_model_json = json_file.read()
  json_file.close()
  loaded_model = model_from_json(loaded_model_json)
  # load weights into new model
  weightsfile= '.'.join(options.predictmodel.split('.')[0:-1]) + '.h5'
  loaded_model.load_weights(weightsfile)
  print("Loaded model from disk")
   
  imagepredict = nib.load(options.predictimage)
  numpypredict= imagepredict.get_data().astype(IMG_DTYPE )
  # error check
  assert numpypredict.shape[0:2] == (globalexpectedpixel,globalexpectedpixel)
  nslice = numpypredict.shape[2]
  resizepredict = skimage.transform.resize(numpypredict,(options.trainingresample,options.trainingresample,nslice ),order=0,preserve_range=True).astype(IMG_DTYPE).transpose(2,0,1)

  # predict slice by slice
  numlabel = 4
  segmentation  = np.zeros( (options.trainingresample,options.trainingresample,numlabel ,nslice )  , dtype=IMG_DTYPE )
  segmentexpect = np.zeros( (globalexpectedpixel,globalexpectedpixel,nslice, numlabel)  , dtype=IMG_DTYPE )
  for iii in range(nslice):
    print ( "%d  " % iii ,end='',flush=True) 
    # NN expect liver in top left
    # evaluate loaded model on test data
    segmentation[...,iii]  = loaded_model.predict(resizepredict[iii:iii+1,:,:,np.newaxis] )
    for jjj in range(numlabel):
      segmentexpect[:,:,iii,jjj] = skimage.transform.resize(segmentation[:,:,jjj,iii],(globalexpectedpixel,globalexpectedpixel),order=0,preserve_range=True).astype(IMG_DTYPE)

  # save segmentation at original resolution
  print ( "writing %s  " % options.segmentation) 
  for jjj in range(numlabel):
    imgnii = nib.Nifti1Image(segmentexpect[:,:,:,jjj] , imagepredict.affine )
    imgnii.to_filename( options.segmentation.replace('.nii.gz','%d.nii.gz' % jjj) )

##########################
# debugging
##########################
elif (options.debug ):
  ORIGINAL_PATH = '../DeepLearningTutorialFeb2018/input/singleSlicePNGs/originals/'
  MASK_PATH = '../DeepLearningTutorialFeb2018/input/singleSlicePNGs/masks/'
  from os import listdir
  from os.path import join, abspath
  import matplotlib.pyplot as plt # not only for plotting, but also for imread()
  
  original_files = [join(ORIGINAL_PATH,f) for f in listdir(ORIGINAL_PATH) if f.endswith('.png')]
  mask_files     = [join(MASK_PATH,f)     for f in listdir(MASK_PATH)     if f.endswith('.png')]
  # File reading is not deterministic, i.e. not in any particular order. This means, the two arrays may be out of sync. Sorting fixes this.
  original_files = sorted(original_files)
  mask_files     = sorted(mask_files)
  assert all([orig.replace('/originals', '/masks') in mask_files for orig in original_files])
  assert len(original_files) == len(mask_files)
  np.random.seed(seed=0) # ensure we get the same results each time we run the code
  permuted_file_indexer = np.random.permutation(len(original_files))
  
  
  # In[6]:
  
  
  # inspect some filename pairs (negative indices count from the back, -1 is the last item)
  for a in [0,300,1000,1500,2000,-1]:
      print(original_files[a], mask_files[a])
  
  
  # In[7]:
  
  
  # the images have sizes between 79x79 and 125x125 pixels;
  # set up cropping for the remainder of the code
  cropping = slice(0,76) # slice off the first 76 voxels in order to ensure a common size
  #cropping = slice(None) # do not crop
  
  # loading into a single array and apply random permutation:
  x_train = np.array([plt.imread(original_files[index])[cropping,cropping] for index in permuted_file_indexer])
  y_train = np.array([plt.imread(mask_files[index])[cropping,cropping]*255 for index in permuted_file_indexer])
  
  # Figure out the smallest image -- this is only needed if not training on patches, but full slices. 
  # 
  print("min size: %d -- max size: %d" % (np.min([np.shape(slice) for slice in x_train]), np.max([np.shape(slice) for slice in x_train])))
##########################
# print help
##########################
else:
  parser.print_help()
