import numpy as np

# raw dicom data is usually short int (2bytes) datatype
# labels are usually uchar (1byte)
IMG_DTYPE = np.int16
SEG_DTYPE = np.uint8

# setup command line parser to control execution
from optparse import OptionParser
parser = OptionParser()
parser.add_option( "--builddb",
                  action="store_true", dest="builddb", default=False,
                  help="load all training data into npy", metavar="FILE")
parser.add_option( "--trainmodel",
                  action="store_true", dest="trainmodel", default=False,
                  help="train model", metavar="FILE")
parser.add_option( "--setuptestset",
                  action="store_true", dest="setuptestset", default=False,
                  help="cross validate test set", metavar="FILE")
parser.add_option( "--predictmodel",
                  action="store", dest="predictmodel", default=None,
                  help="apply model to image", metavar="Path")
parser.add_option( "--predictimage",
                  action="store", dest="predictimage", default=None,
                  help="apply model to image", metavar="Path")
parser.add_option( "--segmentation",
                  action="store", dest="segmentation", default=None,
                  help="model output ", metavar="Path")
parser.add_option( "--trainingsolver",
                  action="store", dest="trainingsolver", default='adadelta',
                  help="setup info", metavar="string")
parser.add_option( "--dbfile",
                  action="store", dest="dbfile", default="./trainingdata.csv",
                  help="training data file", metavar="string")
parser.add_option( "--trainingresample",
                  type="int", dest="trainingresample", default=256,
                  help="setup info", metavar="int")
parser.add_option( "--trainingbatch",
                  type="int", dest="trainingbatch", default=4,
                  help="setup info", metavar="int")
parser.add_option( "--kfolds",
                  type="int", dest="kfolds", default=5,
                  help="setup info", metavar="int")
parser.add_option( "--idfold",
                  type="int", dest="idfold", default=0,
                  help="setup info", metavar="int")
parser.add_option( "--rootlocation",
                  action="store", dest="rootlocation", default='/rsrch1/ip/jacctor/LiTS/LiTS',
                  help="setup info", metavar="string")
parser.add_option("--numepochs",
                  type="int", dest="numepochs", default=10,
                  help="number of epochs for training", metavar="int")
parser.add_option("--outdir",
                  action="store", dest="outdir", default='./',
                  help="directory for output", metavar="string")
(options, args) = parser.parse_args()


# FIXME:  @jonasactor - is there a better software/programming practice to keep track  of the global variables?
_globalnpfile = options.dbfile.replace('.csv','%d.npy' % options.trainingresample )
_globalexpectedpixel=512
print('database file: %s ' % _globalnpfile )


# build data base from CSV file
def GetDataDictionary():
  import csv
  CSVDictionary = {}
  with open(options.dbfile, 'r') as csvfile:
    myreader = csv.DictReader(csvfile, delimiter=',')
    for row in myreader:
       CSVDictionary[int( row['dataid'])]  =  {'image':row['image'], 'label':row['label']}
  return CSVDictionary


# setup kfolds
def GetSetupKfolds(numfolds,idfold):
  import csv
  from sklearn.model_selection import KFold
  # get id from setupfiles
  dataidsfull = []
  with open(options.dbfile, 'r') as csvfile:
    myreader = csv.DictReader(csvfile, delimiter=',')
    for row in myreader:
       dataidsfull.append( int( row['dataid']))
  if (numfolds < idfold or numfolds < 1):
     raise("data input error")
  # split in folds
  if (numfolds > 1):
     kf = KFold(n_splits=numfolds)
     allkfolds = [ (train_index, test_index) for train_index, test_index in kf.split(dataidsfull )]
     train_index = allkfolds[idfold][0]
     test_index  = allkfolds[idfold][1]
  else:
     train_index = np.array(dataidsfull )
     test_index  = None
  print(numfolds, idfold)
  print("train_index:\t", train_index)
  print("test_index:\t", test_index)
  return (train_index,test_index)

##########################
# preprocess database and store to disk
##########################
if (options.builddb):
  import csv
  import nibabel as nib
  from scipy import ndimage
  import skimage.transform

  # create  custom data frame database type
  mydatabasetype = [('dataid', int),
     ('axialliverbounds',bool),
     ('axialtumorbounds',bool),
     ('imagepath','S128'),
     ('imagedata','(%d,%d)int16' %(options.trainingresample,options.trainingresample)),
     ('truthpath','S128'),('truthdata','(%d,%d)uint8' % (options.trainingresample,options.trainingresample))]

  # initialize empty dataframe
  numpydatabase = np.empty(0, dtype=mydatabasetype  )

  # load all data from csv
  totalnslice = 0
  with open(options.dbfile, 'r') as csvfile:
    myreader = csv.DictReader(csvfile, delimiter=',')
    for row in myreader:
      imagelocation = '%s/%s' % (options.rootlocation,row['image'])
      truthlocation = '%s/%s' % (options.rootlocation,row['label'])
      print(imagelocation,truthlocation )

      # load nifti file
      imagedata = nib.load(imagelocation )
      numpyimage= imagedata.get_data().astype(IMG_DTYPE )
      # error check
      assert numpyimage.shape[0:2] == (_globalexpectedpixel,_globalexpectedpixel)
      nslice = numpyimage.shape[2]
      resimage=skimage.transform.resize(numpyimage,
            (options.trainingresample,options.trainingresample,nslice),
            order=0,
            mode='constant',
            preserve_range=True).astype(IMG_DTYPE)

      # load nifti file
      truthdata = nib.load(truthlocation )
      numpytruth= truthdata.get_data().astype(SEG_DTYPE)
      # error check
      assert numpytruth.shape[0:2] == (_globalexpectedpixel,_globalexpectedpixel)
      assert nslice  == numpytruth.shape[2]
      restruth=skimage.transform.resize(numpytruth,(options.trainingresample,options.trainingresample,nslice),order=0,mode='constant',preserve_range=True).astype(SEG_DTYPE)

      # bounding box for each label
      if( np.max(restruth) ==1 ) :
        (liverboundingbox,)  = ndimage.find_objects(restruth)
        tumorboundingbox  = None
      else:
        (liverboundingbox,tumorboundingbox                      )  = ndimage.find_objects(restruth)

      # error check
      if( nslice  == restruth.shape[2]):
        # custom data type to subset
        datamatrix = np.zeros(nslice  , dtype=mydatabasetype )

        # custom data type to subset
        datamatrix ['dataid']          = np.repeat(row['dataid']    ,nslice  )
        # id the slices within the bounding box
        axialliverbounds                              = np.repeat(False,nslice  )
        axialtumorbounds                              = np.repeat(False,nslice  )
        axialliverbounds[liverboundingbox[2]]         = True
        if (tumorboundingbox != None):
          axialtumorbounds[tumorboundingbox[2]]       = True
        datamatrix ['axialliverbounds'   ]            = axialliverbounds
        datamatrix ['axialtumorbounds'  ]             = axialtumorbounds
        datamatrix ['imagepath']                      = np.repeat(imagelocation ,nslice  )
        datamatrix ['truthpath']                      = np.repeat(truthlocation ,nslice  )
        datamatrix ['imagedata']                      = resimage.transpose(2,1,0)
        datamatrix ['truthdata']                      = restruth.transpose(2,1,0)
        numpydatabase = np.hstack((numpydatabase,datamatrix))
        # count total slice for QA
        totalnslice = totalnslice + nslice
      else:
        print('training data error image[2] = %d , truth[2] = %d ' % (nslice,restruth.shape[2]))

  # save numpy array to disk
  np.save( _globalnpfile,numpydatabase )


##########################
# build NN model from anonymized data
##########################
elif (options.trainmodel ):


  ###
  ### set up data
  ###

  # load database
  print('loading memory map db for large dataset')
  numpydatabase = np.load(_globalnpfile)

  #setup kfolds
  (train_index,test_index) = GetSetupKfolds(options.kfolds,options.idfold)

  print('copy data subsets into memory...')
  axialbounds = numpydatabase['axialliverbounds']
  dataidarray = numpydatabase['dataid']
  dbtrainindex= np.isin(dataidarray, train_index )
  dbtestindex = np.isin(dataidarray, test_index  )
  subsetidx_train  = np.all( np.vstack((axialbounds , dbtrainindex)) , axis=0 )
  subsetidx_test   = np.all( np.vstack((axialbounds , dbtestindex )) , axis=0 )

  if  np.sum(subsetidx_train   ) + np.sum(subsetidx_test) != min(np.sum(axialbounds ),np.sum(dbtrainindex )) :
    raise("data error")

  print('copy memory map from disk to RAM...')
  trainingsubset = numpydatabase[subsetidx_train   ]

  np.random.seed(seed=0)
  np.random.shuffle(trainingsubset )

  # subset within bounding box that has liver
  totnslice = len(trainingsubset)
  print("nslice train ",totnslice )

  # load training data as views
  x_train=trainingsubset['imagedata']
  y_train=trainingsubset['truthdata']
  slicesplit =  int(0.9 * totnslice )
  TRAINING_SLICES      = slice(0,slicesplit)
  VALIDATION_SLICES    = slice(slicesplit,totnslice)

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



  ###
  ### set up NN
  ###

  from keras.layers import InputLayer, Conv2D, Lambda, Add
  from keras.models import Model, Sequential
  import keras.backend as K

  # define np sparse matrix npXP TODO
  # define np sparse matrix npXN TODO
  # define np sparse matrix npYP TODO
  # define np sparse matrix npYN TODO

  # cast matrices as tensors
  XP = K.constant(npXP)
  XN = K.constant(npXN)
  YP = K.constant(npYP)
  YN = K.constant(npYN)

  # write layer for sparse stencil operation
  def multXP_relu(x):
      Ax  = K.dot(x,XP)
      return K.relu(Ax)
  def multXN_relu(x):
      Ax  = K.dot(-x,XN)
      return -K.relu(Ax)
  def multYP_relu(x):
      Ax  = K.dot(x,YP)
      return K.relu(Ax)
  def multYN_relu(x):
      Ax  = K.dot(-x,YN)
      return -K.relu(Ax)

  def timestep_layer(in_layer):
      #     stencil multiplications, then ReLu i.e. upwind max
      xp = Lambda(multXP_relu)(in_layer)
      xn = Lambda(multXN_relu)(in_layer)
      yp = Lambda(multYP_relu)(in_layer)
      yn = Lambda(multYN_relu)(in_layer)
      #     sum ReLu outputs for each dimension
      x = Add()([xp,xn])
      y = Add()([yp,yn])
      #     1x1 conv
      #     factors in dt, dx values directly into convolution
      fx = Conv2D(1, (1,1), padding='same', use_bias=False)(x)
      fy = Conv2D(1, (1,1), padding='same', use_bias=False)(y)
      #     add to prev
      return Add()([in_layer,fx,fy])

  def get_upwind_transport_net(_nt, _final_sigma='sigmoid', _num_classes=1):
      in_layer = Input(shape=(crop_size,crop_size,1))
      for ttt in range(_nt):
          in_layer = timestep_layer(in_layer)
      out_layer = Conv2D(_num_classes, (1,1), activation=_final_sigma)(in_layer)
      model = Model(inputs=in_layer, outputs=out_layer)
      return model



  ###
  ### set up Dice scores
  ###

  ### Train model with Dice loss
  def dice_coef(y_true, y_pred, smooth=1):
      """
      Dice = (2*|X & Y|)/ (|X|+ |Y|)
           =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
      ref: https://arxiv.org/pdf/1606.04797v1.pdf
      @url: https://gist.github.com/wassname/7793e2058c5c9dacb5212c0ac0b18a8a
      @author: wassname
      """
      intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
      npixel  =  K.cast(K.prod(K.shape(y_true)[1:]),np.float32)
      return npixel *(2. * intersection + smooth) / (K.sum(K.square(y_true),axis=None) + K.sum(K.square(y_pred),axis=None) + smooth)

  def dice_coef_loss(y_true, y_pred):
      return -dice_coef(y_true, y_pred)

  def dice_imageloss(y_true, y_pred, smooth=0):
      """
      Dice = \sum_Nbatch \sum_Nonehot (2*|X & Y|)/ (|X|+ |Y|)
           = \sum_Nbatch \sum_Nonehot  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
      return negative dice value for minimization. one dsc per one hot image for each batch. Nbatch * Nonehot total images.
      objective function has implicit reduce mean -  /opt/apps/miniconda/miniconda3/lib/python3.6/site-packages/keras/engine/training.py(447)weighted()
      """
      # DSC = DSC_image1 +  DSC_image2 + DSC_image3 + ...
      intersection = 2. *K.abs(y_true * y_pred) + smooth
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



  ###
  ### set up output and logging
  ###

  # output location
  logfileoutputdir= '%s/%03d/%03d' % (options.outdir, options.kfolds, options.idfold)
  print(logfileoutputdir)
  import os
  os.system ('mkdir -p %s' % logfileoutputdir)

  from keras.callbacks import TensorBoard
  from keras.callbacks import Callback as CallbackBase

  tensorboard = TensorBoard(log_dir=logfileoutputdir, histogram_freq=0, write_graph=True, write_images=False)

  class MyHistories(CallbackBase):
      def on_train_begin(self, logs={}):
          self.min_valloss = np.inf

      def on_train_end(self, logs={}):
          return

      def on_epoch_begin(self, epoch, logs={}):
          return

      def on_epoch_end(self, epoch, logs={}):
          if logs.get('val_loss')< self.min_valloss :
             self.min_valloss = logs.get('val_loss')
             # https://machinelearningmastery.com/save-load-keras-deep-learning-models/
             # serialize model to JSON
             model_json = model.to_json()
             with open("%s/tumormodelunet.json" % logfileoutputdir , "w") as json_file:
                 json_file.write(model_json)
             # serialize weights to HDF5
             model.save_weights("%s/tumormodelunet.h5" % logfileoutputdir )
             print("Saved model to disk - val_loss", self.min_valloss  )
          return

      def on_batch_begin(self, batch, logs={}):
          return

      def on_batch_end(self, batch, logs={}):
          return
  callbacksave = MyHistories()



  ###
  ### create and run model
  ###

  model = get_upwind_transport_net(_nt, _final_sigma='sigmoid', _num_classes=t_max+1)
  model.compile(loss=dice_imageloss,metrics=[dice_metric_zero,dice_metric_one,dice_metric_two],optimizer=options.trainingsolver)

  print("Model parameters: {0:,}".format(model.count_params()))
  print("Input shape: ", x_train[TRAINING_SLICES,:,:,np.newaxis].shape)
  history = model.fit(x_train[TRAINING_SLICES ,:,:,np.newaxis],
                          y_train_one_hot[TRAINING_SLICES ],
                          validation_data=(x_train[VALIDATION_SLICES,:,:,np.newaxis],y_train_one_hot[VALIDATION_SLICES]),
                          callbacks = [tensorboard,callbacksave],
                          batch_size=options.trainingbatch,
                          epochs=options.numepochs)



  ###
  ### make predicions on validation set
  ###

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



##########################
# apply model to test set
##########################
elif (options.setuptestset):
  databaseinfo = GetDataDictionary()

  maketargetlist = []
  # open makefile
  with open('kfold%03d-predict.makefile' % options.kfolds ,'w') as fileHandle:
    for iii in range(options.kfolds):
      (train_set,test_set) = GetSetupKfolds(options.kfolds,iii)
      for idtest in test_set:
         uidoutputdir= '%s/%03d/%03d' % (options.outdir, options.kfolds, iii)
         segmaketarget  = '%s/label-%04d.nii.gz' % (uidoutputdir,idtest)
         maketargetlist.append(segmaketarget )
         imageprereq = '$(TRAININGROOT)/%s' % databaseinfo[idtest]['image']
         cvtestcmd = "python3 ./liver2.py --predictimage=%s --predictmodel=%s/tumormodelunet.json --segmentation=%s --dbfile=%s"  % (imageprereq ,uidoutputdir,segmaketarget ,options.dbfile)
         fileHandle.write('%s: %s\n' % (segmaketarget ,imageprereq ) )
         fileHandle.write('\t%s\n' % cvtestcmd)
  # build job list
  with open('kfold%03d-predict.makefile' % options.kfolds, 'r') as original: datastream = original.read()
  with open('kfold%03d-predict.makefile' % options.kfolds, 'w') as modified: modified.write( 'TRAININGROOT=%s\n' % options.rootlocation + "cvtest: %s \n" % ' '.join(maketargetlist) + datastream)

  with open('kfold%03d-stats.makefile' % options.kfolds, 'w') as fileHandle:
    for iii in range(options.kfolds):
      (train_set,test_set) = GetSetupKfolds(options.kfolds,iii)
      for idtest in test_set:
         uidoutputdir= '%s/%03d/%03d' % (options.outdir, options.kfolds, iii)
         segmaketarget  = '%s/label-%04d.nii.gz' % (uidoutputdir,idtest)
         segmaketarget0 = '%s/label-%04d-0.nii.gz' % (uidoutputdir,idtest)
         segmaketargetQ = '%s/label-%04d-?.nii.gz' % (uidoutputdir,idtest)
         predicttarget  = '%s/label-%04d-all.nii.gz' % (uidoutputdir,idtest)
         statstarget    = '%s/stats-%04d.txt' % (uidoutputdir,idtest)
         maketargetlist.append(segmaketarget )
         imageprereq = '$(TRAININGROOT)/%s' % databaseinfo[idtest]['image']
         segprereq   = '$(TRAININGROOT)/%s' % databaseinfo[idtest]['label']
         votecmd = "c3d %s -vote -type uchar -o %s" % (segmaketargetQ, predicttarget)
         infocmd = "c3d %s -info > %s" % (segmaketarget0,statstarget)
         statcmd = "c3d -verbose %s %s -overlap 0 -overlap 1 -overlap 2 > %s" % (predicttarget, segprereq, statstarget)
         fileHandle.write('%s: %s\n' % (segmaketarget ,imageprereq ) )
         fileHandle.write('\t%s\n' % votecmd)
         fileHandle.write('\t%s\n' % infocmd)
         fileHandle.write('\t%s\n' % statcmd)
  # build job list
  with open('kfold%03d-stats.makefile' % options.kfolds, 'r') as original: datastream = original.read()
  with open('kfold%03d-stats.makefile' % options.kfolds, 'w') as modified: modified.write( 'TRAININGROOT=%s\n' % options.rootlocation + "cvtest: %s \n" % ' '.join(maketargetlist) + datastream)

##########################
# apply model to new data
##########################
elif (options.predictmodel != None and options.predictimage != None and options.segmentation != None ):
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
  print(nslice)
  resizepredict = skimage.transform.resize(numpypredict,(options.trainingresample,options.trainingresample,nslice ),order=0,preserve_range=True,mode='constant').astype(IMG_DTYPE).transpose(2,1,0)

  # FIXME: @jonasactor - the numlabel will change depending on the training data... can you make this more robust and the number of labels from the model?
  numlabel = 3

  segout = loaded_model.predict(resizepredict[:,:,:,np.newaxis] )
  for jjj in range(numlabel):
      segout_resize = skimage.transform.resize(segout[...,jjj],(nslice,_glexpx,_glexpx),order=0,preserve_range=True,mode='constant').transpose(2,1,0)
      segout_img = nib.Nifti1Image(segout_resize, None, header=imageheader)
      segout_img.to_filename( options.segmentation.replace('.nii.gz', '-%d.nii.gz' % jjj) )

#########################
# print help
#########################
else:
  parser.print_help()
