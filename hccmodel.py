import numpy as np
import os

# raw dicom data is usually short int (2bytes) datatype
# labels are usually uchar (1byte)
IMG_DTYPE = np.int16
SEG_DTYPE = np.uint8

# setup command line parser to control execution
from optparse import OptionParser
parser = OptionParser()
parser.add_option( "--initialize",
                  action="store_true", dest="initialize", default=False,
                  help="build initial sql file ", metavar = "BOOL")
parser.add_option( "--builddb",
                  action="store_true", dest="builddb", default=False,
                  help="load all training data into npy", metavar="FILE")
parser.add_option( "--traintumor",
                  action="store_true", dest="traintumor", default=False,
                  help="train model for tumor segmentation", metavar="FILE")
parser.add_option( "--setuptestset",
                  action="store_true", dest="setuptestset", default=False,
                  help="cross validate test set", metavar="FILE")
parser.add_option( "--setupcrctestset",
                  action="store_true", dest="setupcrctestset", default=False,
                  help="cross validate test set", metavar="FILE")
parser.add_option( "--debug",
                  action="store_true", dest="debug", default=False,
                  help="compare tutorial dtype", metavar="Bool")
parser.add_option( "--ModelID",
                  action="store", dest="modelid", default=None,
                  help="model id", metavar="FILE")
parser.add_option( "--outputModelBase",
                  action="store", dest="outputModelBase", default=None,
                  help="output location ", metavar="Path")
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
                  action="store", dest="trainingmodel", default='full',
                  help="setup info", metavar="string")
parser.add_option( "--trainingloss",
                  action="store", dest="trainingloss", default='dscimg',
                  help="setup info", metavar="string")
parser.add_option( "--sampleweight",
                  action="store", dest="sampleweight", default=None,
                  help="setup info", metavar="string")
parser.add_option( "--trainingsolver",
                  action="store", dest="trainingsolver", default='adadelta',
                  help="setup info", metavar="string")
parser.add_option( "--databaseid",
                  action="store", dest="databaseid", default='crc',
                  help="available data: hcc, crc", metavar="string")
parser.add_option( "--trainingresample",
                  type="int", dest="trainingresample", default=256,
                  help="setup info", metavar="int")
parser.add_option( "--trainingbatch",
                  type="int", dest="trainingbatch", default=5,
                  help="setup info", metavar="int")
parser.add_option( "--validationbatch",
                  type="int", dest="validationbatch", default=20,
                  help="setup info", metavar="int")
parser.add_option( "--kfolds",
                  type="int", dest="kfolds", default=5,
                  help="setup info", metavar="int")
parser.add_option( "--idfold",
                  type="int", dest="idfold", default=0,
                  help="setup info", metavar="int")
parser.add_option("--numepochs",
                  type="int", dest="numepochs", default=10,
                  help="number of epochs for training", metavar="int")
(options, args) = parser.parse_args()

# current datasets
trainingdictionary = {'hcc':{'dbfile':'/rsrch1/ip/dtfuentes/github/RandomForestHCCResponse/datalocation/trainingdata.csv','rootlocation':'/rsrch1/ip/dtfuentes/github/RandomForestHCCResponse'},
                      'hccnorm':{'dbfile':'/rsrch1/ip/dtfuentes/github/RandomForestHCCResponse/datalocation/trainingnorm.csv','rootlocation':'/rsrch1/ip/dtfuentes/github/RandomForestHCCResponse'},
                      'hccvol':{'dbfile':'/rsrch1/ip/dtfuentes/github/RandomForestHCCResponse/datalocation/tumordata.csv','rootlocation':'/rsrch1/ip/dtfuentes/github/RandomForestHCCResponse'},
                      'hccvolnorm':{'dbfile':'/rsrch1/ip/dtfuentes/github/RandomForestHCCResponse/datalocation/tumornorm.csv','rootlocation':'/rsrch1/ip/dtfuentes/github/RandomForestHCCResponse'},
                      'crc':{'dbfile':'./crctrainingdata.csv','rootlocation':'/rsrch1/ip/jacctor/LiTS/LiTS' }}

# options dependency 
options.dbfile       = trainingdictionary[options.databaseid]['dbfile']
options.rootlocation = trainingdictionary[options.databaseid]['rootlocation']
options.sqlitefile = options.dbfile.replace('.csv','.sqlite' )
options.globalnpfile = options.dbfile.replace('.csv','%d.npy' % options.trainingresample )
_globalexpectedpixel=512
print('database file: %s sqlfile: %s dbfile: %s rootlocation: %s' % (options.globalnpfile,options.sqlitefile,options.dbfile, options.rootlocation ) )
_globaldirectorytemplate = './%slog/%s/%s/%s/%d/%s/%03d%03d/%03d/%03d'
_xstr = lambda s: s or ""

# build data base from CSV file
def GetDataDictionary():
  import sqlite3
  CSVDictionary = {}
  tagsconn = sqlite3.connect(options.sqlitefile)
  cursor = tagsconn.execute(' SELECT aq.* from trainingdata aq ;' )
  names = [description[0] for description in cursor.description]
  sqlStudyList = [ dict(zip(names,xtmp)) for xtmp in cursor ]
  for row in sqlStudyList :
       CSVDictionary[int( row['dataid'])]  =  {'image':row['image'], 'label':row['label'], 'uid':"%s" %row['uid']}  
  return CSVDictionary 

# setup kfolds
def GetSetupKfolds(numfolds,idfold,dataidsfull ):
  from sklearn.model_selection import KFold

  if (numfolds < idfold or numfolds < 1):
     raise("data input error")
  # split in folds
  if (numfolds > 1):
     kf = KFold(n_splits=numfolds)
     allkfolds = [ (list(map(lambda iii: dataidsfull[iii], train_index)), list(map(lambda iii: dataidsfull[iii], test_index))) for train_index, test_index in kf.split(dataidsfull )]
     train_index = allkfolds[idfold][0]
     test_index  = allkfolds[idfold][1]
  else:
     train_index = np.array(dataidsfull )
     test_index  = None  
  return (train_index,test_index)

## Borrowed from
## $(SLICER_DIR)/CTK/Libs/DICOM/Core/Resources/dicom-schema.sql
## 
## --
## -- A simple SQLITE3 database schema for modelling locally stored DICOM files
## --
## -- Note: the semicolon at the end is necessary for the simple parser to separate
## --       the statements since the SQlite driver does not handle multiple
## --       commands per QSqlQuery::exec call!
## -- ;
## TODO note that SQLite does not enforce the length of a VARCHAR. 
## TODO (9) What is the maximum size of a VARCHAR in SQLite?
##
## TODO http://www.sqlite.org/faq.html#q9
##
## TODO SQLite does not enforce the length of a VARCHAR. You can declare a VARCHAR(10) and SQLite will be happy to store a 500-million character string there. And it will keep all 500-million characters intact. Your content is never truncated. SQLite understands the column type of "VARCHAR(N)" to be the same as "TEXT", regardless of the value of N.
initializedb = """
DROP TABLE IF EXISTS 'Images' ;
DROP TABLE IF EXISTS 'Patients' ;
DROP TABLE IF EXISTS 'Series' ;
DROP TABLE IF EXISTS 'Studies' ;
DROP TABLE IF EXISTS 'Directories' ;
DROP TABLE IF EXISTS 'lstat' ;
DROP TABLE IF EXISTS 'overlap' ;

CREATE TABLE 'Images' (
 'SOPInstanceUID' VARCHAR(64) NOT NULL,
 'Filename' VARCHAR(1024) NOT NULL ,
 'SeriesInstanceUID' VARCHAR(64) NOT NULL ,
 'InsertTimestamp' VARCHAR(20) NOT NULL ,
 PRIMARY KEY ('SOPInstanceUID') );
CREATE TABLE 'Patients' (
 'PatientsUID' INT PRIMARY KEY NOT NULL ,
 'StdOut'     varchar(1024) NULL ,
 'StdErr'     varchar(1024) NULL ,
 'ReturnCode' INT   NULL ,
 'FindStudiesCMD' VARCHAR(1024)  NULL );
CREATE TABLE 'Series' (
 'SeriesInstanceUID' VARCHAR(64) NOT NULL ,
 'StudyInstanceUID' VARCHAR(64) NOT NULL ,
 'Modality'         VARCHAR(64) NOT NULL ,
 'SeriesDescription' VARCHAR(255) NULL ,
 'StdOut'     varchar(1024) NULL ,
 'StdErr'     varchar(1024) NULL ,
 'ReturnCode' INT   NULL ,
 'MoveSeriesCMD'    VARCHAR(1024) NULL ,
 PRIMARY KEY ('SeriesInstanceUID','StudyInstanceUID') );
CREATE TABLE 'Studies' (
 'StudyInstanceUID' VARCHAR(64) NOT NULL ,
 'PatientsUID' INT NOT NULL ,
 'StudyDate' DATE NULL ,
 'StudyTime' VARCHAR(20) NULL ,
 'AccessionNumber' INT NULL ,
 'StdOut'     varchar(1024) NULL ,
 'StdErr'     varchar(1024) NULL ,
 'ReturnCode' INT   NULL ,
 'FindSeriesCMD'    VARCHAR(1024) NULL ,
 'StudyDescription' VARCHAR(255) NULL ,
 PRIMARY KEY ('StudyInstanceUID') );

CREATE TABLE 'Directories' (
 'Dirname' VARCHAR(1024) ,
 PRIMARY KEY ('Dirname') );

CREATE TABLE lstat  (
   InstanceUID        VARCHAR(255)  NOT NULL,  --  'studyuid *OR* seriesUID'
   SegmentationID     VARCHAR(80)   NOT NULL,  -- UID for segmentation file 
   FeatureID          VARCHAR(80)   NOT NULL,  -- UID for image feature     
   LabelID            INT           NOT NULL,  -- label id for LabelSOPUID statistics of FeatureSOPUID
   Mean               REAL              NULL,
   StdD               REAL              NULL,
   Max                REAL              NULL,
   Min                REAL              NULL,
   Count              INT               NULL,
   Volume             REAL              NULL,
   ExtentX            INT               NULL,
   ExtentY            INT               NULL,
   ExtentZ            INT               NULL,
   PRIMARY KEY (InstanceUID,SegmentationID,FeatureID,LabelID) );

-- expected csv format
-- FirstImage,SecondImage,LabelID,InstanceUID,MatchingFirst,MatchingSecond,SizeOverlap,DiceSimilarity,IntersectionRatio
CREATE TABLE overlap(
   FirstImage         VARCHAR(80)   NOT NULL,  -- UID for  FirstImage  
   SecondImage        VARCHAR(80)   NOT NULL,  -- UID for  SecondImage 
   LabelID            INT           NOT NULL,  -- label id for LabelSOPUID statistics of FeatureSOPUID 
   InstanceUID        VARCHAR(255)  NOT NULL,  --  'studyuid *OR* seriesUID',  
   -- output of c3d firstimage.nii.gz secondimage.nii.gz -overlap LabelID
   -- Computing overlap #1 and #2
   -- OVL: 6, 11703, 7362, 4648, 0.487595, 0.322397  
   MatchingFirst      int           DEFAULT NULL,     --   Matching voxels in first image:  11703
   MatchingSecond     int           DEFAULT NULL,     --   Matching voxels in second image: 7362
   SizeOverlap        int           DEFAULT NULL,     --   Size of overlap region:          4648
   DiceSimilarity     real          DEFAULT NULL,     --   Dice similarity coefficient:     0.487595
   IntersectionRatio  real          DEFAULT NULL,     --   Intersection / ratio:            0.322397
   PRIMARY KEY (InstanceUID,FirstImage,SecondImage,LabelID) );
"""


#############################################################
# build initial sql file 
#############################################################
if (options.initialize ):
  import sqlite3
  import pandas
  # build new database
  os.system('rm %s'  % options.sqlitefile )
  tagsconn = sqlite3.connect(options.sqlitefile )
  for sqlcmd in initializedb.split(";"):
     tagsconn.execute(sqlcmd )
  # load csv file
  df = pandas.read_csv(options.dbfile,delimiter='\t')
  df.to_sql('trainingdata', tagsconn , if_exists='append', index=False)

##########################
# preprocess database and store to disk
##########################
elif (options.builddb):
  import nibabel as nib  
  from scipy import ndimage
  import skimage.transform

  # create  custom data frame database type
  mydatabasetype = [('dataid', int), ('axialliverbounds',bool), ('axialtumorbounds',bool), ('imagedata','(%d,%d)int16' %(options.trainingresample,options.trainingresample)),('truthdata','(%d,%d)uint8' % (options.trainingresample,options.trainingresample))]

  # initialize empty dataframe
  numpydatabase = np.empty(0, dtype=mydatabasetype  )

  # build data base 
  databaseinfo = GetDataDictionary()

  # load all data 
  totalnslice = 0 
  for idrow in databaseinfo.keys():
    row = databaseinfo[idrow ]
    imagelocation = '%s/%s' % (options.rootlocation,row['image'])
    truthlocation = '%s/%s' % (options.rootlocation,row['label'])

    # load nifti file
    imagedata = nib.load(imagelocation )
    numpyimage= imagedata.get_data().astype(IMG_DTYPE )
    # error check
    assert numpyimage.shape[0:2] == (_globalexpectedpixel,_globalexpectedpixel)
    nslice = numpyimage.shape[2]
    resimage=skimage.transform.resize(numpyimage,(options.trainingresample,options.trainingresample,nslice),order=0,mode='constant',preserve_range=True).astype(IMG_DTYPE)

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
      boundingboxes = ndimage.find_objects(restruth)
      liverboundingbox = boundingboxes[0]

    # FIXME do we need this ?
    tumorboundingbox  = None

    print(idrow, imagelocation,truthlocation, nslice )

    # error check
    if( nslice  == restruth.shape[2]):
      # custom data type to subset  
      datamatrix = np.zeros(nslice  , dtype=mydatabasetype )
      
      # custom data type to subset  
      datamatrix ['dataid']          = np.repeat(idrow ,nslice  ) 
      #datamatrix ['xbounds']      = np.repeat(boundingbox[0],nslice  ) 
      #datamatrix ['ybounds']      = np.repeat(boundingbox[1],nslice  ) 
      #datamatrix ['zbounds']      = np.repeat(boundingbox[2],nslice  ) 
      #datamatrix ['nslice' ]      = np.repeat(nslice,nslice  ) 
      # id the slices within the bounding box
      axialliverbounds                              = np.repeat(False,nslice  ) 
      axialtumorbounds                              = np.repeat(False,nslice  ) 
      axialliverbounds[liverboundingbox[2]]         = True
      if (tumorboundingbox != None):
        axialtumorbounds[tumorboundingbox[2]]       = True
      datamatrix ['axialliverbounds'   ]            = axialliverbounds
      datamatrix ['axialtumorbounds'  ]             = axialtumorbounds
      datamatrix ['imagedata']                      = resimage.transpose(2,1,0) 
      datamatrix ['truthdata']                      = restruth.transpose(2,1,0)  
      numpydatabase = np.hstack((numpydatabase,datamatrix))
      # count total slice for QA
      totalnslice = totalnslice + nslice 
    else:
      print('training data error image[2] = %d , truth[2] = %d ' % (nslice,restruth.shape[2]))

  # save numpy array to disk
  np.save( options.globalnpfile,numpydatabase )

##########################
# build NN model for tumor segmentation
##########################
elif (options.traintumor):

  # load database
  print('loading memory map db for large dataset')
  #numpydatabase = np.load(options.globalnpfile,mmap_mode='r')
  numpydatabase = np.load(options.globalnpfile)
  dataidsfull= list(np.unique(numpydatabase['dataid']))

  #setup kfolds
  (train_validation_index,test_index) = GetSetupKfolds(options.kfolds,options.idfold,dataidsfull)

  #break into independent training and validation sets
  studydict = {'run_a':.9, 'run_b':.8, 'run_c':.7 }
  ntotaltrainval    =  len(train_validation_index)
  trainvalsplit     =  int(studydict[options.trainingid] * ntotaltrainval   )
  train_index       =  train_validation_index[0: trainvalsplit  ]
  validation_index  =  train_validation_index[trainvalsplit:    ]

  print("train_index:",train_index,' validation_index: ',validation_index,' test_index: ',test_index)

  # uses 'views' for efficient memory usage
  # https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.indexing.html
  print('copy data subsets into memory...')
  #axialbounds = numpydatabase['axialliverbounds'].copy()
  #dataidarray = numpydatabase['dataid'].copy()
  axialbounds = numpydatabase['axialliverbounds']
  dataidarray = numpydatabase['dataid']
  

  # setup indicies
  dbtrainindex         = np.isin(dataidarray,      train_index )
  dbvalidationindex    = np.isin(dataidarray, validation_index )
  dbtestindex          = np.isin(dataidarray,      test_index  )
  subsetidx_train      = np.all( np.vstack((axialbounds , dbtrainindex))      , axis=0 )
  subsetidx_validation = np.all( np.vstack((axialbounds , dbvalidationindex)) , axis=0 )
  subsetidx_test       = np.all( np.vstack((axialbounds , dbtestindex ))      , axis=0 )
  # error check
  if  np.sum(subsetidx_train   ) + np.sum(subsetidx_test)  + np.sum(subsetidx_validation ) != np.sum(axialbounds ) :
    raise("data error")
  print('copy memory map from disk to RAM...')

  # load training data as views
  #trainingsubset = numpydatabase[subsetidx   ].copy()
  trainingsubset   = numpydatabase[subsetidx_train      ]
  validationsubset = numpydatabase[subsetidx_validation ]

  # ensure we get the same results each time we run the code
  np.random.seed(seed=0) 
  np.random.shuffle(trainingsubset )
  np.random.shuffle(validationsubset )

  # subset within bounding box that has liver
  totnslice = len(trainingsubset) + len(validationsubset)
  slicesplit =  len(trainingsubset)
  print("nslice: ",totnslice ," split: " ,slicesplit )

  # FIXME - Verify stacking indicies
  x_train=np.vstack((trainingsubset['imagedata'],validationsubset['imagedata']))
  y_train=np.vstack((trainingsubset['truthdata'],validationsubset['truthdata']))
  TRAINING_SLICES      = slice(0,slicesplit)
  VALIDATION_SLICES    = slice(slicesplit,totnslice)


  if (options.sampleweight == None ):
     print('no sample weights')
     myweights = None
  else:
     from scipy import ndimage
     tumorvolumes= np.zeros(len(y_train[:]) )
     for iii in range( len(y_train[:]) ):
        tumorvolumes[iii]= ndimage.sum(y_train[iii],y_train[iii],index=[2])
     nonzerovolume = list(x for x in tumorvolumes if x > 0.)
     if (options.sampleweight == 'volume'):
        allweights = np.clip(1./(.01*tumorvolumes- 1.e-6),0,None)
        nonzeroweight = list(x for x in allweights if x > 0.)
        print('weights min: %12.5e max %12.5e' % (min(nonzeroweight),max(nonzeroweight) ) )
     elif (options.sampleweight == 'volumeshift'):
        allweights = np.clip(1./(.01*tumorvolumes- 1.e-6),0,None) + 1.
        nonzeroweight = list(x for x in allweights if x > 0.)
        print('weights min: %12.5e max %12.5e' % (min(nonzeroweight),max(nonzeroweight) ) )
     elif (options.sampleweight == 'volumehi'):
        allweights = np.clip(1./(.001*tumorvolumes- 1.e-6),0,None)
        nonzeroweight = list(x for x in allweights if x > 0.)
        print('weights min: %12.5e max %12.5e' % (min(nonzeroweight),max(nonzeroweight) ) )
     elif (options.sampleweight == 'volumeshifthi'):
        allweights = np.clip(1./(.001*tumorvolumes- 1.e-6),0,None) + 1.
        nonzeroweight = list(x for x in allweights if x > 0.)
        print('weights min: %12.5e max %12.5e' % (min(nonzeroweight),max(nonzeroweight) ) )
     else:
        raise('unknown weight')
     myweights = allweights[TRAINING_SLICES ]
      
  # import nibabel as nib  
  # print ( "writing training data for reference " ) 
  # imgnii = nib.Nifti1Image(x_train[: ,:,:] , None )
  # imgnii.to_filename( '%s/trainingimg.nii.gz' % anonymizeoutputlocation )
  # segnii = nib.Nifti1Image(y_train[: ,:,:] , None )
  # segnii.to_filename( '%s/trainingseg.nii.gz' % anonymizeoutputlocation )

  import keras
  import tensorflow as tf
  print("keras version: ",keras.__version__, 'TF version:',tf.__version__)
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
  def get_batchnorm_unet_vector(_filters=32, _filters_add=0, _kernel_size=(3,3), _padding='same', _activation='prelu', _kernel_regularizer=None, _final_layer_nonlinearity='sigmoid', _batch_norm=True, _num_classes=1):
      # FIXME - HACK image size
      crop_size = options.trainingresample
      if _padding == 'valid':
          input_layer = Input(shape=(crop_size+40,crop_size+40,2))
      elif _padding == 'same':
          input_layer = Input(shape=(crop_size,crop_size,2))
  
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
  
  def get_bnormfull_unet_vector(_filters=32, _filters_add=0, _kernel_size=(3,3), _padding='same', _activation='prelu', _kernel_regularizer=None, _final_layer_nonlinearity='sigmoid', _batch_norm=True, _num_classes=1):
      # FIXME - HACK image size
      crop_size = options.trainingresample
      if _padding == 'valid':
          input_layer = Input(shape=(crop_size+40,crop_size+40,2))
      elif _padding == 'same':
          input_layer = Input(shape=(crop_size,crop_size,2))
  
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
  
  def get_bnormover_unet_vector(_filters=32, _filters_add=0, _kernel_size=(3,3), _padding='same', _activation='prelu', _kernel_regularizer=None, _final_layer_nonlinearity='sigmoid', _batch_norm=True, _num_classes=1):
      # FIXME - HACK image size
      crop_size = options.trainingresample
      if _padding == 'valid':
          input_layer = Input(shape=(crop_size+40,crop_size+40,2))
      elif _padding == 'same':
          input_layer = Input(shape=(crop_size,crop_size,2))
  
      x0 = addConvBNSequential(input_layer, filters=_filters, kernel_size=_kernel_size, padding=_padding, activation=_activation, kernel_regularizer=_kernel_regularizer, batch_norm=_batch_norm)
      x0 = addConvBNSequential(x0,          filters=_filters, kernel_size=_kernel_size, padding=_padding, activation=_activation, kernel_regularizer=_kernel_regularizer, batch_norm=_batch_norm)
      x1 = MaxPool2D()(x0)
      
      x1 = addConvBNSequential(x1,         filters =_filters+_filters_add, kernel_size=_kernel_size, padding=_padding, activation=_activation, kernel_regularizer=_kernel_regularizer, batch_norm=_batch_norm)
      x1 = addConvBNSequential(x1,         filters =_filters+_filters_add, kernel_size=_kernel_size, padding=_padding, activation=_activation, kernel_regularizer=_kernel_regularizer, batch_norm=_batch_norm)
      x2 = MaxPool2D()(x1)
      
      x2 = addConvBNSequential(x2,         filters =2*(_filters+_filters_add), kernel_size=_kernel_size, padding=_padding, activation=_activation, kernel_regularizer=_kernel_regularizer, batch_norm=_batch_norm)
      x2 = addConvBNSequential(x2,         filters =2*(_filters+_filters_add), kernel_size=_kernel_size, padding=_padding, activation=_activation, kernel_regularizer=_kernel_regularizer, batch_norm=_batch_norm)
      x3 = MaxPool2D()(x2)
      
      x3 = addConvBNSequential(x3,         filters =4*(_filters+_filters_add), kernel_size=_kernel_size, padding=_padding, activation=_activation, kernel_regularizer=_kernel_regularizer, batch_norm=_batch_norm)
      x3 = addConvBNSequential(x3,         filters =4*(_filters+_filters_add), kernel_size=_kernel_size, padding=_padding, activation=_activation, kernel_regularizer=_kernel_regularizer, batch_norm=_batch_norm)
      x4 = MaxPool2D()(x3)
      
      x4 = addConvBNSequential(x4,         filters =8*(_filters+_filters_add), kernel_size=_kernel_size, padding=_padding, activation=_activation, kernel_regularizer=_kernel_regularizer, batch_norm=_batch_norm)
      x4 = addConvBNSequential(x4,         filters =8*(_filters+_filters_add), kernel_size=_kernel_size, padding=_padding, activation=_activation, kernel_regularizer=_kernel_regularizer, batch_norm=_batch_norm)
      x5 = MaxPool2D()(x4)
      
      x5 = addConvBNSequential(x5,         filters=16*(_filters+_filters_add), kernel_size=_kernel_size, padding=_padding, activation=_activation, kernel_regularizer=_kernel_regularizer, batch_norm=_batch_norm)
      x5 = addConvBNSequential(x5,         filters=16*(_filters+_filters_add), kernel_size=_kernel_size, padding=_padding, activation=_activation, kernel_regularizer=_kernel_regularizer, batch_norm=_batch_norm)
      x6 = MaxPool2D()(x5)
      
      x6 = addConvBNSequential(x6,         filters=32*(_filters+_filters_add), kernel_size=_kernel_size, padding=_padding, activation=_activation, kernel_regularizer=_kernel_regularizer, batch_norm=_batch_norm)
      x6 = addConvBNSequential(x6,         filters=32*(_filters+_filters_add), kernel_size=_kernel_size, padding=_padding, activation=_activation, kernel_regularizer=_kernel_regularizer, batch_norm=_batch_norm)
      x7 = UpSampling2D()(x6)
      
      x7 = concatenate([x5,x7])
      x7 = addConvBNSequential(x7,         filters=16*(_filters+_filters_add), kernel_size=_kernel_size, padding=_padding, activation=_activation, kernel_regularizer=_kernel_regularizer, batch_norm=_batch_norm,dropout=.5)
      x7 = addConvBNSequential(x7,         filters=16*(_filters+_filters_add), kernel_size=_kernel_size, padding=_padding, activation=_activation, kernel_regularizer=_kernel_regularizer, batch_norm=_batch_norm,dropout=.5)
      x8 = UpSampling2D()(x7)
      
      x8 = concatenate([x4,x8])
      x8 = addConvBNSequential(x8,         filters =8*(_filters+_filters_add), kernel_size=_kernel_size, padding=_padding, activation=_activation, kernel_regularizer=_kernel_regularizer, batch_norm=_batch_norm,dropout=.5)
      x8 = addConvBNSequential(x8,         filters =8*(_filters+_filters_add), kernel_size=_kernel_size, padding=_padding, activation=_activation, kernel_regularizer=_kernel_regularizer, batch_norm=_batch_norm,dropout=.5)
      x9 = UpSampling2D()(x8)
      
      x9 = concatenate([x3,x9])
      x9 = addConvBNSequential(x9,         filters =4*(_filters+_filters_add), kernel_size=_kernel_size, padding=_padding, activation=_activation, kernel_regularizer=_kernel_regularizer, batch_norm=_batch_norm,dropout=.5)
      x9 = addConvBNSequential(x9,         filters =4*(_filters+_filters_add), kernel_size=_kernel_size, padding=_padding, activation=_activation, kernel_regularizer=_kernel_regularizer, batch_norm=_batch_norm,dropout=.5)
      x10= UpSampling2D()(x9)
      
      x10= concatenate([x2,x10])
      x10= addConvBNSequential(x10,        filters =2*(_filters+_filters_add), kernel_size=_kernel_size, padding=_padding, activation=_activation, kernel_regularizer=_kernel_regularizer, batch_norm=_batch_norm,dropout=.5)
      x10= addConvBNSequential(x10,        filters =2*(_filters+_filters_add), kernel_size=_kernel_size, padding=_padding, activation=_activation, kernel_regularizer=_kernel_regularizer, batch_norm=_batch_norm,dropout=.5)
      x11= UpSampling2D()(x10)
      
      x11= concatenate([x1,x11])
      x11= addConvBNSequential(x11,        filters =_filters+_filters_add, kernel_size=_kernel_size, padding=_padding, activation=_activation, kernel_regularizer=_kernel_regularizer, batch_norm=_batch_norm,dropout=.5)
      x11= addConvBNSequential(x11,        filters =_filters+_filters_add, kernel_size=_kernel_size, padding=_padding, activation=_activation, kernel_regularizer=_kernel_regularizer, batch_norm=_batch_norm,dropout=.5)
      x12= UpSampling2D()(x11)
      
      x12= concatenate([x0,x12])
      x12= addConvBNSequential(x12,        filters =_filters, kernel_size=_kernel_size, padding=_padding, activation=_activation, kernel_regularizer=_kernel_regularizer, batch_norm=_batch_norm,dropout=.5)
      x12= addConvBNSequential(x12,        filters =_filters, kernel_size=_kernel_size, padding=_padding, activation=_activation, kernel_regularizer=_kernel_regularizer, batch_norm=_batch_norm,dropout=.5)
  
      # FIXME - need for arbitrary output
      output_layer = Conv2D(_num_classes, kernel_size=(1,1), activation=_final_layer_nonlinearity)(x12)
      
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
  #  K.eval(K.dot(kimagesum,K.variable([[.1,0.,0.,0.],[0.,.1,0.,0.],[0.,0.,.1,0.],[0.,0.,0., 1]]) ))

  #  NOTE - dice similarity and average are NOT commutative - DSC(AVG) .NE. AVG(DSC)
  #  NOTE - to get the same DSC values in c3d you need to break up the image in batches and compute the dsc for each batch and then average the dsc values per batch.
  #  NOTE - will get different values if you compute the dsc over the whole image without breaking into batches.
  def dice_imageloss(y_true, y_pred, smooth=0):
      """
      Dice = 1/Nbatch * \sum_Nbatch (2*|X & Y|)/ (|X|+ |Y|)
           = 1/Nbatch * \sum_Nbatch  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
      return negative dice value for minimization. average dsc over the batch per one hot image. Nbatch * Nonehot total images. 
      objective function has implicit reduce mean -  /opt/apps/miniconda/miniconda3/lib/python3.6/site-packages/keras/engine/training.py(447)weighted()
      """
      # DSC = DSC_image1 +  DSC_image2 + DSC_image3 + ...
      intersection = 2. *K.abs(y_true * y_pred) + smooth
      # FIXME - hard code sum over 2d image
      sumunion = K.sum(K.square(y_true),axis=(1,2)) + K.sum(K.square(y_pred),axis=(1,2)) + smooth
      dicevalues= K.sum(intersection / K.expand_dims(K.expand_dims(sumunion,axis=1),axis=2), axis=(1,2))
      return -dicevalues

  def dice_weightloss(y_true, y_pred, smooth=0):
      batchdiceloss =  dice_imageloss(y_true, y_pred)
      # increase weight on tumor
      # FIXME - hard code two labels
      return K.dot(batchdiceloss,K.variable([[.1,0.,0.],[0.,.1,0.],[0.,0.,1.]]) )

  def dice_hiweightloss(y_true, y_pred, smooth=0):
      batchdiceloss =  dice_imageloss(y_true, y_pred)
      # increase weight on tumor
      # FIXME - hard code two labels
      return K.dot(batchdiceloss,K.variable([[.01,0.,0.],[0.,.01,0.],[0.,0.,1.]]) )

  def dice_batchloss(y_true, y_pred, smooth=0):
      """
      Dice = (2*|X & Y|)/ (|X|+ |Y|)
           =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
      return negative dice value for minimization. one dsc per one hot image. each batch is treated as a 3d image for the DSC calculation for each one hot image
      objective function has implicit reduce mean that does not affect the scalar value per one hot -  /opt/apps/miniconda/miniconda3/lib/python3.6/site-packages/keras/engine/training.py(447)weighted()
      """
      # DSC = DSC_image1 +  DSC_image2 + DSC_image3 + ...
      intersection = 2. *K.abs(y_true * y_pred) + smooth
      # FIXME - hard code sum over 3d image
      sumunion = K.sum(K.square(y_true),axis=(0,1,2)) + K.sum(K.square(y_pred),axis=(0,1,2)) + smooth
      dicevalues= K.sum(intersection / K.expand_dims(K.expand_dims(K.expand_dims(sumunion,axis=0),axis=1),axis=2), axis=(0,1,2))
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
  def dice_metric_four(y_true, y_pred):
      batchdiceloss =  dice_imageloss(y_true, y_pred)
      return -batchdiceloss[:,4]
  def dice_metric_five(y_true, y_pred):
      batchdiceloss =  dice_imageloss(y_true, y_pred)
      return -batchdiceloss[:,5]
  def dice_volume_zero(y_true, y_pred):
      voldiceloss =  dice_batchloss(y_true, y_pred)
      return -voldiceloss[0]
  def dice_volume_one(y_true, y_pred):
      voldiceloss =  dice_batchloss(y_true, y_pred)
      return -voldiceloss[1]
  def dice_volume_two(y_true, y_pred):
      voldiceloss =  dice_batchloss(y_true, y_pred)
      return -voldiceloss[2]
  def dice_volume_three(y_true, y_pred):
      voldiceloss =  dice_batchloss(y_true, y_pred)
      return -voldiceloss[3]
  def dice_volume_four(y_true, y_pred):
      voldiceloss =  dice_batchloss(y_true, y_pred)
      return -voldiceloss[4]
  def dice_volume_five(y_true, y_pred):
      voldiceloss =  dice_batchloss(y_true, y_pred)
      return -voldiceloss[5]

  # Convert the labels into a one-hot representation
  from keras.utils.np_utils import to_categorical
  
  # Convert to uint8 data and find out how many labels.
  t=y_train.astype(np.uint8)
  t_max=np.max(t)
  print("Range of values: [0, {}]".format(t_max))
  y_train_one_hot = to_categorical(t, num_classes=t_max+1).reshape((y_train.shape)+(t_max+1,))
  print("Shape before: {}; Shape after: {}".format(y_train.shape, y_train_one_hot.shape))
  # The liver neuron should also be active for lesions within the liver
  # FIXME - HACK - data nuances
  if( options.databaseid == 'hcc'):
    liver = np.max(y_train_one_hot[:,:,:,1:-1], axis=3)
  elif( options.databaseid == 'hccnorm'):
    liver = np.max(y_train_one_hot[:,:,:,1:-1], axis=3)
  elif( options.databaseid == 'hccvol'):
    liver = np.max(y_train_one_hot[:,:,:,1:-1], axis=3)
  elif( options.databaseid == 'hccvolnorm'):
    liver = np.max(y_train_one_hot[:,:,:,1:-1], axis=3)
  elif( options.databaseid == 'crc'):
    liver = np.max(y_train_one_hot[:,:,:,1:], axis=3)
  else:
    raise("unknown  dataset")
  y_train_one_hot[:,:,:,1]=liver
  
  # vectorize input assume that liver mask is given
  x_train_vector = np.repeat(x_train[:,:,:,np.newaxis],2,axis=3)
  x_train_vector[:,:,:,1]=liver

  # output location
  logfileoutputdir= _globaldirectorytemplate % (options.databaseid,options.trainingloss+ _xstr(options.sampleweight),options.trainingmodel,options.trainingsolver,options.trainingresample,options.trainingid,options.trainingbatch,options.validationbatch,options.kfolds,options.idfold)

  print(logfileoutputdir)
  # ensure directory exists
  os.system ('mkdir -p %s' % logfileoutputdir)

  # tensor callbacks
  from keras.callbacks import TensorBoard
  tensorboard = TensorBoard(log_dir=logfileoutputdir, histogram_freq=0, write_graph=True, write_images=False)

  # callback to save best model 
  from keras.callbacks import Callback as CallbackBase
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
             model_json = self.model.to_json()
             with open("%s/tumormodelunet.json" % logfileoutputdir , "w") as json_file:
                 json_file.write(model_json)
             # serialize weights to HDF5
             self.model.save_weights("%s/tumormodelunet.h5" % logfileoutputdir )
             print("Saved model to disk - val_loss", self.min_valloss  )

             # output predictions
             if (options.trainingid == 'run_a'):
               import nibabel as nib  
               validationimgnii = nib.Nifti1Image(x_train[VALIDATION_SLICES,:,:] , None )
               validationimgnii.to_filename( '%s/validationimg.nii.gz' % logfileoutputdir )
               validationonehotnii = nib.Nifti1Image(y_train[VALIDATION_SLICES  ,:,:] , None )
               validationonehotnii.to_filename( '%s/validationseg.nii.gz' % logfileoutputdir )
               y_predicted = self.model.predict(x_train_vector[VALIDATION_SLICES,:,:,:])
               # liver mask should be close to 1.
               y_predicted[:,:,:,1] = .5 * y_predicted[:,:,:,1] 
               y_segmentation = np.argmax(y_predicted , axis=-1)
               validationprediction = nib.Nifti1Image(y_predicted, None )
               validationprediction.to_filename( '%s/validationpredict.nii.gz' % logfileoutputdir )
               validationoutput     = nib.Nifti1Image( y_segmentation.astype(np.uint8), None )
               validationoutput.to_filename( '%s/validationoutput.nii.gz' % logfileoutputdir )
          return
   
      def on_batch_begin(self, batch, logs={}):
          return
   
      def on_batch_end(self, batch, logs={}):
          return
  callbacksave = MyHistories()

  # dictionary of models to evaluate
  modeldict = {'half': get_batchnorm_unet_vector(_activation='relu', _batch_norm=True,_filters=64, _filters_add=64,_num_classes=t_max+1),
               'full': get_bnormfull_unet_vector(_activation='relu', _batch_norm=True,_filters=64, _filters_add=64,_num_classes=t_max+1),
               'over': get_bnormover_unet_vector(_activation='relu', _batch_norm=True,_filters=64, _filters_add=64,_num_classes=t_max+1)}

  # restart if previous model available
  modelpath  = "%s/tumormodelunet.json" % logfileoutputdir 
  weightsfile= "%s/tumormodelunet.h5"   % logfileoutputdir 
  if (os.path.isfile(modelpath)  and os.path.isfile(weightsfile)):
    from keras.models import model_from_json
    json_file = open(modelpath, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights(weightsfile)
    print("Loaded model from disk")
  else:
    model = modeldict[options.trainingmodel] 
    print("initialize new model")

  lossdict = {'dscvec': dice_coef_loss,'dscimg': dice_imageloss,'dscwgt': dice_weightloss,'dscwgthi': dice_hiweightloss}
  # FIXME - dice applied to each class separately, and weight each class
  # 
  # ojective function is summed
  #f    weighted          /opt/apps/miniconda/miniconda3/lib/python3.6/site-packages/keras/engine/training.py
  #             function:_weighted_masked_objective
  #             def weighted(y_true, y_pred, weights, mask=None):
  #model.compile(loss='categorical_crossentropy',optimizer='adadelta')
  metricsList=[dice_metric_zero,dice_metric_one,dice_metric_two,dice_metric_three,dice_metric_four,dice_metric_five]
  volumesList=[dice_volume_zero,dice_volume_one,dice_volume_two,dice_volume_three,dice_volume_four,dice_volume_five]
  model.compile(loss=lossdict[options.trainingloss],metrics=metricsList[:(t_max+1)]+volumesList[:(t_max+1)],optimizer=options.trainingsolver)
  print("Model parameters: {0:,}".format(model.count_params()))
  # FIXME - better to use more epochs on a single one-hot model? or break up into multiple models steps?
  # FIXME -  IE liver mask first then resize to the liver for viable/necrosis ? 

  from keras.preprocessing.image import ImageDataGenerator
  # Data generator for training. Allows different workers to request batches without interfering with other workers
  train_gen = ImageDataGenerator()
  valid_gen = ImageDataGenerator()
  #steps_per_epoch = (len(x_train_vector[TRAINING_SLICES,...]) // options.trainingbatch) // hvd.size() 
  steps_per_epoch = len(x_train_vector) // options.trainingbatch
  ## config file for image processor
  ## $ cat ~/.keras/keras.json 
  ## {
  ##     "floatx": "float32",
  ##     "epsilon": 1e-07,
  ##     "backend": "tensorflow",
  ##     "image_data_format": "channels_first"
  ## }
  train_iter = train_gen.flow(x_train_vector[TRAINING_SLICES ,:,:,:],
                              y_train_one_hot[TRAINING_SLICES ],
                              sample_weight=myweights,
                              batch_size = options.trainingbatch)
  valid_iter = valid_gen.flow(x_train_vector[VALIDATION_SLICES ,:,:,:],
                              y_train_one_hot[VALIDATION_SLICES ], 
                              batch_size = options.validationbatch)

  # fit_generator must be used instead of model.fit for distributed training
  history = model.fit_generator(train_iter,
                      steps_per_epoch=steps_per_epoch,
                      validation_data=valid_iter,
                      callbacks = [tensorboard,callbacksave],  # Note callbacksave is disabled
                      #workers = 1,           # More testing needs to be done to see how workers/use_multiprocessing impact horovod
                      #use_multiprocessing = False,
                      verbose = 1,
                      epochs=options.numepochs)


  ## history = model.fit(x_train_vector[TRAINING_SLICES ,:,:,:],
  ##                     y_train_one_hot[TRAINING_SLICES ],
  ##                     validation_data=(x_train_vector[VALIDATION_SLICES,:,:,:],y_train_one_hot[VALIDATION_SLICES]),
  ##                     callbacks = [tensorboard,callbacksave], sample_weight=myweights[TRAINING_SLICES ],
  ##                     batch_size=options.trainingbatch, epochs=options.numepochs)
  ##                     #batch_size=10, epochs=300
  
##########################
# apply model to test set
##########################
elif (options.setuptestset):
  # get id from setupfiles
  databaseinfo = GetDataDictionary()
  dataidsfull = list(databaseinfo.keys()) 

  uiddictionary = {}
  modeltargetlist = []

  makefilename = '%s%dkfold%03d.makefile' % (options.databaseid,options.trainingresample,options.kfolds) 
  # open makefile
  with open(makefilename ,'w') as fileHandle:
    for iii in range(options.kfolds):
      (train_set,test_set) = GetSetupKfolds(options.kfolds,iii,dataidsfull)
      uidoutputdir= _globaldirectorytemplate % (options.databaseid,options.trainingloss+ _xstr(options.sampleweight),options.trainingmodel,options.trainingsolver,options.trainingresample,options.trainingid,options.trainingbatch,options.validationbatch,options.kfolds,iii)
      modelprereq    = '%s/tumormodelunet.json' % uidoutputdir
      fileHandle.write('%s: \n' % modelprereq  )
      fileHandle.write('\tpython hccmodel.py --databaseid=%s --traintumor --idfold=%d --kfolds=%d --trainingresample=%d --numepochs=50\n' % (options.databaseid,iii,options.kfolds,options.trainingresample))
      modeltargetlist.append(modelprereq    )
      uiddictionary[iii]=[]
      for idtest in test_set:
         # write target
         imageprereq    = '$(TRAININGROOT)/%s' % databaseinfo[idtest]['image']
         maskprereq     = '$(TRAININGROOT)/ImageDatabase/%s/unet/mask.nii.gz' % databaseinfo[idtest]['uid']
         segmaketarget = '$(TRAININGROOT)/ImageDatabase/%s/unet%s/tumor.nii.gz' % (databaseinfo[idtest]['uid'], options.databaseid )
         uiddictionary[iii].append(databaseinfo[idtest]['uid'] )
         cvtestcmd = "python ./applymodel.py --predictimage=$< --modelpath=$(word 3, $^) --maskimage=$(word 2, $^) --segmentation=$@"  
         fileHandle.write('%s: %s %s %s\n' % (segmaketarget ,imageprereq,maskprereq,    modelprereq  ) )
         fileHandle.write('\t%s\n' % cvtestcmd)

  # build job list
  with open(makefilename , 'r') as original: datastream = original.read()
  with open(makefilename , 'w') as modified:
     modified.write( 'TRAININGROOT=%s\n' % options.rootlocation +'DATABASEID=unet%s\n' % options.databaseid + 'SQLITEDB=%s\n' % options.sqlitefile + "models: %s \n" % ' '.join(modeltargetlist))
     for idkey in uiddictionary.keys():
        modified.write("UIDLIST%d=%s \n" % (idkey,' '.join(uiddictionary[idkey])))
     modified.write("UIDLIST=%s \n" % " ".join(map(lambda x : "$(UIDLIST%d)" % x, uiddictionary.keys()))    +datastream)

##########################
# apply model to test set
##########################
elif (options.setupcrctestset):
  # get id from setupfiles
  databaseinfo = GetDataDictionary()
  dataidsfull = list(databaseinfo.keys()) 

  uiddictionary = {}
  modeltargetlist = []
  trainingsolverList = ['adadelta','RMSprop']
  makefileoutput = '%skfold%03d.makefile' % (options.databaseid,options.kfolds) 
  # open makefile
  with open(makefileoutput ,'w') as fileHandle:
    for trainingsolverid in trainingsolverList:
      for iii in range(options.kfolds):
        (train_set,test_set) = GetSetupKfolds(options.kfolds,iii,dataidsfull)
        uidoutputdir= _globaldirectorytemplate % (options.databaseid,options.trainingloss+ _xstr(options.sampleweight),options.trainingmodel,trainingsolverid      ,options.trainingresample,options.trainingid,options.trainingbatch,options.validationbatch,options.kfolds,iii)
        modelprereq    = '%s/tumormodelunet.json' % uidoutputdir
        modelweights   = '%s/tumormodelunet.h5' % uidoutputdir
        fileHandle.write('%s: \n' % modelprereq  )
        fileHandle.write('\tpython hccmodel.py --databaseid=%s --traintumor --trainingsolver=%s --idfold=%d --kfolds=%d --numepochs=50\n' % (options.databaseid,trainingsolverid,iii,options.kfolds))
        modeltargetlist.append(modelprereq    )
        uiddictionary[iii]=[]
        for idtest in test_set:
           # write target
           imageprereq    = '$(TRAININGROOT)/%s' % databaseinfo[idtest]['image']
           labelprereq    = '$(TRAININGROOT)/%s' % databaseinfo[idtest]['label']
           setuptarget    = '$(WORKDIR)/%s/unet%s/setup' % (databaseinfo[idtest]['uid'],trainingsolverid)
           uiddictionary[iii].append(databaseinfo[idtest]['uid'] )
           cvtestcmd = "python ./applymodel.py --predictimage=$< --modelpath=$(word 3, $^) --maskimage=$(word 2, $^) --segmentation=$@"  
           fileHandle.write('%s: \n' % (setuptarget  ) )
           fileHandle.write('\tmkdir -p   $(@D)          \n'                  )
           fileHandle.write('\tln -snf %s $(@D)/image.nii\n' % imageprereq    )
           fileHandle.write('\tln -snf %s $(@D)/label.nii\n' % labelprereq    )
           fileHandle.write('\tln -snf ../../../%s $(@D)/tumormodelunet.json\n' % modelprereq  )
           fileHandle.write('\tln -snf ../../../%s $(@D)/tumormodelunet.h5\n' % modelweights  )

  # build job list
  with open(makefileoutput , 'r') as original: datastream = original.read()
  with open(makefileoutput , 'w') as modified:
     modified.write( 'TRAININGROOT=%s\n' % options.rootlocation + 'SQLITEDB=%s\n' % options.sqlitefile + "models: %s \n" % ' '.join(modeltargetlist))
     for idkey in uiddictionary.keys():
        modified.write("UIDLIST%d=%s \n" % (idkey,' '.join(uiddictionary[idkey])))
     modified.write("UIDLIST=%s \n" % " ".join(map(lambda x : "$(UIDLIST%d)" % x, uiddictionary.keys()))    +datastream)


##########################
# print help
##########################
else:
  parser.print_help()
