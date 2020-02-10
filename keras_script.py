# https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
import numpy as np
import keras

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


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=32, dim=(32,32,32), n_channels=1,
                 n_classes=10, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = np.load('data/' + ID + '.npy')

            # Store class
            y[i] = self.labels[ID]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)


import numpy as np

from keras.models import Sequential

numpydatabase = np.load('/home/fuentes/dbgtrainingdata256.npy')

dataidsfull= list(np.unique(numpydatabase['dataid']))

#setup kfolds
(train_validation_index,test_index) = GetSetupKfolds(5,0,dataidsfull)

#break into independent training and validation sets
ntotaltrainval    =  len(train_validation_index)
trainvalsplit     =  int(.9 * ntotaltrainval   )
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
# 

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


# Parameters
params = {'dim': (32,32,32),
          'batch_size': 64,
          'n_classes': 6,
          'n_channels': 1,
          'shuffle': True}

# Datasets
#partition = # IDs
#labels = # Labels

# Generators
training_generator = DataGenerator(partition['train'], labels, **params)
validation_generator = DataGenerator(partition['validation'], labels, **params)

# Design model
model = Sequential()
[...] # Architecture
model.compile()

# Train model on dataset
model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=True,
                    workers=6)
