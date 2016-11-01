import numpy
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.contrib.learn.python.learn.datasets import base
import resource

def read_features_labels(filenameX,filenameY):
  print('Read label from ', filenameY, ' and features from: ' , filenameX)
  y = numpy.load(filenameY)
  x = numpy.load(filenameX)
  y = y.astype('float32')
  x = x.astype('float32')
  return x,y

def extract_features_labels(filenameX,filenameY):
  
  print('Extracting label from ', filenameY, ' and features from: ' , filenameX)
 # print('free and happy ', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000)
  y = numpy.load(filenameY)
  y = y.astype('float32')

# below reduces to 2 D
#  print(y)
#  print(y.shape)
#  print('shapes')
#  Y_true = numpy.vstack((y[:,0],y[:,3])).transpose() 
#  Y_false = numpy.vstack((y[:,1],y[:,2]))
#  Y_false = numpy.vstack((Y_false,y[:,4])).transpose() 
  #print( Y_false)
#  print( Y_false.shape)
#  Y_false = numpy.sum( Y_false, axis=1)
#  Y_true = numpy.sum( Y_true, axis=1)
#  y = numpy.vstack( (Y_true, Y_false )).transpose() 
#  print(y.shape)
#  print(y)


#  y = numpy.delete(y, [3,4],1)
#  print('Read y ', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000)
  X = numpy.load(filenameX)
  mynames = X.dtype.names
  print ('number of branches (root jargon) ', len(mynames) )
  for index , name in enumerate(mynames):
         print (' branch index ', index, ' branch name ',name)
  CheckMVA = X['Jet_cMVAv2']
  x = X.view(numpy.float32).reshape(X.shape + (-1,))
  x = x.astype('float32')
  # drop branches you do not want to be trained on
#  x = numpy.delete(x, [6,7], 1)
  
  for i in range(6):
    print(90-i)
    x = numpy.delete(x, [90-i], 1)
# CSV  x = numpy.delete(x, [2,3,4,5,6,7, 61,62,65,66,69,70,73,74,77,78,81,82], 1)
  x = numpy.delete(x, [4,5,61,62,65,66,69,70,73,74,77,78,81,82], 1)

  print(y)
  #print( x[0][0],' ' , x[0][1], ' ', x[0][2],' ' , x[0][3], ' ',x[0][4],' ' , x[0][5], ' ', x[0][6],' ' , x[0][7] , ' and ' , CheckMVA)
  
  #  print('Read X ', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000)


  
  # subtract mean and 
  for i in range (x.shape[1]):
    x[:,i][ numpy.isnan(  x[:,i])  ] = 1.
    x[:,i][ numpy.isinf(  x[:,i])  ] = -0.2
    if (i==53 or i==54 or i==55):
      print   (x[:,i])
      x[:,i][ x[:,i] == 0  ] = -99
      print (x[:,i])
#          x[:,i][]
    Cur = x[:,i]
    
  # values are set upstream (in CMSSW to -99 if not present). We want mean and std without using the -99s
    CutTHres = Cur[Cur>-90]
#    print (i, 'before a threshold ' , x[:,i].size, 'before a threshold ' , CutTHres.size)
    CutTHresMean=CutTHres.mean(axis=0)
    CutTHresStdv=CutTHres.std(axis=0)
    x[:,i] = numpy.subtract(x[:,i],CutTHresMean)
    # values are set upstream (in CMSSW to -99 if not present). We want them at 0, which e.g. also UCI did.
    x[:,i][ x[:,i]<-90-CutTHresMean] = 0.
    x[:,i] = numpy.divide( x[:,i],CutTHresStdv)    
    print('Feature ', i, ' after rescaling. mean: ', x[:,i].mean(axis=0) , ' , std: ' ,x[:,i].std(axis=0), x.shape)

  print(' x (feature) shape ', x.shape, ' y (truth) shape ', y.shape)
  # deletes useless branches (for now)
 # for i in range(6,57):
#    print(i)
#    x = numpy.delete(x, [i], 1)
 
  print('Finished extraction')
  return x,y
 
def split_train_test_validayion(x,y,val,test):
  print('Slice dataset')
  length = len(x)
  trainInt = int(round(length*(1-val-test)))
  valInt = int(round(length*(1-test)))
  xtrain = x[0:trainInt]
  ytrain = y[0:trainInt]
  xval = x[trainInt+1:valInt]
  yval = y[trainInt+1:valInt]
  xtest = x[valInt+1:length]
  ytest = y[valInt+1:length]
  print('Finished slicing')
  return [xtrain,ytrain,xval,yval,xtest,ytest]

class DataSet(object):

  def __init__(self,
               features,
               labels,
               fake_data=False,
               one_hot=False,
               dtype=dtypes.float32,
               reshape=True):
    """Construct a DataSet.
    one_hot arg is used only if fake_data is true.  `dtype` can be either
    `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
    `[0, 1]`.
    """
    dtype = dtypes.as_dtype(dtype).base_dtype
    # chechs number of jets to train on
    self._num_examples = features.shape[0]
    # makes extra sure we use float 32
    if dtype == dtypes.float32:
        features = features.astype(numpy.float32)
        
    self._features = features
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def features(self):
    return self._features

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, fake_data=False):
    """Return the next `batch_size` examples from this data set."""
    if fake_data:
      fake_image = [1] * 784
      if self.one_hot:
        fake_label = [1] + [0] * 9
      else:
        fake_label = 0
      return [fake_image for _ in xrange(batch_size)], [
          fake_label for _ in xrange(batch_size)
      ]
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1

      # Shuffle the data
      #perm = numpy.arange(self._num_examples)
      #numpy.random.shuffle(perm)
      #self._features = self._features[perm]
      #self._labels = self._labels[perm]

      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._features[start:end], self._labels[start:end]

def read_btag_data(filename, one_hot=True,
                                    fake_data=False):

  x , y,  = read_features_labels("/afs/cern.ch/work/m/mstoye/root_numpy/newMean/allMix_Conv+_train_X.npy","/afs/cern.ch/work/m/mstoye/root_numpy/newMean/allMix_train2_Y.npy")
  #x , y,  = extract_features_labels("mainlyQCD/all_small_X.npy","mainlyQCD/all_small_Y.npy")
 # x , y,  = extract_features_labels("QCD/QCDflat_X.npy","QCD/QCDflat_Y.npy")

#  [ xtrain, ytrain , xval , yval, xtest , ytest] = split_train_test_validayion(x,y,0.01,0.01)
  xtest_external , ytest_external = read_features_labels("/afs/cern.ch/work/m/mstoye/root_numpy/newMean/JetTaggingVariablesQCD50to80_X_Conv.npy","/afs/cern.ch/work/m/mstoye/root_numpy/newMean/JetTaggingVariablesQCD50to80_Y.npy") 
#  xtest_external , ytest_external = read_features_labels("ttbar/ttbar_Conv_X.npy","ttbar/ttbar_Conv_Y.npy") 
  xval , yval = read_features_labels("/afs/cern.ch/work/m/mstoye/root_numpy/newMean/JetTaggingVariables_X.npy","/afs/cern.ch/work/m/mstoye/root_numpy/newMean/JetTaggingVariables_Y.npy") 
#  xval , yval = extract_features_labels("QCD/JetTaggingVariablesQCD50to80_X.npy","QCD/JetTaggingVariablesQCD50to80_Y.npy") 

  train = DataSet(x, y )
  validation = DataSet(xval,yval)
  test =DataSet(xtest_external , ytest_external)
  return base.Datasets(train=train, validation=validation, test=test)

 
