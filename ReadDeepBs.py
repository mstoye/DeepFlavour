import numpy
#import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.contrib.learn.python.learn.datasets import base
import resource

def read_features_labels(filenameX,filenameY):
  print('Read label from ', filenameY, ' and features from: ' , filenameX)
  y = numpy.load(filenameY)
  x = numpy.load(filenameX)
 # print(x[:,5])
# fixing the bug did not make a difference.
 # print(x[:,9])
 # x[:,9][ x[:,9] > (0.3-0.064)/0.233  ] = -0.065
 # print('After debug 9 ', x[:,9])
  
  y = y.astype('float32')
  x = x.astype('float32')
#  x = numpy.delete(x, [2,3,4,5], 1)
  

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

  #x , y,  = read_features_labels("/afs/cern.ch/work/m/mstoye/root_numpy/newMean/allMix_Conv+_train_X.npy","/afs/cern.ch/work/m/mstoye/root_numpy/newMean/allMix_train2_Y.npy")
#  x , y,  = read_features_labels("/afs/cern.ch/work/m/mstoye/root_numpy/newMean/QCDflat_X_Conv.npy","/afs/cern.ch/work/m/mstoye/root_numpy/newMean/QCDflat_Y.npy")
  #x , y,  = read_features_labels("/afs/cern.ch/work/m/mstoye/root_numpy/bigSamples/bcQCDu_X.npy","/afs/cern.ch/work/m/mstoye/root_numpy/bigSamples/bcQCDu_Y.npy")
  x , y,  = read_features_labels("/afs/cern.ch/work/m/mstoye/root_numpy/debugTrack//MIX_X.npy","/afs/cern.ch/work/m/mstoye/root_numpy/debugTrack//MIX_Y.npy")
#  x , y = read_features_labels("/afs/cern.ch/work/m/mstoye/root_numpy/bigSamples/JetTaggingVariables_ttbar_clean_prepro_X.npy","/afs/cern.ch/work/m/mstoye/root_numpy/bigSamples/JetTaggingVariables_ttbar_clean_Y.npy")

  xtest_external , ytest_external = read_features_labels("/afs/cern.ch/work/m/mstoye/root_numpy/debugTrack/JetTaggingVariablesDebug_prepro_X.npy","/afs/cern.ch/work/m/mstoye/root_numpy/debugTrack/JetTaggingVariablesDebug_Y.npy")
#  xtest_external , ytest_external = read_features_labels("ttbar/ttbar_Conv_X.npy","ttbar/ttbar_Conv_Y.npy") 
#  xval , yval = read_features_labels("/afs/cern.ch/work/m/mstoye/root_numpy/newMean/JetTaggingVariables_X.npy","/afs/cern.ch/work/m/mstoye/root_numpy/newMean/JetTaggingVariables_Y.npy") 
#  xval , yval = read_features_labels("/afs/cern.ch/work/m/mstoye/root_numpy/bigSamples/JetTaggingVariablesDebug_prepro_X.npy","/afs/cern.ch/work/m/mstoye/root_numpy/bigSamples/JetTaggingVariablesDebug_Y.npy") 
  xval , yval = read_features_labels("/afs/cern.ch/work/m/mstoye/root_numpy/debugTrack/JetTaggingVariables_ttbar_prepro_X.npy","/afs/cern.ch/work/m/mstoye/root_numpy/debugTrack/JetTaggingVariables_ttbar_clean_Y.npy")

  train = DataSet(x, y )
  validation = DataSet(xval,yval)
  test =DataSet(xtest_external , ytest_external)
  return base.Datasets(train=train, validation=validation, test=test)

 
