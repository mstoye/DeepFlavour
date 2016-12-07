from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from ReadDeepBs import *
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', '/Users/markusstoye/DeepTensor/tensorflow-master/tensorflow/mycrap', 'Directory for storing data')
flags.DEFINE_boolean('fake_data', False, 'If true, uses fake data '
                     'for unit testing.')

from keras import backend as K 

def trunc_norm_init(shape, name = None):
    #myscale = 0.1
    #values = numpy.random.normal(loc=0,scale=myscale,size=shape)
    #values = numpy.clip(values,-2*myscale,2*myscale)
    #return K.variable(values,name = None)
    initial = tf.truncated_normal(shape, stddev=0.05)
    return K.variable(initial,name = name)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

model = Sequential()
#model.add(Dense(100, activation='relu', input_dim=67,init=trunc_norm_init))
#model.add(Dense(100, activation='relu',init=trunc_norm_init))
#model.add(Dense(100, activation='relu',init=trunc_norm_init))
#model.add(Dense(100, activation='relu',init=trunc_norm_init))
model.add(Dense(100, activation='relu', input_dim=70,init='lecun_uniform'))
model.add(Dense(100, activation='relu',init='lecun_uniform'))
model.add(Dense(100, activation='relu',init='lecun_uniform'))
model.add(Dense(100, activation='relu',init='lecun_uniform'))
#model.add(Dense(100, activation='relu',init='lecun_uniform'))
#model.add(Dropout(0.1))
model.add(Dense(5, activation='softmax',init='lecun_uniform'))
sgd = SGD(lr=0.03, decay=1e-6,  momentum=0.9,  nesterov=True)
from keras.optimizers import Adam
adam = Adam(lr=0.0003)

model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
#model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

DeepBdata = read_btag_data(FLAGS.data_dir, one_hot=True, fake_data=FLAGS.fake_data)
model.fit(DeepBdata.train.features, DeepBdata.train.labels, nb_epoch=250, batch_size=50000)
score = model.evaluate(DeepBdata.test.features, DeepBdata.test.labels, batch_size=50000)
predict_test = model.predict( DeepBdata.test.features,batch_size=50000)
predict_val =  model.predict( DeepBdata.validation.features,batch_size=50000)

predict_test_write = numpy.core.records.fromarrays(  predict_test.transpose(), 
                                             names='prob_b, prob_c, prob_u,prob_bb, prob_cc',
                                             formats = 'float32,float32,float32,float32,float32')

predict_val_write = numpy.core.records.fromarrays(  predict_val.transpose(), 
                                             names='prob_b, prob_c, prob_u,prob_bb, prob_cc',
                                             formats = 'float32,float32,float32,float32,float32')

numpy.save("KERAS_test_result_v6.npy",predict_test_write)
numpy.save("KERAS_tval_result._v6npy",predict_val_write)
from keras.models import load_model
model.save("DeeBFlovour_KERAS_cMVA_Debug_v6.h5")
print (model.trainable_weights)
model.save_weights('Weights_cMVA_v6.h5')
json_string = model.to_json()
text_file = open("Architecture_cMVA_v6.json", "w")
text_file.write(json_string)
text_file.close()

