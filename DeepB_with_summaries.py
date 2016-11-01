from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from ReadDeepBs import *



flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_boolean('fake_data', False, 'If true, uses fake data '
                     'for unit testing.')
flags.DEFINE_integer('max_steps', 101, 'Number of steps to run trainer.')
flags.DEFINE_float('learning_rate', 0.003, 'Initial learning rate.')
flags.DEFINE_float('dropout', 0.9, 'Keep probability for training dropout.')
flags.DEFINE_string('data_dir', '/Users/markusstoye/DeepTensor/tensorflow-master/tensorflow/mycrap', 'Directory for storing data')
flags.DEFINE_string('summaries_dir', '/afs/cern.ch/work/m/mstoye/DeepGPU/DeepBs_logs', 'Summaries directory')


def train():
  # Import data
  DeepBdata = read_btag_data(FLAGS.data_dir,
                                    one_hot=True,
                                    fake_data=FLAGS.fake_data)

  sess = tf.InteractiveSession()
#  from keras import backend as K
#  K.set_session(sess)
  # Create a multilayer model.

  # Input placeholders
  with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 71], name='x-input')
    x_track = tf.placeholder(tf.float32, [None, 15, 25], name='x_track-input')
    y_ = tf.placeholder(tf.float32, [None, 5], name='y-input')

  # We can't initialize these variables to 0 - the network will get stuck.
  def weight_variable(shape):
    """Create a weight variable with appropriate initialization."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

  def bias_variable(shape):
    """Create a bias variable with appropriates initialization."""
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)

  def variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor."""
    with tf.name_scope('summaries'):
      mean = tf.reduce_mean(var)
      tf.scalar_summary('mean/' + name, mean)
      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
      tf.scalar_summary('sttdev/' + name, stddev)
      tf.scalar_summary('max/' + name, tf.reduce_max(var))
      tf.scalar_summary('min/' + name, tf.reduce_min(var))
      tf.histogram_summary(name, var)

  def nn_shared_weights_layer(input_tensor, input_dim, input_tensor_tracks,ntrack,block_track, tracklen, output_dim, layer_name, act=tf.nn.relu):
    activations = nn_layer(input_tensor, input_dim, output_dim, layer_name+"_expert")
    # loops over blocks of tracks that sharw weights
    for i in range(0,block_track):
      activations = nn_layer_tracks(input_tensor_tracks,i*ntrack/block_track, (i+1)*ntrack/block_track,tracklen, output_dim, layer_name+i,activations)
    return activations

  def nn_layer_tracks(input_tensor, first_track, last_track, tracklen, output_dim, layer_name, activations, act=tf.nn.relu):
    # This applies shared weights to the block of tracks
    
    with tf.name_scope(layer_name):
      # books shared weights
      with tf.name_scope('weights'):
          weights = weight_variable([tracklen, output_dim])
          variable_summaries(weights, layer_name + '/weights')
      with tf.name_scope('biases'):
          biases = bias_variable([output_dim])
          variable_summaries(biases, layer_name + '/biases')
      # loops over tracks and 
      for i in range(first_track, last_track):
        itensor = tf.slice(input_tensor,[0,i,0],[tf.shape(input_tensor)[0],1,tf.shape(input_tensor)[2]])
 #       with tf.name_scope('Wx_plus_b'):
        # calculates actiovation per track and sums them up
        preactivate = tf.matmul(itensor, weights) + biases
        activations = activations + act(preactivate, name='activation')
    return activations
      
  def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    """Reusable code for making a simple neural net layer.
 
    It does a matrix multiply, bias add, and then uses relu to nonlinearize.
    It also sets up name scoping so that the resultant graph is easy to read,
    and adds a number of summary ops.
    """
    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
      # This Variable will hold the state of the weights for the layer
      with tf.name_scope('weights'):
        weights = weight_variable([input_dim, output_dim])
        variable_summaries(weights, layer_name + '/weights')
      with tf.name_scope('biases'):
        biases = bias_variable([output_dim])
        variable_summaries(biases, layer_name + '/biases')
      with tf.name_scope('Wx_plus_b'):
#        biases    = tf.Print(biases, [biases], message=layer_name+"This is biases: ")
#        weights    = tf.Print(weights, [ weights], message=layer_name+"This is weights: ")        
#        input_tensor    = tf.Print(input_tensor, [ input_tensor], message=layer_name+"This is input_tensor: ")
        preactivate = tf.matmul(input_tensor, weights) + biases
#        preactivate    = tf.Print(preactivate, [preactivate], message=layer_name+" preactivate")     
        tf.histogram_summary(layer_name + '/pre_activations', preactivate)
      activations = act(preactivate, name='activation')
      tf.histogram_summary(layer_name + '/activations', activations)
      return activations
 # print(' this is a shape for x ' , tf.Variable.get_shape(x))
 # print(' printed x ',x)
 # print(' printed y ',y_)
  
 # print('really 1')
  hidden1 = nn_layer(x, 71, 100, 'layer1')
  hidden2 = nn_layer(hidden1, 100, 100, 'layer2')
  hidden3 = nn_layer(hidden2, 100, 100, 'layer3')
  hidden4 = nn_layer(hidden3, 100, 100, 'layer4')

  with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    tf.scalar_summary('dropout_keep_probability', keep_prob)
    dropped = tf.nn.dropout(hidden4, keep_prob)

#  y = nn_layer(dropped, 52, 2, 'layer3', act=tf.nn.softmax)
  y = nn_layer(dropped, 100, 5, 'layer5', act=tf.nn.softmax)
  
  with tf.name_scope('cross_entropy'):
 #   diff = y_ * tf.log(   tf.clip_by_value(y,1e-10,1.0))
    diff = y_ * tf.log(y)
 #   y = tf.Print(y, [y], message="This is y: ")
 #   x = tf.Print(x, [x], message="This is x: ")

    with tf.name_scope('total'):
      cross_entropy = -tf.reduce_mean(diff)
#      cross_entropy = tf.nn.softmax_cross_entropy_with_logits( y_,y)    
    tf.scalar_summary('cross entropy', cross_entropy)
    cross_entropy = tf.Print(cross_entropy, [cross_entropy], message="This is cross_entropy: ")
   

  with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(
        cross_entropy)

  with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
      correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
#      correct_prediction = tf.Print(correct_prediction, [correct_prediction], message="This is correct_prediction: ")
    with tf.name_scope('accuracy'):
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#      accuracy = tf.Print(accuracy, [accuracy], message="This is accuracy: ")
    
    tf.scalar_summary('accuracy', accuracy)

  # Merge all the summaries and write them out to /tmp/DeepBdata_logs (by default)
#  merged =  tf.constant(1)
  merged = tf.merge_all_summaries()
  train_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/train',
                                        sess.graph)
  test_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/test')
  tf.initialize_all_variables().run()

  # Train the model, and also write summaries.
  # Every 10th step, measure test-set accuracy, and write test summaries
  # All other steps, run train_step on training data, & add training summaries

  def feed_dict(train):
    """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
    if train==1 or FLAGS.fake_data:
      xs, ys = DeepBdata.train.next_batch(50000, fake_data=FLAGS.fake_data)
      k = FLAGS.dropout
      print("next batch")
    elif train==0:
      print("calling a test")
      xs, ys = DeepBdata.test.features, DeepBdata.test.labels
      k = 1.0
    elif train == 2:
      print("calling a validation")
      xs, ys = DeepBdata.validation.features, DeepBdata.validation.labels
      k = 1.0
    return {x: xs, y_: ys, keep_prob: k}
    
  for i in range(FLAGS.max_steps):
    if i % 100 == 0:  # Record summaries and test-set accuracy
      summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))     
      test_writer.add_summary(summary, i)
#      accVal = sess.run([ accuracy], feed_dict=feed_dict(2))
      print('Accuracy at step %s: %s' % (i, acc))
 #     print('Accuracy at step %s: %s' % (i, acc), ' and for QCD acc: %s' % ( accVal))
    else:  # Record train set summaries, and train
     # print('getting started')
      if i % 100 == 99:  # Record execution stats
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        summary, _ = sess.run([merged, train_step],
                              feed_dict=feed_dict(True),
                              options=run_options,
                              run_metadata=run_metadata)
        train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
        train_writer.add_summary(summary, i)
        print('Adding run metadata for', i)
      else:  # Record a summary
     #   print('make a summary')
        summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
        train_writer.add_summary(summary, i)
  y_write = sess.run(y, feed_dict=feed_dict(False))

  YrecArray = numpy.core.records.fromarrays(  y_write.transpose(), 
                                             names='prob_b, prob_c, prob_u,prob_bb, prob_cc',
                                             formats = 'float32,float32,float32,float32,float32')
#                                              names='prob_b, prob_c',
#                                              formats = 'float32,float32')
                          #, dtype=[('b_prob', numpy.float32), ('c_prob', numpy.float32), ('u_prob', numpy.float32), ('bb_prob', numpy.float32),('cc_prob', numpy.float32) ])
  numpy.save("test_result.npy",YrecArray)
  y_write_val = sess.run(y, feed_dict=feed_dict(2))
  YrecArray_val = numpy.core.records.fromarrays(  y_write_val.transpose(),
 #                                                 names='prob_b, prob_c',
#                                                  formats = 'float32,float32')
                                names='prob_b, prob_c, prob_u,prob_bb, prob_cc',
                                             formats = 'float32,float32,float32,float32,float32')
  numpy.save("test_result_val.npy",YrecArray_val)

  saver = tf.train.Saver()
  save_path = saver.save(sess, "model.npy")
 # print("Model saved in file: %s" % save_path)
  train_writer.close()
  test_writer.close()
#  from keras.models import load_model
#  model.save("ha.h5")
#  delete model
def main(_):
  if tf.gfile.Exists(FLAGS.summaries_dir):
    tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
  tf.gfile.MakeDirs(FLAGS.summaries_dir)
  train()


if __name__ == '__main__':
  tf.app.run()
