# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A library to train Inception using multiple GPUs with synchronous updates.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
from datetime import datetime
import os.path
import re
import time

import numpy as np
import tensorflow as tf
import sys
sys.path.append('/home/lyt/inception_without_summary/inception')
from inception import inception_model as inception
from inception.slim import slim

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/tmp/imagenet_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 10000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_string('subset', 'train',
                           """Either 'train' or 'validation'.""")

# Flags governing the hardware employed for running TensorFlow.
tf.app.flags.DEFINE_integer('num_gpus', 1,
                            """How many GPUs to use.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")

# Flags governing the type of training.
tf.app.flags.DEFINE_boolean('fine_tune', False,
                            """If set, randomly initialize the final layer """
                            """of weights in order to train the network on a """
                            """new task.""")

# **IMPORTANT**
# Please note that this learning rate schedule is heavily dependent on the
# hardware architecture, batch size and any changes to the model architecture
# specification. Selecting a finely tuned learning rate schedule is an
# empirical process that requires some experimentation. Please see README.md
# more guidance and discussion.
#
# With 8 Tesla K40's and a batch size = 256, the following setup achieves
# precision@1 = 73.5% after 100 hours and 100K steps (20 epochs).
# Learning rate decay factor selected from http://arxiv.org/abs/1404.5997.
tf.app.flags.DEFINE_float('initial_learning_rate', 0.1,
                          """Initial learning rate.""")
tf.app.flags.DEFINE_float('num_epochs_per_decay', 30.0,
                          """Epochs after which learning rate decays.""")
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.16,
                          """Learning rate decay factor.""")

# Constants dictating the learning rate schedule.
RMSPROP_DECAY = 0.9                # Decay term for RMSProp.
RMSPROP_MOMENTUM = 0.9             # Momentum in RMSProp.
RMSPROP_EPSILON = 1.0              # Epsilon term for RMSProp.

def _tower_logits(images, num_classes, scope, reuse_variables=None):
  restore_logits = not FLAGS.fine_tune

  # Build inference Graph.
  with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):
    logits = inception.inference(images, num_classes, for_training=False,
                                 restore_logits=restore_logits,
                                 scope=scope)
  return logits
 
def _tower_loss(images, labels, num_classes, scope, reuse_variables=None):
  """
  Calculate the total loss on a single tower running the ImageNet model.

  We perform 'batch splitting'. This means that we cut up a batch across
  multiple GPUs. For instance, if the batch size = 32 and num_gpus = 2,
  then each tower will operate on an batch of 16 images.

  Args:
    images: Images. 4D tensor of size [batch_size, FLAGS.image_size,
                                       FLAGS.image_size, 3].
    labels: 1-D integer Tensor of [batch_size].
    num_classes: number of classes
    scope: unique prefix string identifying the ImageNet tower, e.g.
      'tower_0'.

  Returns:
     Tensor of shape [] containing the total loss for a batch of data
  """
  # When fine-tuning a model, we do not restore the logits but instead we
  # randomly initialize the logits. The number of classes in the output of the
  # logit is the number of classes in specified Dataset.
  restore_logits = not FLAGS.fine_tune

  # Build inference Graph.
  with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):
    logits = inception.inference(images, num_classes, for_training=True,
                                 restore_logits=restore_logits,
                                 scope=scope)

  # Build the portion of the Graph calculating the losses. Note that we will
  # assemble the total_loss using a custom function below.
  split_batch_size = images.get_shape().as_list()[0]
  inception.loss(logits, labels, batch_size=split_batch_size)

  # Assemble all of the losses for the current tower only.
  losses = tf.get_collection(slim.losses.LOSSES_COLLECTION, scope)

  # Calculate the total loss for the current tower.
  regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
  total_loss = tf.add_n(losses + regularization_losses, name='single_loss')

  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summmary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on TensorBoard.
    loss_name = re.sub('%s_[0-9]*/' % inception.TOWER_NAME, '', l.op.name)
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.

  with tf.control_dependencies([loss_averages_op]):
    total_loss = tf.identity(total_loss, name = 'total_loss')
  return total_loss, logits


np.random.seed(1)
ima = np.random.ranf(size=[32, 299, 299, 3])
ima = ima.astype('float32')
images = tf.Variable(ima, name='images', trainable=False)
#images = tf.placeholder(tf.float32, shape=(32, 299, 299, 3), name='images')
#images = tf.get_variable('images', [16,224,224,3], initializer=tf.random_normal_initializer(0,0.3))
#labels = np.random.randint(1000, size=32, dtype = 'int32')
#labels = tf.placeholder(tf.int32, shape=(None), name='labels')
'''
total_loss, logits = _tower_loss(images = images, labels = labels, num_classes = 1000 + 1, scope = 'tower', reuse_variables=None)
global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
num_batches_per_epoch = 100
decay_steps = int(num_batches_per_epoch * FLAGS.num_epochs_per_decay)
lr = tf.train.exponential_decay(FLAGS.initial_learning_rate,
                                global_step,
                                decay_steps,
                                FLAGS.learning_rate_decay_factor,
                                staircase=True)
opt = tf.train.RMSPropOptimizer(lr, RMSPROP_DECAY,
                                momentum=RMSPROP_MOMENTUM,
                                epsilon=RMSPROP_EPSILON)


batchnorm_updates = tf.get_collection(slim.ops.UPDATE_OPS_COLLECTION)

grads = opt.compute_gradients(total_loss)
apply_gradient_op = opt.apply_gradients(grads, global_step=global_step, name='apply_gradients')
variable_averages = tf.train.ExponentialMovingAverage(inception.MOVING_AVERAGE_DECAY, global_step)

variables_to_average = (tf.trainable_variables() +
                        tf.moving_average_variables())
variables_averages_op = variable_averages.apply(variables_to_average)
batchnorm_updates_op = tf.group(*batchnorm_updates)
train_op = tf.group(apply_gradient_op, variables_averages_op,
                    batchnorm_updates_op, name='train_op')

'''
logits = _tower_logits(images = images, num_classes = 1000 + 1, scope = 'tower', reuse_variables=None)
mylogits = tf.identity(logits, name='mylogits')
""" training """
init = tf.global_variables_initializer()
#opname = '_retval_mylogits_0_0'
#op = tf.get_default_graph().get_operation_by_name(opname)
sess = tf.Session()
os.rename('/home/lyt/static_analysis/replacement/replace_init.txt', '/home/lyt/static_analysis/replacement/replace.txt')
sess.run(init)
os.rename('/home/lyt/static_analysis/replacement/replace.txt', '/home/lyt/static_analysis/replacement/replace_init.txt')


os.rename('/home/lyt/static_analysis/replacement/replace_train.txt', '/home/lyt/static_analysis/replacement/replace.txt')
sess.run(mylogits)


for i in range(20):
    sess.run(mylogits)

start = time.clock()
for i in range(100):
    sess.run(mylogits)
print((time.clock() - start) * 1000 / 100)
os.rename('/home/lyt/static_analysis/replacement/replace.txt', '/home/lyt/static_analysis/replacement/replace_train.txt')
#print(sess.run([train_op, op]))
'''
with tf.Session() as sess:
    sess.run(init)
    subg_fd = open('/home/lyt/static_analysis/result/graph_1.txt', "r")
    line = subg_fd.readline()
    dic = dict()
    while line:
        line = line.split('||')[0]
        split_line = line.split('\t')
        if len(split_line) > 1:
            curID = int(split_line[0].split('[')[1].split(']')[0])
            for ID in split_line[1:]:
                if int(ID) not in dic.keys():
                    dic[int(ID)] = set()
                dic[int(ID)].add(curID)
        line = subg_fd.readline()
    dic[0] = set()
    subg_fd.close()
    path = list()
    pathtime = list()
    path.append(1)
    subg_fd = open('/home/lyt/static_analysis/result/graph_1.txt', "r")
    lines = subg_fd.readlines()
    subg_fd.close()
    def get_run_time(op_id, sess):
        op_id_name = lines[op_id].split('[')[3].split(']')[0]
        if op_id_name == '_SOURCE':
            return 0
        try:
            op = tf.get_default_graph().get_operation_by_name(op_id_name)
        except KeyError as e:
            print("can't run the op, try to find previous ops")
            max_time = 0
            for parent_id in dic[op_id]:
                temp_time = get_run_time(parent_id, sess)
                if temp_time > max_time:
                    max_time = temp_time
            return max_time
        else:
            for i in range(20):
                sess.run(op)
            #print("warm up done")
            start = time.clock()
            for i in range(30):
                sess.run(op)
            res_duration = (time.clock() - start) * 1000 / 30
            #print("duration: ", res_duration, "ms")
            return res_duration


    current_op_id = 1
    start_time = time.clock()
    while current_op_id in dic.keys() and len(dic[current_op_id]) > 0:
        max_time = 0
        max_time_id = 0
        for op_id in dic[current_op_id]:
            temp_time = get_run_time(op_id, sess)
            if max_time < temp_time:
                max_time = temp_time
                max_time_id = op_id
        if max_time_id  % 10 == 0:
            print("id: ",max_time_id,"duration: ", max_time, "ms")
        current_op_id = max_time_id
        path.append(current_op_id)
        pathtime.append(max_time)

    print(path)
    filewriter = open('result_by_graph_1_minibatch32','w')
    for x in path:
        filewriter.write(lines[x])
    filewriter.close()
    filewriter = open('time_by_graph_1_minibatch32', 'w')
    for x in pathtime:
        filewriter.write(str(x) + '\n')
    filewriter.close()
    print("total duration: ", (time.clock() - start_time), "s.")
'''
