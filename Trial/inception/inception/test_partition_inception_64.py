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
import sys
sys.path.append('/home/lyt/inception_without_summary/inception')
import numpy as np
import tensorflow as tf

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
  total_loss = tf.add_n(losses + regularization_losses, name='total_loss')

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
    total_loss = tf.identity(total_loss)
  return total_loss


np.random.seed(1)
ima = np.random.ranf(size=[64, 224, 224, 3])
ima = ima.astype('float32')
images = tf.Variable(ima, name='images')
#images = tf.get_variable('images', [16,224,224,3], initializer=tf.random_normal_initializer(0,0.3))
labels = np.random.randint(1000, size=64, dtype = 'int32')

total_loss = _tower_loss(images = images, labels = labels, num_classes = 1000 + 1, scope = 'tower', reuse_variables=None)
global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
num_batches_per_epoch = 1000
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
apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
variable_averages = tf.train.ExponentialMovingAverage(inception.MOVING_AVERAGE_DECAY, global_step)

variables_to_average = (tf.trainable_variables() +
                        tf.moving_average_variables())
variables_averages_op = variable_averages.apply(variables_to_average)
batchnorm_updates_op = tf.group(*batchnorm_updates)
train_op = tf.group(apply_gradient_op, variables_averages_op,
                    batchnorm_updates_op, name="train_op")
init = tf.global_variables_initializer()


subg_fd = open('/home/lyt/static_analysis/result/graph_1.txt', "r")
line = subg_fd.readline()
    # dic is a dict like:
    # {1:{2,3,4},5:{6,7,8}}
    #which means op1's  predecessor is {2,3,4}
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
print('read dic')

path = list()
filereader = open('result_by_graph_1_minibatch32', 'r')
line = filereader.readline()
numnum = 0
while line:
    #print(line.split("[")[1])
    if len(line)>1:
        split_line = line.split("[")
        split_line = split_line[1]
        cur = split_line.split("]")[0]
        path.append(int(cur))
    line = filereader.readline()
filereader.close()

print('read path')
pathtime = list()
filereader = open('time_by_graph_1_minibatch32', 'r')
line = filereader.readline()
while line:
    pathtime.append(float(line))
    line = filereader.readline()
filereader.close()
print('read time')
""" LYT: partition policy """
pathset = set(path)

#set the path from top to bottom
path = path[::-1]
pathtime = pathtime[::-1]
print('pathlen: ',len(path), '                    pathtimelen: ', len(pathtime))
subg_fd = open('/home/lyt/static_analysis/result/graph_1.txt', "r")
op_information = subg_fd.readlines()
subg_fd.close()
print('read op_information')


node_num = len(op_information)
uf = list()
def myfind(uf, a):
    if uf[a] != a:
        uf[a] = myfind(uf, uf[a])
    return uf[a]
def myunion(uf, a, b):
    fathera = myfind(uf, a)
    fatherb = myfind(uf, b)
    if fathera != fatherb:
        uf[fatherb] = fathera 
#init unionfind
for i in range(node_num):
  uf.append(i)

for i in range(node_num):
  curline = op_information[i]
  curdata = curline.split('||')[1].strip(' ')
  if(curdata != '\n'):
    curdata = curdata.split(' ')
    for pair in curdata[:-1]:
        dtype = pair.split(',')[0].split('{')[1]
        successor = pair.split(',')[1].split('}')[0]
        if int(dtype) >= 100:
            myunion(uf, i,int(successor))
allroot = set()
for i in range(len(uf)):
    allroot.add(myfind(uf, i))
pathinuf = set() #这个集合记录了critical path中的节点所在的uf根节点 
for node in path:
  pathinuf.add(myfind(uf, node))

#读取init图记录为map
graph_0 = open('/home/lyt/static_analysis/result/graph_0.txt', "r")
line = graph_0.readline()
graph_0_map = dict()
#{name: id,}
while line:
    line = line.split('[')
    cur_id = int(line[1].split(']')[0])
    cur_name = line[3].split(']')[0]
    graph_0_map[cur_name] = cur_id
    line = graph_0.readline()
graph_0.close()
#读取init图记录为并查集
graph_0 = open('/home/lyt/static_analysis/result/graph_0.txt', "r")
graph_0_information = graph_0.readlines()
graph_0.close()
node_num = len(graph_0_information)
graph_0_uf = list()
for i in range(node_num):
    graph_0_uf.append(i)

for i in range(node_num):
  curline = graph_0_information[i]
  curdata = curline.split('||')[1].strip(' ')
  if(curdata != '\n'):
    curdata = curdata.split(' ')
    for pair in curdata[:-1]:
        dtype = pair.split(',')[0].split('{')[1]
        successor = pair.split(',')[1].split('}')[0]
        if int(dtype) >= 100:
            myunion(graph_0_uf, i,int(successor))
replacement_num = 0
graph_0_replacement_num = 0
q = list()
reducetime = 0.0

logWriter = open('log64.txt', 'a')
for curdeep in range(1, len(path)-1):
    print("cur deep: ", curdeep)
    replacement_list = set()
    #q = Queue()
    q.append(path[curdeep])
    while len(q) > 0:
        cur = q[0]
        q.remove(q[0])
        if cur in dic.keys():
            for predecessor in dic[cur]:
                if predecessor not in pathset:
                    q.append(predecessor)
                    replacement_list.add(predecessor)
    gpu_id = 1
    #检查在应该从replacement_list中删除的节点
    replacement_remove = set()
    for node in replacement_list:
      if myfind(uf, node) in pathinuf:
        replacement_remove.add(node)
    
    replacement_list = replacement_list - replacement_remove
    
    replacement_add = set()
    #检查应该向replacement_list中增加的节点
    for node in replacement_list:
      for x in range(len(uf)):
        if myfind(uf, x) == myfind(uf, node):
          replacement_add.add(x)
    
    replacement_list = replacement_list | replacement_add
    print('placement_list_len: ', len(replacement_list))

    if len(replacement_list) > 0:
        fileWriter = open('/home/lyt/static_analysis/replacement/replace_train.txt', 'a')
        current_write = 0
        graph_0_replacement_list = set()
        for x in replacement_list:
            data = op_information[x].split('[')[3]
            data = data.split(']')[0]
            #添加graph_0需要replace的节点
            if str(data) in graph_0_map.keys():
                graph_0_replacement_list.add(graph_0_map[str(data)])
            
            fileWriter.write(str(data) + ' ' + str(gpu_id) + '\n')
            replacement_num = replacement_num + 1
            current_write = current_write + 1
        fileWriter.close()

        graph_0_replacement_add = set()
        for node in graph_0_replacement_list:
            for x in range(len(graph_0_uf)):
                if myfind(graph_0_uf, x) == myfind(graph_0_uf, node):
                    graph_0_replacement_add.add(x)
        graph_0_replacement_list = graph_0_replacement_list | graph_0_replacement_add
        fileWriter = open('/home/lyt/static_analysis/replacement/replace_init.txt', 'a')
        graph_0_current_write = 0
        for x in graph_0_replacement_list:
            data = graph_0_information[x].split('[')[3]
            data = data.split(']')[0]
            fileWriter.write(str(data) + ' ' + str(gpu_id) + '\n')
            graph_0_current_write += 1
            graph_0_replacement_num += 1
        fileWriter.close()

        #如果当前replace无法运行，说明这些节点无法被分割出去
        '''
        try:
            sess.run(train_op)
        except FailedPreconditionError as e:
            print('we can\'t partition these op')
            with open('/home/lyt/static_analysis/replacement/replace.txt', 'r') as old_file:
              with open('/home/lyt/static_analysis/replacement/replace.txt', 'r+') as new_file:
                current_line = 0
                while current_line < (replacement_num - current_write):
                  old_file.readline()
                  current_line = current_line + 1
                replacement_num = replacement_num - current_write 
                seek_point = old_file.tell()
                new_file.seek(seek_point, 0)
                new_file.truncate()
            
            for key in dic:      
                dic[key] = dic[key] - set(replacement_list)
            continue
        '''       
        op_name = op_information[path[curdeep]].split('[')[3].split(']')[0]

        sess = tf.Session()
        #增加替换replace.txt的代码
        os.rename('/home/lyt/static_analysis/replacement/replace_init.txt', '/home/lyt/static_analysis/replacement/replace.txt')
        sess.run(init)
        #增加替换replace.txt的代码
        os.rename('/home/lyt/static_analysis/replacement/replace.txt', '/home/lyt/static_analysis/replacement/replace_init.txt')
        os.rename('/home/lyt/static_analysis/replacement/replace_train.txt', '/home/lyt/static_analysis/replacement/replace.txt')
        cur_op = tf.get_default_graph().get_operation_by_name(op_name)
        
        print('cur_op')
        for i in range(20):
            sess.run(cur_op)
        start = time.clock()
        for i in range(30):
            sess.run(cur_op)
        dur = (time.clock() - start) * 1000 / 30
        os.rename('/home/lyt/static_analysis/replacement/replace.txt', '/home/lyt/static_analysis/replacement/replace_train.txt')
        #we can not partition this part because of the bad result
        if dur >= (pathtime[curdeep]- reducetime):
            gpu_id = 0
            print('                               no reduce time')
            logWriter.write('curdeep:' + str(curdeep) + '    curtime: ' + str(dur) + 'ms          pathtime: ' + str(pathtime[curdeep]) +  'ms' + '\n')
            print('curtime: ', dur, 'ms', '           pathtime: ', pathtime[curdeep], 'ms')
            with open('/home/lyt/static_analysis/replacement/replace_train.txt', 'r') as old_file:
              with open('/home/lyt/static_analysis/replacement/replace_train.txt', 'r+') as new_file:
                current_line = 0
                while current_line < (replacement_num - current_write):
                  old_file.readline()
                  current_line = current_line + 1
                replacement_num = replacement_num - current_write 
                seek_point = old_file.tell()
                new_file.seek(seek_point, 0)
                new_file.truncate()

            with open('/home/lyt/static_analysis/replacement/replace_init.txt', 'r') as old_file:
              with open('/home/lyt/static_analysis/replacement/replace_init.txt', 'r+') as new_file:
                current_line = 0
                while current_line < (graph_0_replacement_num - graph_0_current_write):
                  old_file.readline()
                  current_line = current_line + 1
                graph_0_replacement_num = graph_0_replacement_num - graph_0_current_write
                seek_point = old_file.tell()
                new_file.seek(seek_point, 0)
                new_file.truncate()          

        # we can partition it so we need to change the topo
        else:
          reducetime = pathtime[curdeep] - dur 
          print('opname: ', op_name, 'runningtime: ', dur, 'ms.', 'reducetime:        ', reducetime, 'ms')
          for key in dic:      
            dic[key] = dic[key] - set(replacement_list)

logWriter.close()

sess = tf.Session()

os.rename('/home/lyt/static_analysis/replacement/replace_init.txt', '/home/lyt/static_analysis/replacement/replace.txt')
sess.run(init)
os.rename('/home/lyt/static_analysis/replacement/replace.txt', '/home/lyt/static_analysis/replacement/replace_init.txt')
os.rename('/home/lyt/static_analysis/replacement/replace_train.txt', '/home/lyt/static_analysis/replacement/replace.txt')
for i in range(20):
    sess.run(train_op)
start = time.clock()
for i in range(30):
    sess.run(train_op)
print((time.clock() - start) * 1000 / 30)
os.rename('/home/lyt/static_analysis/replacement/replace.txt', '/home/lyt/static_analysis/replacement/replace_train.txt')
print('end!')
