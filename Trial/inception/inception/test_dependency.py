
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
tf.app.flags.DEFINE_float('initial_learning_rate', 0.1,
                          """Initial learning rate.""")
tf.app.flags.DEFINE_float('num_epochs_per_decay', 30.0,
                          """Epochs after which learning rate decays.""")
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.16,
                          """Learning rate decay factor.""")
RMSPROP_DECAY = 0.9                # Decay term for RMSProp.
RMSPROP_MOMENTUM = 0.9             # Momentum in RMSProp.
RMSPROP_EPSILON = 1.0

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
  return total_loss

np.random.seed(1)
ima = np.random.ranf(size=[32, 299, 299, 3])
ima = ima.astype('float32')
images = tf.Variable(ima, name='images', trainable=False)
#images = tf.get_variable('images', [16,224,224,3], initializer=tf.random_normal_initializer(0,0.3))
labels = np.random.randint(1000, size=32, dtype = 'int32')

total_loss = _tower_loss(images = images, labels = labels, num_classes = 1000 + 1, scope = 'tower', reuse_variables=None)
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


path = list()
filereader = open('result_by_graph_1_minibatch32_299', 'r')
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
pathset = set(path)


def get_subgraph(sink):
    queue = list()
    result = set()
    queue.append(sink)
    while len(queue) > 0:
        cur = queue[0]
        result.add(cur)
        queue.remove(queue[0])
        if cur in dic.keys():
            for predecessor in dic[cur]:
                if predecessor not in pathset:
                    queue.append(predecessor)
    return result

allpre_10551 = dict()
for pre in dic[10551]:
    allpre_10551[pre] = get_subgraph(pre)

print(allpre_10551)

subg_fd = open('/home/lyt/static_analysis/result/graph_1.txt', "r")
op_information = subg_fd.readlines()
subg_fd.close()

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

for key in allpre_10551:
    replacement_list = allpre_10551[key]

    replacement_remove = set()
    for node in replacement_list:
        if myfind(uf, node) in pathinuf:
            replacement_remove.add(node)
    replacement_list = replacement_list - replacement_remove
    replacement_add = set()
    for node in replacement_list:
        for x in range(len(uf)):
            if myfind(uf, x) == myfind(uf, node):
                replacement_add.add(x)
    replacement_list = replacement_list | replacement_add
    gpu_id = 1
    fileWriter = open('/home/lyt/static_analysis/replacement/replace_train.txt', 'a')
    graph_0_replacement_list = set()
    for x in replacement_list:
        data = op_information[x].split('[')[3]
        data = data.split(']')[0]
        #添加graph_0需要replace的节点
        if str(data) in graph_0_map.keys():
            graph_0_replacement_list.add(graph_0_map[str(data)])
        fileWriter.write(str(data) + ' ' + str(gpu_id) + '\n')

    fileWriter.close()

    graph_0_replacement_add = set()
    for node in graph_0_replacement_list:
        for x in range(len(graph_0_uf)):
            if myfind(graph_0_uf, x) == myfind(graph_0_uf, node):
                graph_0_replacement_add.add(x)
    graph_0_replacement_list = graph_0_replacement_list | graph_0_replacement_add
    fileWriter = open('/home/lyt/static_analysis/replacement/replace_init.txt', 'a')
    for x in graph_0_replacement_list:
        data = graph_0_information[x].split('[')[3]
        data = data.split(']')[0]
        fileWriter.write(str(data) + ' ' + str(gpu_id) + '\n')
    fileWriter.close()

    """ training """
    init = tf.global_variables_initializer()
    opname = 'tower/logits/predictions'
    op = tf.get_default_graph().get_operation_by_name(opname)
    sess = tf.Session()
    os.rename('/home/lyt/static_analysis/replacement/replace_init.txt', '/home/lyt/static_analysis/replacement/replace.txt')
    sess.run(init)
    os.rename('/home/lyt/static_analysis/replacement/replace.txt', '/home/lyt/static_analysis/replacement/replace_init.txt')

    os.rename('/home/lyt/static_analysis/replacement/replace_train.txt', '/home/lyt/static_analysis/replacement/replace.txt')

    sess.run(train_op)
    for i in range(50):
        sess.run(train_op)

    start = time.clock()
    for i in range(500):
        sess.run(train_op)
    dur = (time.clock() - start) * 1000 / 500
    print("key: ", key, "    time: ", dur)
    logWriter = open('time_log_1.txt', 'a')
    logWriter.write('key: ' + str(key) + '    time: ' + str(dur) + '\n')
    logWriter.close()
    os.rename('/home/lyt/static_analysis/replacement/replace.txt', '/home/lyt/static_analysis/replacement/replace_train.txt')

    with open('/home/lyt/static_analysis/replacement/replace_train.txt', 'r') as old_file:
        with open('/home/lyt/static_analysis/replacement/replace_train.txt', 'r+') as new_file:
            current_line = 0
            while current_line < 158:
                old_file.readline()
                current_line = current_line + 1
            seek_point = old_file.tell()
            new_file.seek(seek_point, 0)
            new_file.truncate()
    
    with open('/home/lyt/static_analysis/replacement/replace_init.txt', 'r') as old_file:
        with open('/home/lyt/static_analysis/replacement/replace_init.txt', 'r+') as new_file:
            current_line = 0
            while current_line < 139:
                old_file.readline()
                current_line = current_line + 1
            seek_point = old_file.tell()
            new_file.seek(seek_point, 0)
            new_file.truncate()

