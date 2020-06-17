import tensorflow as tf
import numpy as np
import time
import os

input_data = np.random.random((32,64,64,3))

input = tf.placeholder(dtype=tf.float32, shape=[32,64,64,3])
temp_1 = tf.layers.dense(input, 3)
#branch 1
filter1 = np.random.random((256,256,3,3))
conv1 = temp_1
conv1 = tf.concat([conv1, conv1], 1)
conv1 = tf.concat([conv1, conv1], 1)
conv1 = tf.concat([conv1, conv1], 2)
conv1 = tf.concat([conv1, conv1], 2)
for i in range(100):
    conv1 = tf.nn.conv2d(conv1, filter1, [1,2,2,1], 'SAME')
    conv1 = tf.layers.dense(conv1, 3)
conv1 = tf.nn.max_pool(conv1, [1,4,4,1], [1,4,4,1], 'SAME')
#with tf.device('/gpu:1'):
#branch 2
filter2 = np.random.random((256,256,3,3))
conv2 = temp_1
conv2 = tf.concat([conv2, conv2], 1)
conv2 = tf.concat([conv2, conv2], 1)
conv2 = tf.concat([conv2, conv2], 2)
conv2 = tf.concat([conv2, conv2], 2)
for i in range(100):
    conv2=tf.nn.conv2d(conv2, filter2, [1,2,2,1], 'SAME')
    conv2=tf.layers.dense(conv2, 3)
conv2 = tf.nn.max_pool(conv2, [1,4,4,1], [1,4,4,1], 'SAME')

res = tf.concat([conv1, conv2], -1)

sess = tf.Session()
#os.rename('/home/v-yali6/workspace_yal/rldp/Trial/inception/replace_init.txt', '/home/v-yali6/workspace_yal/rldp/Trial/inception/replace.txt')
sess.run(tf.global_variables_initializer())
#os.rename('/home/v-yali6/workspace_yal/rldp/Trial/inception/replace.txt', '/home/v-yali6/workspace_yal/rldp/Trial/inception/replace_init.txt')

#os.rename('/home/v-yali6/workspace_yal/rldp/Trial/inception/replace_train.txt', '/home/v-yali6/workspace_yal/rldp/Trial/inception/replace.txt')
sess.run(res, feed_dict={input: input_data})
time_file = open('running_time.txt', 'w')
#Train_writer = tf.summary.FileWriter('/tmp/mytestnet', sess.graph)
start = time.time()
for i in range(5):
    sess.run(res, feed_dict={input: input_data})
time_file.write(str((time.time() - start) / 5))
#print(end - start)
#os.rename('/home/v-yali6/workspace_yal/rldp/Trial/inception/replace.txt', '/home/v-yali6/workspace_yal/rldp/Trial/inception/replace_train.txt')
sess.close()

