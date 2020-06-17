import tensorflow as tf
import numpy as np
import time

input_data = np.random.random((32,64,64,3))

input = tf.placeholder(dtype=tf.float32, shape=[32,64,64,3])
temp_1 = tf.layers.dense(input, 3)
with tf.device('/gpu:0'):
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

sess.run(tf.global_variables_initializer())
sess.run(res, feed_dict={input: input_data})
Train_writer = tf.summary.FileWriter('/tmp/mytestnet', sess.graph)
start = time.time()
sess.run(res, feed_dict={input: input_data})
end = time.time()
print(end - start)
