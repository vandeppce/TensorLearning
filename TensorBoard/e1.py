import tensorflow as tf
import numpy as np

#可视化网络结构

with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [1, None], name='x_in')
    ys = tf.placeholder(tf.float32, [1, None], name='y_in')

def add_layer(inputs, in_size, out_size, activation_function=None):
    with tf.name_scope('layer'):
        with tf.name_scope('weights'):
            weights = tf.Variable(tf.random_normal([out_size, in_size]), name='W')
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([out_size, 1]) + 0.1, name='b')
        with tf.name_scope('wx_plut_b'):
            wx_plus_b = tf.add(tf.matmul(weights, inputs), biases)
        if activation_function == None:
            outputs = wx_plus_b
        else:
            outputs = activation_function(wx_plus_b)
        return outputs

l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)

prediction = add_layer(l1, 10, 1)

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.square(ys - prediction))

with tf.name_scope('train'):
    train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

sess = tf.Session() # get session
# tf.train.SummaryWriter soon be deprecated, use following
writer = tf.summary.FileWriter("logs/", sess.graph)