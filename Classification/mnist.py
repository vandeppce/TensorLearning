import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
from Network.nn import add_layer

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre, 0), tf.argmax(v_ys, 0))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result

'''
def add_layer(inputs, in_size, out_size, activation_function=None,):
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1,)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b,)
    return outputs
'''

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

#xs = tf.placeholder(tf.float32, [None, 784])
#ys = tf.placeholder(tf.float32, [None, 10])

xs = tf.placeholder(tf.float32, [784, None])
ys = tf.placeholder(tf.float32, [10, None])

prediction = add_layer(xs, 784, 10, activation_function=tf.nn.softmax)

cross_entropy = tf.reduce_mean(-ys * tf.log(prediction)) # loss

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={xs: batch_xs.T, ys: batch_ys.T})

        if i % 50 == 0:
            print(compute_accuracy(mnist.test.images.T, mnist.test.labels.T))