import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

def add_layer(inputs, in_size, out_size, activation_function=None):
    weights = tf.Variable(tf.random_normal([out_size, in_size]))
    biases = tf.Variable(tf.zeros([out_size, 1]) + 0.1)

    wx_plus_b = tf.matmul(weights, inputs) + biases

    if activation_function is None:
        outputs = wx_plus_b
    else:
        outputs = activation_function(wx_plus_b)

    return outputs

x_data = np.linspace(-1,1,300, dtype=np.float32)[np.newaxis, :]
noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
y_data = np.square(x_data) - 0.5 + noise

xs = tf.placeholder(tf.float32, [1, None])
ys = tf.placeholder(tf.float32, [1, None])

l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)

prediction = add_layer(l1, 10, 1, activation_function=None)


loss = tf.reduce_mean(tf.square(ys - prediction))

train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for i in range(1000):
        sess.run(train, feed_dict={xs: x_data, ys: y_data})

        if i % 50 == 0:
            prediction_value = sess.run(prediction, feed_dict={xs: x_data})

'''
优化器
tf.train.GradientDescentOptimizer
tf.train.AdadeltaOptimizer
tf.train.AdagradOptimizer
tf.train.AdagradDAOptimizer
tf.train.MomentumOptimizer
tf.train.AdamOptimizer
tf.train.FtrlOptimizer
tf.train.RMSPropOptimizer
'''


