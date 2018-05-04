import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(1)

mnist = input_data.read_data_sets('../CNN/MNIST_data/', one_hot=True)

lr = 0.001                                # learning rate
trainging_iters = 100000                  # train step 上限
batch_size = 128

n_inputs = 28                             # mnist data input (img shape: 28*28)
n_steps = 28                              # time steps
n_hidden_units = 128                      # neurons in hidden layer
n_classes = 10                            # mnist classed (0-9 digits)

x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])     # 按行读取
y = tf.placeholder(tf.float32, [None, n_classes])

# 对 weights biases 初始值的定义
# rnn相当于一个cell, 在这个cell之前和之后各加一层隐含层，也就是说原始数据经过隐含层处理输入rnn cell, rnn处理完后再经过一层隐含层，输出结果

weights = {
    # shape (28, 128)
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
    # shape (128, 10)
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}
biases = {
    # shape (128, )
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
    # shape (10, )
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
}

# RNN主体结构, 共包含input_layer, cell, output_layer三个部分

def RNN(X, weights, biases):
    # hidden layer for input to cell
    ################################

    # transpose the inputs shape
    # X ==> (128 batch * 28 steps, 28 inputs)
    X = tf.reshape(X, [-1, n_inputs])

    # into hidden
    # X_in = (128 batch * 28 steps, 28 inputs)
    X_in = tf.matmul(X, weights['in']) + biases['in']

    # X_in ==> (128 batch, 28 steps, 128 hidden)
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])

    # cell
    ################################

    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    # lstm cell is divided into two parts (c_state, h_state)
    init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)

    outputs, states = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=init_state, time_major=False)

    # hidden layer for output as the final results
    ################################

    results = tf.matmul(states[1], weights['out']) + biases['out']

    return results

pred = RNN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    step = 0

    while step * batch_size < trainging_iters:
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])

        sess.run(train_op, feed_dict={x: batch_xs, y: batch_ys})
        if step % 20 == 0:
            print(sess.run(accuracy, feed_dict={
            x: batch_xs,
            y: batch_ys,
        }))
        step += 1