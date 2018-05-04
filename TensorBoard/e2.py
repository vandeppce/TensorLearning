import tensorflow as tf
import numpy as np

#可视化训练过程

x_data= np.linspace(-1, 1, 300, dtype=np.float32)[np.newaxis, :]
noise=  np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
y_data= np.square(x_data) -0.5+ noise

xs = tf.placeholder(tf.float32, [1, None], name='x_in')
ys = tf.placeholder(tf.float32, [1, None], name='y_in')

def add_layer(inputs, in_size, out_size, n_layer, activation_function=None):
    layer_name = 'layer' + str(n_layer)
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            weights = tf.Variable(tf.random_normal([out_size, in_size]), name='W')
            tf.summary.histogram(layer_name + '/weights', weights)

        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([out_size, 1]) + 0.1, name='b')
            tf.summary.histogram(layer_name + '/biases', biases)

        with tf.name_scope('wx_plus_b'):
            wx_plus_b = tf.add(tf.matmul(weights, inputs), biases)

        if activation_function is None:
            outputs = wx_plus_b
        else:
            outputs = activation_function(wx_plus_b)

        tf.summary.histogram(layer_name + '/outputs', outputs)
    return outputs

l1 = add_layer(xs, 1, 10, n_layer=1, activation_function=tf.nn.relu)
prediction = add_layer(l1, 10, 1, n_layer=2)

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.square(ys - prediction))
    # tf.scalar_summary('loss',loss) # tensorflow < 0.12
    tf.summary.scalar('loss', loss)  # tensorflow >= 0.12
train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
    # merged= tf.merge_all_summaries()    # tensorflow < 0.12
    merged = tf.summary.merge_all()  # tensorflow >= 0.12

    # writer = tf.train.SummaryWriter('logs/', sess.graph)    # tensorflow < 0.12
    writer = tf.summary.FileWriter("logs/", sess.graph)  # tensorflow >=0.12

    # sess.run(tf.initialize_all_variables()) # tf.initialize_all_variables() # tf 马上就要废弃这种写法
    sess.run(tf.global_variables_initializer())  # 替换成这样就好

    for i in range(1000):
        sess.run(train, feed_dict={xs: x_data, ys: y_data})

        if i % 50 == 0:
            rs = sess.run(merged, feed_dict={xs: x_data, ys: y_data})
            writer.add_summary(rs, i)