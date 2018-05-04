import tensorflow as tf
import numpy as np

#训练数据
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

#w， b
weights = tf.Variable(tf.random_uniform((1, 1), -1.0, 1.0, dtype='float32'))
biases = tf.Variable(tf.zeros((1, 1)))

#预测值
y = weights * x_data + biases

#损失函数
loss = tf.reduce_mean(tf.square(y-y_data))

#梯度下降
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for step in range(200):
        sess.run(train)
        if step % 20 == 0:
            print(step, sess.run(weights), sess.run(biases))

