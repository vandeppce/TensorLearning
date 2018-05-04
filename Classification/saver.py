import tensorflow as tf
import numpy as np

# save to file
# remember to define the same dtype and shape when restore

'''
w = tf.Variable([[1, 2, 3], [3, 4, 5]], dtype=tf.float32, name='weights')
b = tf.Variable([[1, 2, 3]], dtype=tf.float32, name='biases')

init = tf.global_variables_initializer()


# 保存时, 首先要建立一个 tf.train.Saver() 用来保存, 提取变量. 
# 再创建一个名为my_net的文件夹, 用这个 saver 来保存变量到这个目录 "my_net/save_net.ckpt".


saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    save_path = saver.save(sess, "my_net/save_net.ckpt")
    print("Save to path: ", save_path)

'''

# restore variables
# redefine the same shape and same type for variables

w = tf.Variable(np.arange(6).reshape((2, 3)), dtype=tf.float32, name='weights')
b = tf.Variable(np.arange(3).reshape((1, 3)), dtype=tf.float32, name='biases')

saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, "my_net/save_net.ckpt")
    print("weights: ", sess.run(w))
    print("biases: ", sess.run(b))
    