import tensorflow as tf

# Variable
state = tf.Variable(0, name='counter')

# 常量
one = tf.constant(1)

# 加法
new_value = tf.add(state, one)

# 更新
update = tf.assign(state, new_value)

# 初始化
init = tf.global_variables_initializer()

'''
with tf.Session() as sess:
    sess.run(init)
    for i in range(3):
        sess.run(update)
        print(sess.run(state))
'''

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

output = tf.multiply(input1, input2)

with tf.Session() as sess:
    print(sess.run(output, feed_dict={input1: [7.], input2: [2.]}))