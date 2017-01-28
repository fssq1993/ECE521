import tensorflow as tf
import numpy as np

if __name__ == '__main__':
    N = 2
    X = np.array([[1, 2], [8, 9]])
    Z = np.array([[1, 1], [1, 1], [3, 3]])

    x = tf.placeholder(tf.float32, [None, N])
    z = tf.placeholder(tf.float32, [None, N])

    s_x = tf.reduce_sum(tf.square(x), 1)
    s_x = tf.reshape(s_x, [-1, 1])
    s_z = tf.reduce_sum(tf.square(z), 1)
    s_xz = tf.matmul(x, z, transpose_b=True)
    pair_dis = s_x - 2 * s_xz + s_z

    init = tf.initialize_all_variables()
    sess = tf.InteractiveSession()
    sess.run(init)

    D = sess.run(pair_dis, feed_dict={x: X, z: Z})
    print D
