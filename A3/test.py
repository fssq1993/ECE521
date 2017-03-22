import numpy as np
import tensorflow as tf


def buildGraph(k,std):
    # Variable creation
    X = tf.placeholder(tf.float32, [None, 2], name='input_x')
    u = tf.placeholder(tf.float32,[k,2],name='center')
    # u = tf.Variable(tf.truncated_normal(shape=[k,2],stddev=std),name='central point')
    dis=-2*tf.matmul(X,tf.transpose(u))+tf.reshape(tf.reduce_sum(tf.square(u),1),(1,-1))\
                                        +tf.reshape(tf.reduce_sum(tf.square(X),1),(-1,1))
    assignment=tf.argmin(dis,dimension=1);

    sparse_labels = tf.reshape(assignment, [-1, 1])
    derived_size = tf.shape(assignment)[0]
    indices = tf.reshape(tf.range(0, derived_size, 1), [-1, 1])
    concated = tf.concat(1, [indices, tf.cast(sparse_labels, tf.int32)])
    outshape = tf.pack([derived_size, k])
    assignment = tf.sparse_to_dense(concated, outshape, 1.0, 0.0)

    return X,u,assignment


def run(k,std):
    a = np.array([[1, 2], [3, 4], [4, 5], [5, 3], [4, 6], [8, 3], [2, 3], [4, 5], [3, 5], [0, 0]])
    center = np.array([[0, 0], [1, 1],[2,2]])

    # Initialize session
    X,u,assignment=buildGraph(k,std)

    init = tf.initialize_all_variables()
    sess = tf.InteractiveSession()
    sess.run(init)

    r_X,r_assignment= sess.run([X,assignment],feed_dict={X: a, u: center})

    # r_assignment = (np.arange(k) == r_assignment[:, None]).astype(np.float32)
    print r_assignment.shape


if __name__ == '__main__':
    # Parameters
    k = 3
    std= 0.5
    data = np.load("data2D.npy")
    np.set_printoptions(precision=2)

    run(k,std)
