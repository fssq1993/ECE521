import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def pairwise_distance(X, Z, N, K):
    x = tf.placeholder(tf.float32, [None, N])
    z = tf.placeholder(tf.float32, [None, N])

    s_x = tf.reduce_sum(tf.square(x), 1)
    s_x = tf.reshape(s_x, [-1, 1])
    s_z = tf.reduce_sum(tf.square(z), 1)
    s_xz = tf.matmul(x, z, transpose_b=True)
    pair_dis = s_x - 2 * s_xz + s_z

    k_values, k_indices = tf.nn.top_k(-tf.transpose(pair_dis), k=K)

    init = tf.initialize_all_variables()
    sess = tf.InteractiveSession()
    sess.run(init)

    D, values, indices = sess.run([pair_dis, k_values, k_indices], feed_dict={x: X, z: Z})

    return D, values, indices


def responsibilities_loss(X, X_target, Z, Z_target, N, K):
    responsibilities = np.zeros([Z.shape[0], X.shape[0]])
    D, values, indices = pairwise_distance(X, Z, N, K)
    responsibilities[np.repeat(np.arange(Z.shape[0]), K), indices.ravel()] = 1.0 / K
    prediction = np.dot(responsibilities, X_target)
    loss = np.sum(np.square(prediction - Z_target)) / (2.0 * X.shape[0])

    return responsibilities, loss


if __name__ == '__main__':
    np.random.seed(521)
    Data = np.linspace(1.0, 10.0, num=100)[:, np.newaxis]
    Target = np.sin(Data) + 0.1 * np.power(Data, 2) \
             + 0.5 * np.random.randn(100, 1)
    randIdx = np.arange(100)
    np.random.shuffle(randIdx)
    trainData, trainTarget = Data[randIdx[:80]], Target[randIdx[:80]]
    validData, validTarget = Data[randIdx[80:90]], Target[randIdx[80:90]]
    testData, testTarget = Data[randIdx[90:100]], Target[randIdx[90:100]]

    # print trainData.shape, trainTarget.shape
    # print validData.shape, validTarget.shape
    # print testData.shape, testTarget.shape
    #
    # plt.plot(trainData, trainTarget, 'ro')
    # plt.ylabel('Target')
    # plt.show()

    # N = 2
    # X = np.array([[1, 2], [1,1],[100,100],[500,500]])
    # Z = np.array([[1, 1], [100, 101],[500,501]])

    N=1
    K=3

    responsibilities, loss = responsibilities_loss(trainData, trainTarget, validData, validTarget, N, K)
    y = np.dot(responsibilities, trainTarget)
    print "valid_prediction :", y
    print "valid_loss :", loss
    plt.plot(validData, y, 'ro', color='b')
    plt.plot(validData, validTarget, 'ro', color='r')
    plt.ylabel('Target')
    plt.show()
