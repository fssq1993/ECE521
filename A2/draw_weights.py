# # Import MNIST data
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])


def loadData():
    with np.load("notMNIST.npz") as data:
        Data, Target = data["images"], data["labels"]
        np.random.seed(521)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data = Data[randIndx] / 255.
        Target = Target[randIndx]
        trainData, trainTarget = Data[:15000], Target[:15000]
        validData, validTarget = Data[15000:16000], Target[15000:16000]
        testData, testTarget = Data[16000:], Target[16000:]

        trainData = trainData.reshape(-1, 28 * 28)
        validData = validData.reshape(-1, 28 * 28)
        testData = testData.reshape(-1, 28 * 28)

        trainTarget = (np.arange(10) == np.array(trainTarget)[:, None]).astype(np.float32)
        validTarget = (np.arange(10) == np.array(validTarget)[:, None]).astype(np.float32)
        testTarget = (np.arange(10) == np.array(testTarget)[:, None]).astype(np.float32)

        return trainData, trainTarget, validData, validTarget, testData, testTarget


# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # layer_1 = tf.nn.dropout(layer_1, keep_prob=0.5)

    # Output layer with linear activation
    out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
    # out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer


def buildGraph(l_rate, lam):
    # Store layers weight & bias

    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1], mean=0.0, stddev=3.0 / (n_input + n_hidden_1))),
        'out': tf.Variable(tf.random_normal([n_hidden_1, n_classes], mean=0.0, stddev=3.0 / (n_hidden_1 + n_classes)))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }
    # tf Graph input
    x = tf.placeholder("float", [None, n_input])
    y = tf.placeholder("float", [None, n_classes])

    # Construct model
    pred = multilayer_perceptron(x, weights, biases)

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y) +
                          lam * tf.nn.l2_loss(weights['out']) + lam * tf.nn.l2_loss(weights['h1']))
    # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=l_rate)
    train = optimizer.minimize(cost)

    # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, y_target) + lam * tf.nn.l2_loss(W))

    return x, y, pred, cost, train,weights


def run(lr, lam, batch_size, num_epoch):
    # Loading my data
    trainData, trainTarget, validData, validTarget, testData, testTarget = loadData()

    # Build computation graph
    # W, b, X, y_target, y_predicted, error, train = buildGraph(lr, lam)
    X, y_target, y_predicted, error, train, weights = buildGraph(lr, lam)

    # Initialize session
    init = tf.initialize_all_variables()

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    sess = tf.InteractiveSession()
    saver.restore(sess, "/home/yuanyi/ECE521/A2/neural_network/model20.ckpt")
    weights = sess.run(weights)
    pix=weights['h1'].T
    print pix.shape
    pix=pix.reshape(1000,28,28)
    print pix.shape

    plt.figure(0)
    plt.clf()
    for i in xrange(1000):
        fig=plt.subplot(32,32,i+1)
        plt.imshow(pix[i], cmap=plt.cm.gray)
        plt.axis('off')

    plt.draw()
    # plt.show()
    plt.savefig('20.png')


    
    # plt.figure(0)
    # plt.clf()
    # for i in xrange(means.shape[3]):
    #     plt.subplot(2, 4, i+1)
    #     #plt.imshow(means[:, i].reshape(5, 5), cmap=plt.cm.gray)
    #     plt.imshow(means[:, :,0,i], cmap=plt.cm.gray)
    # plt.draw()
    # raw_input('Press Enter.')

# Parameters
learning_rate = 0.001
training_epochs = 45
batch_size = 1000
lam = 3e-4

# Network Parameters
# n_hidden_1 = 100  # 1st layer number of features
# n_hidden_1 = 500  # 1st layer number of features
n_hidden_1 = 1000  # 1st layer number of features
# n_hidden_2 = 500
n_input = 784  # MNIST data input (img shape: 28*28)
n_classes = 10  # MNIST total classes (0-9 digits)

if __name__ == '__main__':
    np.set_printoptions(precision=2)

    run(learning_rate, lam, batch_size, training_epochs)
