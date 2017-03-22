# # Import MNIST data
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


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

    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # layer_2 = tf.nn.dropout(layer_2,keep_prob=0.5)

    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.relu(layer_3)
    # layer_3 = tf.nn.dropout(layer_3,keep_prob=0.5)
    layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
    layer_4 = tf.nn.relu(layer_4)

    # Output layer with linear activation
    # out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
    # out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    # out_layer = tf.matmul(layer_3, weights['out']) + biases['out']
    out_layer = tf.matmul(layer_4, weights['out']) + biases['out']
    return out_layer


def buildGraph(l_rate, lam):
    # Store layers weight & bias

    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1], mean=0.0, stddev=3.0 / (n_input + n_hidden_1))),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], mean=0.0,stddev=3.0/(n_hidden_1+n_hidden_2))),
        'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3], mean=0.0,stddev=3.0/(n_hidden_2+n_hidden_3))),
        'h4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4], mean=0.0,stddev=3.0/(n_hidden_3+n_hidden_4))),
        # 'out': tf.Variable(tf.random_normal([n_hidden_1, n_classes], mean=0.0, stddev=3.0 / (n_hidden_1 + n_classes)))
        # 'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes], mean=0.0, stddev=3.0 / (n_hidden_2 + n_classes)))
        # 'out': tf.Variable(tf.random_normal([n_hidden_3, n_classes], mean=0.0, stddev=3.0 / (n_hidden_3 + n_classes)))
        'out': tf.Variable(tf.random_normal([n_hidden_4, n_classes], mean=0.0, stddev=3.0 / (n_hidden_4 + n_classes)))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'b3': tf.Variable(tf.random_normal([n_hidden_3])),
        'b4': tf.Variable(tf.random_normal([n_hidden_4])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }
    # tf Graph input
    x = tf.placeholder("float", [None, n_input])
    y = tf.placeholder("float", [None, n_classes])

    # Construct model
    pred = multilayer_perceptron(x, weights, biases)

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y) +
                          lam * tf.nn.l2_loss(weights['out']) + lam * tf.nn.l2_loss(weights['h1'])
                          + lam*tf.nn.l2_loss(weights['h2'])
                          + lam*tf.nn.l2_loss(weights['h3'])
                          + lam*tf.nn.l2_loss(weights['h4'])
                          )
    # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=l_rate)
    train = optimizer.minimize(cost)

    # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, y_target) + lam * tf.nn.l2_loss(W))

    return x, y, pred, cost, train


def run(lr, lam, batch_size, num_epoch):
    # Loading my data
    trainData, trainTarget, validData, validTarget, testData, testTarget = loadData()

    # Build computation graph
    # W, b, X, y_target, y_predicted, error, train = buildGraph(lr, lam)
    X, y_target, y_predicted, error, train = buildGraph(lr, lam)

    # Initialize session
    init = tf.initialize_all_variables()


    sess = tf.InteractiveSession()
    sess.run(init)

    loss_list = []
    loss_valid = []
    loss_test = []

    ac_list = []
    ac_valid = []
    ac_test = []

    num_update = 0
    rnd_idx = np.arange(trainData.shape[0])
    num_train_cases = trainData.shape[0]
    if batch_size == -1:
        batch_size = num_train_cases
    num_steps = int(np.ceil(num_train_cases / batch_size))

    for i in xrange(num_epoch):
        np.random.shuffle(rnd_idx)
        inputs_train = trainData[rnd_idx]
        target_train = trainTarget[rnd_idx]

        for step in xrange(num_steps):
            # Select minibatch
            start = step * batch_size
            end = min(num_train_cases, (step + 1) * batch_size)
            x = inputs_train[start: end]
            t = target_train[start: end]

            # Training with batch
            _, err, yhat = sess.run([train, error, y_predicted], feed_dict={X: x, y_target: t})



        # Evaluation
        errTrain, train_predict = sess.run([error, y_predicted], feed_dict={X: trainData, y_target: trainTarget})

        errValid, valid_predict = sess.run([error, y_predicted], feed_dict={X: validData, y_target: validTarget})
        errTest, test_predict = sess.run([error, y_predicted], feed_dict={X: testData, y_target: testTarget})

        loss_list.append(errTrain)
        loss_valid.append(errValid)
        loss_test.append(errTest)

        train_accuracy = accuracy(train_predict, trainTarget)
        valid_accuracy = accuracy(valid_predict, validTarget)
        test_accuracy = accuracy(test_predict, testTarget)

        ac_list.append(train_accuracy)
        ac_valid.append(valid_accuracy)
        ac_test.append(test_accuracy)

        num_update += 1
        print(
            "Epoch: %3d, Loss-train: %4.2f, Train Accuracy: %4.2f, Validation Loss: %4.2f Validation Accuracy: %4.2f"
            % (i, errTrain, train_accuracy, errValid, valid_accuracy))


    # Final evaluation on the validation set
    print "-------------------------------------------------"
    print("Final train Cross Entropy+regularization: %.2f" % errTrain)
    errValid, valid_predict = sess.run([error, y_predicted], feed_dict={X: validData, y_target: validTarget})
    print("Final valid Cross Entropy+regularization: %.2f" % errValid)

    # Final evaluation on the test set
    errTest, test_predict = sess.run([error, y_predicted], feed_dict={X: testData, y_target: testTarget})
    print("Final testing Cross Entropy+regularization: %.2f" % errTest)

    # print "-------------------------------------------------"
    # print "Train Data Shape:", trainData.shape
    # print "Valid Data Shape:", validData.shape
    # print "Test Data Shape:", testData.shape

    valid_accuracy = accuracy(valid_predict, validTarget)
    test_accuracy = accuracy(test_predict, testTarget)

    print "-------------------------------------------------"
    print "Train Data Accuracy:", train_accuracy
    print "Valid Data Accuracy:", valid_accuracy
    print "Test Data Accuracy:", test_accuracy

    plt.figure()
    plt.title(
        "Loss learning rate: %.2f weight_decay: %.2f batch size: %.2f num_epoch: %.2f" % (
            lr, lam, batch_size, num_epoch))
    plt.plot(np.arange(num_update), loss_list, label="Training Set")
    plt.plot(np.arange(num_update), loss_valid, label="Validation Set")
    plt.plot(np.arange(num_update), loss_test, label="Test Set")
    plt.legend(loc='upper right')

    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()

    plt.figure()
    plt.title(
        "Accuracy learning rate: %.2f weight_decay: %.2f batch size: %.2f num_epoch: %.2f" % (
            lr, lam, batch_size, num_epoch))
    plt.plot(np.arange(num_update), ac_list, label="Training Set")
    plt.plot(np.arange(num_update), ac_valid, label="Validation Set")
    plt.plot(np.arange(num_update), ac_test, label="Test Set")
    plt.legend(loc='lower right')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.show()


# Parameters
learning_rate = 0.00677298300329
training_epochs = 40
batch_size = 1000
lam = 0.000173286949779

# Network Parameters
# n_hidden_1 = 100  # 1st layer number of features
# n_hidden_1 = 500  # 1st layer number of features
n_hidden_1 = 198  # 1st layer number of features
n_hidden_2 = 198
n_hidden_3 = 198
n_hidden_4 = 198
n_input = 784  # MNIST data input (img shape: 28*28)
n_classes = 10  # MNIST total classes (0-9 digits)

if __name__ == '__main__':
    np.set_printoptions(precision=2)

    run(learning_rate, lam, batch_size, training_epochs)