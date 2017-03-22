import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def sign(X):
    for i, num in enumerate(X):
        if num >= 0.5:
            X[i] = 1
        else:
            X[i] = 0
    return X

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

        trainTarget=(np.arange(10)==np.array(trainTarget)[:,None]).astype(np.float32)
        validTarget=(np.arange(10)==np.array(validTarget)[:,None]).astype(np.float32)
        testTarget=(np.arange(10)==np.array(testTarget)[:,None]).astype(np.float32)

        # labels = (np.arange(num_labels) == np.array(iris.target)[:, None]).astype(np.float32)

        print trainTarget.shape
        print validTarget.shape
        print testTarget.shape
        return trainData, trainTarget, validData, validTarget, testData, testTarget


def buildGraph(l_rate, lam):
    # Variable creation
    W = tf.Variable(tf.truncated_normal(shape=[784, 10], stddev=0.5), name='weights')
    b = tf.Variable(tf.zeros([10]))

    X = tf.placeholder(tf.float32, [None, 784], name='input_x')
    y_target = tf.placeholder(tf.float32, [None, 10], name='target_y')

    # Graph definition
    logits = tf.matmul(X, W) + b

    # loss function
    # cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits, y_target) + lam * tf.nn.l2_loss(W))
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, y_target) + lam * tf.nn.l2_loss(W))


    # Training mechanism
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=l_rate)
    optimizer = tf.train.AdamOptimizer(learning_rate=l_rate)
    train = optimizer.minimize(loss=cost)
    return W, b, X, y_target, logits, cost, train


def run(lr, lam, batch_size, num_epoch):
    # Loading my data
    trainData, trainTarget, validData, validTarget, testData, testTarget = loadData()

    # Build computation graph
    W, b, X, y_target, y_predicted, error, train = buildGraph(lr, lam)

    # Initialize session
    init = tf.initialize_all_variables()

    sess = tf.InteractiveSession()
    sess.run(init)

    loss_list = []
    loss_valid=[]
    loss_test=[]

    ac_list=[]
    ac_valid=[]
    ac_test=[]

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
            _, err, currentW, currentb, yhat = sess.run([train, error, W, b, y_predicted],
                                                        feed_dict={X: x, y_target: t})

            # Evaluation
            errTrain, train_predict = sess.run([error,y_predicted],
                                               feed_dict={X: trainData, y_target: trainTarget})

            errValid, valid_predict = sess.run([error, y_predicted],
                                               feed_dict={X: validData, y_target: validTarget})

            errTest, test_predict = sess.run([error, y_predicted],
                                               feed_dict={X: testData, y_target: testTarget})

            loss_list.append(errTrain)
            loss_valid.append(errValid)
            loss_test.append(errTest)

            train_accuracy=accuracy(train_predict,trainTarget)
            valid_accuracy=accuracy(valid_predict,validTarget)
            test_accuracy=accuracy(test_predict,testTarget)

            ac_list.append(train_accuracy)
            ac_valid.append(valid_accuracy)
            ac_test.append(test_accuracy)

            num_update += 1

            print("Epoch: %3d, Iter: %3d, Loss-train: %4.2f, Train Accuracy: %4.2f, Validation Loss: %4.2f Validation Accuracy: %4.2f"
                  % (i, step, errTrain,train_accuracy,errValid,valid_accuracy))

    # Final evaluation on the validation set
    print "-------------------------------------------------"
    print("Final train Cross Entropy+regularization: %.2f" % errTrain)
    errValid, valid_predict = sess.run([error, y_predicted], feed_dict={X: validData, y_target: validTarget})
    print("Final valid Cross Entropy+regularization: %.2f" % errValid)

    # Final evaluation on the test set
    errTest, test_predict = sess.run([error, y_predicted], feed_dict={X: testData, y_target: testTarget})
    print("Final testing Cross Entropy+regularization: %.2f" % errTest)

    print "-------------------------------------------------"
    print "Train Data Shape:", trainData.shape
    print "Valid Data Shape:", validData.shape
    print "Test Data Shape:", testData.shape

    valid_accuracy=accuracy(valid_predict,validTarget)
    test_accuracy=accuracy(test_predict,testTarget)

    print "-------------------------------------------------"
    print "Train Data Accuracy:", train_accuracy
    print "Valid Data Accuracy:", valid_accuracy
    print "Test Data Accuracy:", test_accuracy

    plt.figure()
    plt.title(
        "Loss learning rate: %.2f weight_decay: %.2f batch size: %.2f num_epoch: %.2f" % (lr, lam, batch_size, num_epoch))
    plt.plot(np.arange(num_update), loss_list,label="Training Set")
    plt.plot(np.arange(num_update), loss_valid,label="Validation Set")
    plt.plot(np.arange(num_update), loss_test,label="Test Set")
    plt.legend(loc='upper right')

    plt.ylabel('loss')
    plt.xlabel('num of updates')
    plt.show()

    plt.figure()
    plt.title(
        "Accuracy learning rate: %.2f weight_decay: %.2f batch size: %.2f num_epoch: %.2f" % (lr, lam, batch_size, num_epoch))
    plt.plot(np.arange(num_update), ac_list,label="Training Set")
    plt.plot(np.arange(num_update), ac_valid,label="Validation Set")
    plt.plot(np.arange(num_update), ac_test,label="Test Set")
    plt.legend(loc='lower right')
    plt.ylabel('accuracy')
    plt.xlabel('num of updates')
    plt.show()

if __name__ == '__main__':
    # Parameters
    learning_rate = 0.01
    training_epochs = 20

    batch_size = 500
    lam = 0.01

    np.set_printoptions(precision=2)

    run(learning_rate, lam, batch_size, training_epochs)

    # 1.3 Comparison with linear regression
    # run(learning_rate, 0, batch_size, training_epochs)
