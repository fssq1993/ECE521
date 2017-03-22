import tensorflow as tf
import numpy as np
import numpy
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

def sign(X):
    for i,num in enumerate(X):
        if num>=0.5:
            X[i]=1
        else:
            X[i]=0
    return X


def loadData():
    # Loading my data

    with np.load("tinymnist.npz") as data:
        trainData, trainTarget = data["x"], data["y"]
        validData, validTarget = data["x_valid"], data["y_valid"]
        testData, testTarget = data["x_test"], data["y_test"]

    return trainData, trainTarget, validData, validTarget, testData, testTarget


def buildGraph(l_rate, Lambda):
    # Variable creation
    W = tf.Variable(tf.truncated_normal(shape=[64, 1], stddev=0.5), name='weights')
    b = tf.Variable(0.0, name='biases')
    X = tf.placeholder(tf.float32, [None, 64], name='input_x')
    y_target = tf.placeholder(tf.float32, [None, 1], name='target_y')

    # Graph definition
    y_predicted = tf.matmul(X, W) + b

    # Error definition
    meanSquaredError = tf.reduce_mean(tf.reduce_mean(tf.square(y_predicted - y_target),
                                                     reduction_indices=1,
                                                     name='squared_error'),
                                      name='mean_squared_error') + tf.reduce_sum(tf.square(W)) * Lambda / 2

    # Training mechanism
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=l_rate)
    train = optimizer.minimize(loss=meanSquaredError)
    return W, b, X, y_target, y_predicted, meanSquaredError, train


def run(lr, lam, batch_size, num_epoch):
    # Build computation graph
    W, b, X, y_target, y_predicted, meanSquaredError, train = buildGraph(lr, lam)

    # Loading my data
    trainData, trainTarget, validData, validTarget, testData, testTarget = loadData()

    # Initialize session
    init = tf.initialize_all_variables()

    sess = tf.InteractiveSession()
    sess.run(init)

    initialW = sess.run(W)
    initialb = sess.run(b)

    # print("Initial weights: %s, initial bias: %.2f" % (initialW, initialb))
    # Training model
    wList = []
    loss_list = []
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
            # Select random minibatch
            # indices = np.random.choice(trainData.shape[0], batch_size)
            # X_batch, y_batch = trainData[indices], trainTarget[indices]

            # Select minibatch
            start = step * batch_size
            end = min(num_train_cases, (step + 1) * batch_size)
            x = inputs_train[start: end]
            t = target_train[start: end]

            # _, err, currentW, currentb, yhat = sess.run([train, meanSquaredError, W, b, y_predicted],
            #                                             feed_dict={X: X_batch, y_target: y_batch})

            _, err, currentW, currentb, yhat = sess.run([train, meanSquaredError, W, b, y_predicted],
                                                        feed_dict={X: x, y_target: t})

            errTrain = sess.run(meanSquaredError, feed_dict={X: trainData, y_target: trainTarget})

            # wList.append(currentW)
            # loss_list.append(err)
            loss_list.append(errTrain)
            num_update += 1

            # print("Epoch: %3d, Iter: %3d, MSE-train: %4.2f, bias: %.2f" % (i, step, err, currentb))
            print("Epoch: %3d, Iter: %3d, MSE-train: %4.2f, bias: %.2f" % (i, step, errTrain, currentb))

            # if not (step % 50) or step < 10:
            #     print("Iter: %3d, MSE-train: %4.2f, bias: %.2f" % (step, err, currentb))
            #     print("Iter: %3d, MS E-train: %4.2f, weights: %s, bias: %.2f" % (step, err, currentW.T, currentb))

    # Test on the validation set
    errValid, valid_predict = sess.run([meanSquaredError,y_predicted], feed_dict={X: validData, y_target: validTarget})
    print("Final valid MSE: %.2f" % errValid)

    # Testing model
    errTest, test_predict = sess.run([meanSquaredError,y_predicted], feed_dict={X: testData, y_target: testTarget})
    print("Final testing MSE: %.2f" % errTest)

    # print loss_list
    # print len(loss_list)
    # print num_update

    print "Train Data Shape:", trainData.shape
    print "Valid Data Shape:", validData.shape
    print "Test Data Shape:", testData.shape

    valid_predict=sign(valid_predict)
    test_predict=sign(test_predict)

    valid_accuracy=accuracy_score(validTarget, valid_predict)
    test_accuracy=accuracy_score(testTarget,test_predict)

    print "Valid Data Accuracy:", valid_accuracy
    print "Test Data Accuracy:", test_accuracy


    plt.figure()
    plt.title("learning rate: %.2f weight_decay: %.2f batch size: %.2f num_epoch: %.2f" % (lr, lam, batch_size,num_epoch))
    plt.plot(np.arange(num_update), loss_list)
    plt.ylabel('loss')
    plt.xlabel('num of updates')
    plt.show()

    #2.2.3
    plt.figure()





if __name__ == '__main__':
    # trainData, trainTarget, validData, validTarget, testData, testTarget = loadData()
    # print trainData.shape
    # print trainTarget.shape
    np.set_printoptions(precision=2)
    # run(lr=0.01, lam=1, batch_size=50, num_epoch=20)

    #2.2.1 lr=0.01 is the best parameter for the convergence
    # run(lr=0.01, lam=1, batch_size=50, num_epoch=20)

    #2.2.2 change the batch_size and tune the learning rate separately for each mini-batch size
    # run(lr=0.01, lam=1, batch_size=10, num_epoch=5)
    # run(lr=0.01, lam=1, batch_size=50, num_epoch=20)
    # run(lr=0.1, lam=1, batch_size=100, num_epoch=5)
    # run(lr=0.1, lam=1, batch_size=700, num_epoch=30)

    #2.2.3 run SGD with mini-batch size B=50 and use validation set to choose the best weight decay coefficient
    # that gives the best classification accuracy on the test set from
    # Lambda=[0, 0.0001, 0.001, 0.01, 0.1, 1]
    # run(lr=0.01, lam=0., batch_size=50, num_epoch=300)
    # run(lr=0.01, lam=0.0001, batch_size=50, num_epoch=300)
    # run(lr=0.01, lam=0.001, batch_size=50, num_epoch=300)
    # run(lr=0.01, lam=0.01, batch_size=50, num_epoch=800)
    # run(lr=0.01, lam=0.1, batch_size=50, num_epoch=400)
    # run(lr=0.01, lam=1, batch_size=50, num_epoch=500)
