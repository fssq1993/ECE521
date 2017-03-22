import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from utils import reduce_logsumexp, logsoftmax
from time import time
import matplotlib.mlab as mlab

from distance_functions import square_distance


def loadData():
    # data = np.load("data2D.npy").astype(np.float32)
    data = np.load("data100D.npy").astype(np.float32)
    return data

# 2.1.2
def log_prob_density(x, mu, sigma):

    D = tf.to_float(tf.rank(x))
    logp = -D*tf.log(tf.sqrt(2 * np.pi * sigma))

    dist = (-2 * tf.matmul(x, tf.transpose(mu))) + tf.reshape(tf.reduce_sum(tf.square(mu), 1), (1, -1)) + tf.reshape(
        tf.reduce_sum(tf.square(x), 1), (-1, 1))
    dist = tf.transpose(dist)
    # dist = square_distance(x, mu) # (x - mu)^2
    logp =logp - dist / (2*sigma)

    return logp

# 2.1.3
def log_cluster_probability(x, logpz, mu, sigma):

    log_px_gz = log_prob_density(x, mu, sigma) # logP(x | z)
    p_xz = logpz + tf.transpose(log_px_gz) #?????????????????????????????????transpose
    p_x = reduce_logsumexp(p_xz, 0)
    log_pz_gx = p_xz/p_x # pz * P(x | z) / P(x)

    return log_pz_gx

# 2.2
def marginal_log_likelihood(x, logpz, mu, sigma):

    pxn = reduce_logsumexp(tf.transpose(logpz) + log_prob_density(x, mu, sigma),0)
    px = tf.reduce_sum(pxn,0)

    return px

def buildGraph(k,dimension,EXP=1e-5):

    tf.set_random_seed(time())

    pz = tf.Variable(tf.zeros([1,k]))
    logpz = logsoftmax(pz) # Enforce simplex constraint over P(z)

    sigma = tf.Variable(tf.ones([k, 1])*(-3))
    expsigma = tf.exp(sigma) # Enforce sigma > 0
    print expsigma

    mu = tf.Variable(tf.random_normal([k, dimension],mean=0,stddev=0.01),dtype=tf.float32)
    x = tf.placeholder(tf.float32, [None, dimension])

    cost = -marginal_log_likelihood(x, logpz, mu, expsigma)
    iter_var = tf.Variable(0)
    optimizer = tf.train.AdamOptimizer(0.03, beta1=0.9, beta2=0.99, epsilon=1e-5)
    train = optimizer.minimize(cost, global_step=iter_var)

    return x, mu, cost,expsigma,logpz, train

def run(k, thresold,max_updates,dimension):
    data=loadData()

    randIndx = np.arange(len(data))
    np.random.shuffle(randIndx)
    Data= data[randIndx]
    trainData= Data[:1-len(data)/3]
    validData= Data[1-len(data)/3:len(data)]

    # **************************************************************************
    # If hold 1/3 of the data out for validation
    # data=trainData
    # **************************************************************************

    x,mu,cost,expsigma,logpz,train= buildGraph(k,dimension,thresold)

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    with sess.as_default():
        loss_list = []
        valid_loss_list=[]
        # best_cost = float('inf')
        last_cost = float('inf')
        # print "------------------"
        # print "P(x):  ", tf.exp(logpz).eval()
        # print "Sigma: ", expsigma.eval().reshape((1, k))
        # print "Mu:    ", mu.eval()
        # print "------------------"
        count = 0
        # try:
        while True:
        # for i in range(num_updates)
            _,iter_cost,mean,variance = sess.run([train,cost,mu,expsigma], feed_dict={x: data})
            count += 1
            print "------------------------------------------------------------"
            print "Iter: ", count, "Loss: ", iter_cost
            # print "Mean: "
            # print mean
            # print "Variance: "
            # print variance
            # print "------------------------------------------------------------"
            loss_list.append(iter_cost)
            # -----------------------Evaluation on the validation set -------------------------------------
            valid_loss = sess.run([cost], feed_dict={x: validData})
            print "The loss for the validation data:", valid_loss
            valid_loss_list.append(valid_loss)

            if count > max_updates or abs(iter_cost - last_cost) < thresold:
                print "Converged!"
                break
            else:
                last_cost = iter_cost



    # print "Best cost: %f" %valcost
    fclusters = [logpz, mu, variance]

    with sess.as_default():
        logpx = log_cluster_probability(validData, *fclusters)
        px = tf.nn.softmax(logpx)
        valid_assignments = px.eval()

        logpx = log_cluster_probability(data, *fclusters)
        px = tf.nn.softmax(logpx)
        train_assignments = px.eval()

    # return fclusters, assignments, fcosts, valcost


    # -----------------------Evaluation on the validation set -------------------------------------
    valid_loss = sess.run([cost], feed_dict={x: validData})
    print "The final loss for the validation data:",valid_loss

    # -----------------------Plot loss vs. num_updates----------------------------
    plt.figure(1)
    plt.title(
        "k= %.2f" % (k))
    plt.plot(np.arange(count), loss_list,label="The loss of MOG")
    plt.plot(np.arange(count), valid_loss_list,label="The validtion loss of MOG")

    plt.legend(loc='upper right')

    plt.ylabel('loss')
    plt.xlabel('num of updates')
    plt.show()

    # # -----------------------Plot data points -------------------------------------
    #
    # plt.figure(2)
    # for i in xrange(len(data)):
    #     # k=1
    #     # plt.scatter(data[i, 0], data[i, 1], marker='x', color='m', s=30)
    #
    #     # k=2
    #     # plt.scatter(data[i, 0], data[i, 1], marker='x', color=(train_assignments[i,0],0,train_assignments[i,1]), s=30)
    #
    #     # k=3
    #     # plt.scatter(data[i, 0], data[i, 1], marker='x', color=(train_assignments[i,0],train_assignments[i,1],train_assignments[i,2]), s=30)
    #
    #     # k=4
    #     # c=['r','b','g','m']
    #     # index=np.argmax(train_assignments[i])
    #     # print i,index
    #     # print train_assignments[i]
    #     # plt.scatter(data[i, 0], data[i, 1], marker='x', color=c[index], s=30)
    #     # plt.scatter(data[i, 0], data[i, 1], marker='x', color=(train_assignments[i,0],train_assignments[i,1],train_assignments[i,2]), s=30)
    #
    #     # k=5
    #     c = ['r', 'b', 'g', 'm','y']
    #     index = np.argmax(train_assignments[i])
    #     print i, index
    #     print train_assignments[i]
    #     plt.scatter(data[i, 0], data[i, 1], marker='x', color=c[index], s=30)
    #
    # for i in xrange(k):
    #     plt.scatter(mean[i,0], mean[i, 1], marker='x', color='k',label="Centers" if i == 0 else "", s=60)
    # plt.legend(loc='lower right')
    #
    # plt.show()
    # # -----------------------Plot data points (Gaussian)-------------------------------------
    #
    # plt.figure(3)
    # for i in xrange(k):
    #     delta = 0.025
    #     # data[:0].min
    #     x = np.arange(np.min(data[:,0])-1, np.max(data[:,0])+1, delta)
    #     y = np.arange(np.min(data[:,1])-1, np.max(data[:,1])+1, delta)
    #     X, Y = np.meshgrid(x, y)
    #     Z = mlab.bivariate_normal(X, Y, variance[i], variance[i], mean[i,0], mean[i,1])
    #     ax1 = plt.gca()
    #     # cm = plt.get_cmap('Set1', k)
    #     # cs = range(k)
    #     # ax1.scatter(data[:, 0], data[:, 1], s=25, alpha=0.6, cmap=cm, label="Data")
    #     ax1.scatter(data[:, 0], data[:, 1], marker='x',color='m',s=30,label="Data" if i == 0 else "")
    #     # ax1.scatter(mean[:, 0], mean[:, 1], marker='*', c=cs, s=250, linewidths=3, cmap=cm, label="Clusters")
    #     ax1.scatter(mean[:, 0], mean[:, 1], marker='x', color='k',s=60,label="Centers" if i == 0 else "")
    #     ax1.contour(X,Y,Z, colors='k')
    #
    # plt.legend(loc='lower right')
    #
    # plt.show()


if __name__ == '__main__':
    # Parameters
    k = 10
    std = 1
    thresold=1e-5
    max_updates=2000
    # dimentsion=2
    dimentsion=100
    np.set_printoptions(precision=2)

    run(k,thresold,max_updates,dimentsion)

