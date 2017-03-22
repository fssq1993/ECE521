import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf




def loadData():
    # data = np.load("data2D.npy")
    data = np.load("data100D.npy")
    return data


def buildGraph(LEARNINGRATE, k, std,EXP=1e-6):
    # Variable creation
    data=loadData()
    # X = tf.placeholder(tf.float32, [None, 2], name='input_x')
    X = tf.placeholder(tf.float32, [None, 100], name='input_x')
    # u = tf.Variable(tf.random_normal(shape=[k, 2],stddev=std), name='central_point')
    u = tf.Variable(tf.random_normal(shape=[k, 100],stddev=std), name='central_point')

    #Assign
    dis = (-2 * tf.matmul(X, tf.transpose(u))) + tf.reshape(tf.reduce_sum(tf.square(u), 1), (1, -1)) + tf.reshape(
        tf.reduce_sum(tf.square(X), 1), (-1, 1))

    assignment = tf.argmin(dis, dimension=1);
    assignment = tf.reshape(assignment, (-1, 1));
    # assignment = (np.arange(k) == assignment[:, None]).astype(np.float32)


    # One Hot Encoder
    sparse_labels = tf.reshape(assignment, [-1, 1])
    derived_size = tf.shape(assignment)[0]
    indices = tf.reshape(tf.range(0, derived_size, 1), [-1, 1])
    concated = tf.concat(1, [indices, tf.cast(sparse_labels, tf.int32)])
    outshape = tf.pack([derived_size, k])
    assignment = tf.sparse_to_dense(concated, outshape, 1.0, 0.0)

    u_X = tf.matmul(assignment, u)

    # loss function
    cost = tf.reduce_sum(tf.square(X - u_X))

    # min_dis=tf.reduce_min(dis,1)
    # cost = tf.reduce_sum(min_dis)
    iter_var = tf.Variable(0)


    # Training mechanism
    optimizer = tf.train.AdamOptimizer(LEARNINGRATE, beta1=0.9, beta2=0.99, epsilon=1e-5)

    train = optimizer.minimize(loss=cost,global_step=iter_var)

    # return X, u,assignment, cost, train,dis,min_dis
    return X, u,assignment, cost, train,dis

def run(lr, k, std, num_updates):
    # Loading my data
    data = loadData()
    randIndx = np.arange(len(data))
    np.random.shuffle(randIndx)
    Data= data[randIndx]
    trainData= Data[:1-len(data)/3]
    validData= Data[1-len(data)/3:len(data)]

    # **************************************************************************
    # If hold 1/3 of the data out for validation

    # data=trainData

    # **************************************************************************


    # Build computation graph
    X, u, assignment, cost, train,dis = buildGraph(lr, k, std)

    # Initialize session
    init = tf.initialize_all_variables()
    sess = tf.InteractiveSession()
    sess.run(init)

    loss_list=[]
    with sess.as_default():
        for i in xrange(num_updates):
            _,loss,center,assign = sess.run([train,cost,u,assignment], feed_dict={X: data})
            print("Iter: %3d, Loss: %4.2f" % (i, loss))
            loss_list.append(loss)


    print "Percentage of the data points assigned to k clusters",np.sum(assign,axis=0)/len(data)
    # print center

    # -----------------------Plot loss vs. num_updates----------------------------
    plt.figure(1)
    plt.title(
        "Loss learning rate: %.2f k: %.2f num_update: %.2f" % (lr, k,num_updates))
    plt.plot(np.arange(num_updates), loss_list,label="Training Set")

    plt.legend(loc='upper right')

    plt.ylabel('loss')
    plt.xlabel('num of updates')
    plt.show()

    # -----------------------Plot data points -------------------------------------
    #
    # color=['m','r','c','y','k']
    # plt.figure(2)
    # for i in xrange(len(data)):
    #     for j in xrange(k):
    #         if assign[i,j]==1:
    #             plt.scatter(data[i, 0], data[i, 1], marker='x', color=color[j], s=30)
    #
    # for i in xrange(k):
    #     plt.scatter(center[i,0], center[i, 1], marker='x', color='b',label="Centers" if i == 0 else "", s=60)
    #
    # plt.legend(loc='lower right')
    #
    # plt.show()

    # -----------------------Evaluation on the validation set -------------------------------------
    valid_loss, valid_assign = sess.run([cost, assignment], feed_dict={X: validData})
    print "The loss for the validation data:",valid_loss


if __name__ == '__main__':
    # Parameters
    lr = 0.01
    num_updates = 600
    k = 6
    std = 1

    # np.set_printoptions(precision=2)

    run(lr, k, std, num_updates)
