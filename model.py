import math

import numpy as np
import tensorflow as tf


def get_batch(Xs, ys, batch_size):
    inds = np.random.choice(Xs.shape[0], batch_size, replace=True)
    return Xs[inds,:], ys[inds,:]


def train_nn_softmax(Xs, ys, mnist, batch_size, iterations, shape):

    X_width = Xs.shape[1]
    y_width = ys.shape[1]

    x = tf.placeholder(tf.float32, [None, X_width])
    y_ = tf.placeholder(tf.float32, [None, y_width])

    hidden1_units = shape[0]
    hidden2_units = shape[1]

    # Hidden 1
    with tf.name_scope('hidden1'):
        weights = tf.Variable(tf.truncated_normal([X_width, hidden1_units], 
                                                  stddev=1.0 / math.sqrt(float(X_width))),
                              name='weights')
        biases = tf.Variable(tf.zeros([hidden1_units]),name='biases')
        hidden1 = tf.nn.relu(tf.matmul(x, weights) + biases)
    # Hidden 2
    with tf.name_scope('hidden2'):
        weights = tf.Variable(tf.truncated_normal([hidden1_units, hidden2_units], 
                                                  stddev=1.0 / math.sqrt(float(hidden1_units))),
                              name='weights')
        biases = tf.Variable(tf.zeros([hidden2_units]),name='biases')
        hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
    # Linear
    with tf.name_scope('softmax_linear'):
        weights = tf.Variable(tf.truncated_normal([hidden2_units, y_width],
                                                   stddev=1.0 / math.sqrt(float(hidden2_units))),
                              name='weights')
        biases = tf.Variable(tf.zeros([y_width]),name='biases')
        logits = tf.matmul(hidden2, weights) + biases
        y = tf.nn.softmax(logits)

    cross_entropy = -tf.reduce_sum(y_*tf.log(y))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    for i in range(iterations):
        #batch_Xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_Xs, batch_ys = get_batch(Xs, ys, batch_size)
        sess.run(train_step, feed_dict={x: batch_Xs, y_: batch_ys})

    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # may want to judge accuracy on test set instead
    feed_dict={x: Xs, y_: ys}
    #feed_dict={x: mnist.test.images, y_: mnist.test.labels}
    acc = sess.run(correct_prediction, feed_dict)
    print (sum(acc) * 1.) / len(acc)

    print sess.run(accuracy, feed_dict)

    prediction = tf.argmax(y,1)
    max_weight_label = sess.run(prediction, feed_dict)
    weights = sess.run(y, feed_dict)
    return weights, max_weight_label

# def predict(sess, Xs):
#     with sess.as_default():
#         feed_dict={x: Xs}
#         prediction=tf.argmax(y,1)
#         print prediction.eval(feed_dict)
#         preds = (sess.run(y, feed_dict))
#         print preds
#         #pd.DataFrame(preds).plot(alpha=.5)