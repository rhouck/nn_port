import math
import time

import numpy as np
import tensorflow as tf


def get_batch(Xs, ys, batch_size):
    inds = np.random.choice(Xs.shape[0], batch_size, replace=True)
    return Xs[inds,:], ys[inds,:]

def add_layer(input, name, out_size, activation):
    with tf.name_scope(name):
        in_size = int(input._shape[1])
        weights_dim =[in_size, out_size]
        stddev = 1.0 / math.sqrt(float(in_size))
        weights = tf.Variable(tf.truncated_normal(weights_dim, stddev=stddev, name='weights'))
        biases = tf.Variable(tf.zeros([out_size]), name='biases')
        logits = tf.matmul(input, weights) + biases
        return activation(logits) if activation else logits

def calc_loss(logits, y_):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, y_, name='xentropy')
    loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    #loss = -tf.reduce_sum(y_*tf.log(tf.nn.softmax(logits)), name='cross_entropy')
    return loss

def tracked_train_step(loss, learning_rate):
    tf.scalar_summary(loss.op.name, loss)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_step = optimizer.minimize(loss, global_step=global_step)
    #train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    return train_step

def train_nn_softmax(Xs, ys, mnist, batch_size, iterations, shape):

    learning_rate = 0.01
    # run tensorboard:
    # ./venv/bin/tensorboard --logdir==/Users/ryanchouck/dev/tf/data/tensor_board
    train_dir = '/Users/ryanchouck/dev/tf/data/tensor_board'
 
    with tf.Graph().as_default():

        x = tf.placeholder(tf.float32, [None, Xs.shape[1]])
        y_ = tf.placeholder(tf.float32, [None, ys.shape[1]])

        # define model
        prep_hidden = lambda x: ('hidden_{0}'.format(shape.index(x) + 1), x, tf.nn.relu)
        layers = list(map(prep_hidden, shape))
        layers.append(('softmax_linear', ys.shape[1], None))
        logits = reduce(lambda inp, layer: add_layer(inp, layer[0], layer[1], layer[2]), layers, x)
        y = tf.nn.softmax(logits)
          
        # set up objective
        loss = calc_loss(logits, y_)
        train_step = tracked_train_step(loss, learning_rate)
        summary_op = tf.merge_all_summaries()
        init = tf.initialize_all_variables()
        
        with tf.Session() as sess:         
            
            start_time = time.time()
            sess.run(init)
            summary_writer = tf.train.SummaryWriter(train_dir, graph_def=sess.graph_def)
            feed_dict={x: Xs, y_: ys}
            for i in xrange(iterations):
                batch_Xs, batch_ys = get_batch(Xs, ys, batch_size)
                _, loss_value = sess.run([train_step, loss], 
                                         feed_dict={x: batch_Xs, y_: batch_ys})
                if i % 100 == 0:
                    duration = time.time() - start_time
                    msg = 'step {0:>5}:\tloss: {1:.2f}\t({2:.2f} sec)'
                    print(msg.format(i, loss_value, duration))
                    summary_str = sess.run(summary_op, feed_dict=feed_dict)
                    summary_writer.add_summary(summary_str, i)
            
            # calculate accuracy on train set
            correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            print('accuracy:\t{0}'.format(sess.run(accuracy, feed_dict)))
            print('cross entropy:\t{0}'.format(sess.run(loss, feed_dict))) 

            # return predictions on train set
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