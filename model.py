import math
import time

import numpy as np
import tensorflow as tf


def get_batch(Xs, ys, batch_size):
    inds = np.random.choice(Xs.shape[0], batch_size, replace=True)
    return Xs[inds,:], ys[inds,:]

def get_penalties(items, name, alpha):
    """returns penalties for a given set of weights or biases"""
    penalties = tf.nn.l2_loss(items) * alpha
    name = 'l2_{0}_pen'.format(name)
    _ = tf.histogram_summary(name, penalties)
    return penalties

def create_layer(input, name, out_size, activation, alpha):
    with tf.name_scope(name):
        in_size = int(input._shape[1])
        weights_dim =[in_size, out_size]
        stddev = 1.0 / math.sqrt(float(in_size))
        weights = tf.Variable(tf.random_normal(weights_dim, stddev=stddev, name='weights'))
        biases = tf.Variable(tf.zeros([out_size]), name='biases')
        logits = tf.matmul(input, weights) + biases
        _ = tf.histogram_summary('weights', weights)
        _ = tf.histogram_summary('biases', biases)
        response = activation(logits) if activation else logits
        
        penalties = []
        if alpha:
            i = ((weights, 'weights'), (biases, 'biases'))
            penalties = map(lambda x: get_penalties(x[0], x[1], alpha), i)
        
        return response, penalties

def calc_loss(logits, y_):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, y_, name='xentropy')
    loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    return loss

def tracked_train_step(loss, learning_rate):
    _ = tf.scalar_summary(loss.op.name, loss)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_step = optimizer.minimize(loss, global_step=global_step)
    return train_step

def train_nn_softmax(Xs, ys, shape, iterations, batch_size, learning_rate, 
                     penalty_alpha=0., logdir=None):

    with tf.Graph().as_default():

        with tf.Session() as sess:           

            x = tf.placeholder(tf.float32, [None, Xs.shape[1]])
            y_ = tf.placeholder(tf.float32, [None, ys.shape[1]])

            # define model
            prep_hidden = lambda x: ('hidden_{0}'.format(shape.index(x) + 1), x, tf.sigmoid, penalty_alpha)
            layer_defs = list(map(prep_hidden, shape))
            layer_defs.append(('softmax_linear', ys.shape[1], None, 0.))
            
            def add_layer_and_pens(inp, layer_def):
                model, existing_penalties = inp
                model, new_penalties = create_layer(model, *layer_def)                    
                penalties = existing_penalties + new_penalties
                return model, penalties

            logits, penalties = reduce(lambda inp, ld: add_layer_and_pens(inp, ld), layer_defs, (x, []))
            y = tf.nn.softmax(logits)
            _ = tf.histogram_summary('y', y)
             

            # set up objective
            loss = calc_loss(logits, y_) + sum(penalties)
            train_step = tracked_train_step(loss, learning_rate)
            _ = tf.scalar_summary('cross_entropy', loss)
            
            correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            _ = tf.scalar_summary('accuracy', accuracy)
            
            if logdir:
                summary_op = tf.merge_all_summaries()
                summary_writer = tf.train.SummaryWriter(logdir, graph_def=sess.graph_def)
            
            # train on batches
            start_time = time.time()
            init = tf.initialize_all_variables()
            sess.run(init)
            feed_dict={x: Xs, y_: ys}
            for i in xrange(iterations):
                bXs, bys = get_batch(Xs, ys, batch_size)
                _, loss_value = sess.run([train_step, loss], feed_dict={x: bXs, y_: bys})
                if i % 100 == 0:
                    duration = time.time() - start_time
                    msg = 'step {0:>5}:\tloss: {1:.2f}\t({2:.2f} sec)'
                    print(msg.format(i, loss_value, duration))
                    if logdir:
                        summary_str = sess.run(summary_op, feed_dict=feed_dict)
                        summary_writer.add_summary(summary_str, i)     
            
            res = sess.run([accuracy, loss], feed_dict)
            stats = {'accuracy': res[0],'cross_entropy': res[1]}
            print('accuracy:\t{0}'.format(res[0]))
            print('cross entropy:\t{0}'.format(res[1])) 

            # return predictions on train set
            prediction = tf.argmax(y,1)
            max_weight_label = sess.run(prediction, feed_dict)
            weights = sess.run(y, feed_dict)
            return weights, max_weight_label, stats