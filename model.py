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

def create_fc_layer(input, name, out_size, activation, alpha):
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

def create_conv_layer(input, name, out_depth):
    with tf.name_scope(name):
        width = input._shape[2]._value
        inp_depth = input._shape[3]._value
        stddev = 1.0 / math.sqrt(float(inp_depth))
        initial_weights = tf.random_normal([1, width, inp_depth, out_depth], stddev=stddev)
        #initial = tf.truncated_normal([1, width, inp_depth, out_depth], stddev=0.1)
        weights = tf.Variable(initial_weights, name='weights')
        biases = tf.Variable(tf.zeros([out_depth]), name='biases')
        #initial = tf.constant(0.1, shape=[out_depth])
        #biases = tf.Variable(initial, name='biases')
        _ = tf.histogram_summary('weights', weights)
        _ = tf.histogram_summary('biases', biases)
        conv2d = tf.nn.conv2d(input, weights, strides=[1, 1, 1, 1], padding='VALID')
        return tf.nn.relu(conv2d + biases)

def flatten_conv_layer(layer):
    depth = layer._shape[3]._value
    width = layer._shape[2]._value
    height = layer._shape[1]._value
    return tf.reshape(layer, [-1, depth*width*height])

def validate_dtype(array):
    return array.dtype == np.float32

def train_nn_softmax(Xs, ys, structure, iterations, batch_size, learning_rate, 
                     penalty_alpha=0., logdir=None):
    """train model on train set, test on train test set
    Xs and ys: lists contaiing train set and optionally a test set
    structure: list contianing convlayers depth and hidden layers depth
    returns: dict of model predictions, dict of stats
    """
    Xs_train = Xs[0]
    Xs_test = Xs[1] if len(Xs) > 1 else np.array([])
    ys_train = ys[0]
    ys_test = ys[1] if len(ys) > 1 else np.array([])
    for i in (Xs_train, Xs_test):
        if i.any() and not validate_dtype(i):
            raise TypeError('Xs must be numpy float32 type')

    with tf.Graph().as_default():

        with tf.Session() as sess:           

            if structure and  all(isinstance(i, list) for i in structure):
                conv_struct = structure[0]
                fc_struct = structure[1]
            else:
                conv_struct = None
                fc_struct = structure

            # define model
            y_ = tf.placeholder(tf.float32, [None, ys_train.shape[1]])
            Xs_shape = [None] + list(Xs_train.shape[1:])
            x = tf.placeholder(tf.float32, Xs_shape)
            
            if conv_struct:
                conv_name = lambda x: 'conv_layer_{0}'.format(conv_struct.index(x) + 1)
                layer_defs = map(lambda x: (conv_name(x), x), conv_struct)          
                x_4d = tf.reshape(x, [-1, Xs_shape[1], Xs_shape[2], 1])
                layer = reduce(lambda inp, ld: create_conv_layer(inp, ld[0], ld[1]), layer_defs, x_4d)
                init_fc_layer = flatten_conv_layer(layer)
            else:
                init_fc_layer = x

            fc_name = lambda x: 'fully_connected_{0}'.format(fc_struct.index(x) + 1)
            prep_fc_layers = lambda x: (fc_name(x), x, tf.sigmoid, penalty_alpha)
            layer_defs = list(map(prep_fc_layers, fc_struct))
            layer_defs.append(('softmax_linear', ys_train.shape[1], None, 0.))
            
            def add_layer_and_pens(inp, layer_def):
                model, existing_penalties = inp
                model, new_penalties = create_fc_layer(model, *layer_def)                    
                penalties = existing_penalties + new_penalties
                return model, penalties

            logits, penalties = reduce(lambda inp, ld: add_layer_and_pens(inp, ld), layer_defs, (init_fc_layer, []))
            y = tf.nn.softmax(logits)
            _ = tf.histogram_summary('y', y)
             
            # set up objective function and items to measure
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
            train_feed_dict={x: Xs_train, y_: ys_train}
            test_feed_dict={x: Xs_test, y_: ys_test}
            for i in xrange(iterations):
                bXs, bys = get_batch(Xs_train, ys_train, batch_size)
                _, train_loss_value = sess.run([train_step, loss], feed_dict={x: bXs, y_: bys})
                if i % 100 == 0:
                    test_loss_value = sess.run(loss, feed_dict=test_feed_dict) if Xs_test.any() else np.nan
                    duration = time.time() - start_time
                    msg = 'step {0:>5}:\ttrain loss: {1:.2f}\ttest loss: {2:.2f}\t\t({3:.2f} sec)'
                    print(msg.format(i, train_loss_value, test_loss_value, duration))
                    if logdir:
                        summary_str = sess.run(summary_op, feed_dict=train_feed_dict)
                        summary_writer.add_summary(summary_str, i)     
            
            # calc model predictions and summary stats
            predictions, stats = {}, {}
            prediction = tf.argmax(y,1)
            for i in (('train', train_feed_dict), ('test', test_feed_dict)):
                try: 
                    preds_res = sess.run([prediction, y], i[1])
                    stats_res = sess.run([accuracy, loss], i[1])
                except:
                    preds_res = np.nan, np.nan
                    stats_res = np.nan, np.nan
                predictions[i[0]] = {'max_weight_label': preds_res[0], 'weights': preds_res[1]} 
                stats[i[0]] = {'accuracy': stats_res[0], 'cross_entropy': stats_res[1]}
            
            msg = 'train accuracy:\t{0:.2f}\ttest accuracy:\t{1:.2f}'
            print(msg.format(stats['train']['accuracy'], stats['test']['accuracy']))
            msg = 'train loss:\t{0:.2f}\ttest loss:\t{1:.2f}'
            print(msg.format(stats['train']['cross_entropy'], stats['test']['cross_entropy']))

            return predictions, stats