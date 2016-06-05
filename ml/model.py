import math
import time

import numpy as np
import tensorflow as tf


def get_batch(Xs, ys, returns, batch_size):
    inds = np.random.choice(Xs.shape[0], batch_size, replace=True)
    return Xs[inds,:], ys[inds,:], returns[inds,:]

def get_penalties(items, alpha):
    """returns penalties for a given set of weights or biases"""
    penalties = tf.nn.l2_loss(items) * alpha
    name = 'l2_{0}_pen'.format(items.name.split(':')[0])
    _ = tf.histogram_summary(name, penalties)
    return penalties

def calc_loss(logits, y, y_, returns_, activation):
    if activation is None:
        name = 'squared_error'
        loss_func = tf.square(y - y_, name=name)
    elif activation.__name__ == 'softmax':
        name ='softmax_xentropy'
        loss_func = tf.nn.softmax_cross_entropy_with_logits(logits, y_, name=name)
    elif activation.__name__ == 'sigmoid':
        name = 'sigmoid_xentropy'
        loss_func = tf.nn.sigmoid_cross_entropy_with_logits(logits, y_, name=name)
    else:
        raise Exception('no appropriate loss function paired with this activation')
    loss_objective = tf.reduce_mean(loss_func, name=name + '_mean')
    return loss_objective

def tracked_train_step(loss, learning_rate):
    _ = tf.scalar_summary(loss.op.name, loss)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_step = optimizer.minimize(loss, global_step=global_step)
    return train_step

def get_initial_weights_and_biases(weights_dim, activation):
    stddev = math.sqrt(1.0 / float(weights_dim[-2]))
    if activation is None or activation.__name__ in ('sigmoid', 'tanh'):
        weights = tf.random_normal(weights_dim, stddev=stddev)
        biases = tf.zeros([weights_dim[-1]])
    elif activation.__name__ == 'relu':  
        stddev *= math.sqrt(2.)
        weights = tf.truncated_normal(weights_dim, stddev=stddev)
        biases = tf.constant(.1, shape=[weights_dim[-1]])
    else:
        raise Exception('get init weights not yet implemented for this activation')

    weights = tf.Variable(weights, name='weights')
    biases = tf.Variable(biases, name='biases')
    return weights, biases

def create_fc_layer(input, name, out_size, activation):
    with tf.name_scope(name):
        in_size = int(input._shape[1])
        weights_dim =[in_size, out_size]
        weights, biases = get_initial_weights_and_biases(weights_dim, activation)
        _ = tf.histogram_summary(name + '_weights', weights)
        _ = tf.histogram_summary(name + '_biases', biases)
        logits = tf.matmul(input, weights) + biases
        response = activation(logits) if activation else logits
        return response, weights, biases

def create_conv_layer(input, name, out_depth, activation):
    with tf.name_scope(name):
        width = input._shape[2]._value
        inp_depth = input._shape[3]._value
        weights_dim = [1, width, inp_depth, out_depth]
        weights, biases = get_initial_weights_and_biases(weights_dim, activation)
        _ = tf.histogram_summary(name + '_weights', weights)
        _ = tf.histogram_summary(name + '_biases', biases)
        conv2d = tf.nn.conv2d(input, weights, strides=[1, 1, 1, 1], padding='VALID')
        logits = conv2d + biases
        response = activation(logits) if activation else logits
        return response, weights, biases

def add_layer(inp, layer_func, dropout_rate, *layer_def_args):
    """feeds model layers into subsequent model layers while 
    consolidating weights and biases to calculate loss function penalties"""
    layer, weights, biases = inp             
    create_layer_inps = [layer,] +  list(layer_def_args)
    layer, new_weights, new_biases = layer_func(*create_layer_inps)                    
    layer = tf.nn.dropout(layer, 1. - dropout_rate)
    if new_weights:
        weights.append(new_weights)
    if new_biases:
        biases.append(new_biases)
    return layer, weights, biases

def combine_layers(create_layer_func, dropout_rate, layer_defs, inputs):
    return reduce(lambda inp, ld: add_layer(inp, create_layer_func, dropout_rate, *ld), layer_defs, inputs)

def define_layers(base_name, layer_structure, *args):
    """returns list of lists of inputs to feed 'add layer' functions""" 
    name_func = lambda x: '{0}_{1}'.format(base_name, x + 1)
    ind_shape_pairs = zip(range(len(layer_structure)), layer_structure)
    return map(lambda x: [name_func(x[0]), x[1]] + list(args), ind_shape_pairs)          

def flatten_conv_layer(layer):
    depth = layer._shape[3]._value
    width = layer._shape[2]._value
    height = layer._shape[1]._value
    return tf.reshape(layer, [-1, depth*width*height])

def split_fc_conv_input(inp):
    if not isinstance(inp, list):
        inp = [inp,]
    if len(inp) == 1:
        conv = inp[0]
        fc = inp[0]
    else:
        conv = inp[0]
        fc = inp[1]
    return conv, fc

def train_nn(data, structure, iterations, batch_size, learning_rate, 
             penalty_alpha=0., 
             train_dropout_rate=0., 
             logdir=None, 
             verbosity=100, 
             conv_layer_activation=tf.nn.relu,
             fc_hidden_layer_activation=tf.sigmoid, 
             fc_final_layer_activation=tf.nn.softmax,
             loss_func=calc_loss,
             train_step_func=tracked_train_step,
             performance_funcs={}):
    """train model on train set, test on train test set
    Xs and ys: lists contaiing train set and optionally a test set
    structure: list contianing convlayers depth and hidden layers depth
    returns: dict of model predictions, dict of stats
    """

    # define inputs and model structure
    def split_train_test(array):
        array_train = array[0]
        array_test = array[1] if len(array) > 1 else np.array([[]])
        return array_train, array_test
    
    Xs_train, Xs_test = split_train_test(data[0])
    ys_train, ys_test = split_train_test(data[1])
    try:
        returns_train, returns_test = split_train_test(data[2])
    except:
        returns_train = np.empty(ys_train.shape) * np.nan
        returns_test = np.empty(ys_test.shape) * np.nan
    
    if structure and  all(isinstance(i, list) for i in structure):
        conv_struct = structure[0]
        fc_struct = structure[1]
    else:
        conv_struct = None
        fc_struct = structure

    conv_penalty_alpha, fc_penalty_alpha = split_fc_conv_input(penalty_alpha)
    train_conv_dropout_rate, train_fc_dropout_rate = split_fc_conv_input(train_dropout_rate)
    
    penalties = []
    with tf.Graph().as_default():
        with tf.Session() as sess:           

            # setup placeholders
            y_ = tf.placeholder(tf.float32, [None, ys_train.shape[1]], name='y_')
            returns_ = tf.placeholder(tf.float32, [None, returns_train.shape[1]], name='returns_')
            Xs_shape = [None] + list(Xs_train.shape[1:])
            x = tf.placeholder(tf.float32, Xs_shape, name='x')
            conv_dropout_rate = tf.placeholder("float")
            fc_dropout_rate = tf.placeholder("float")

            # define and create convolution layers if needed
            if conv_struct:
                conv_layer_defs = define_layers('conv_layer', conv_struct, conv_layer_activation)
                x_4d = tf.reshape(x, [-1, Xs_shape[1], Xs_shape[2], 1])
                conv_inps = (x_4d, [], [])
                final_conv_layer, conv_weights, conv_biases = combine_layers(create_conv_layer, conv_dropout_rate, conv_layer_defs, conv_inps)
                init_fc_layer = flatten_conv_layer(final_conv_layer)
                penalties.append(sum(map(lambda x: get_penalties(x, conv_penalty_alpha), conv_weights)))
            else:
                init_fc_layer = x
            
            # define fully connected hidden layers and final softmax layer
            fc_layer_defs = define_layers('fully_connected_hidden', fc_struct, fc_hidden_layer_activation)
            fc_layer_defs.append(['fully_connected_final_layer', ys_train.shape[1], None])
            fc_inps = (init_fc_layer, [], [])
            logits, fc_weights, fc_biases = combine_layers(create_fc_layer, fc_dropout_rate, fc_layer_defs, fc_inps)
            y = fc_final_layer_activation(logits) if fc_final_layer_activation else logits
            _ = tf.histogram_summary('y', y)
            penalties.append(sum(map(lambda x: get_penalties(x, fc_penalty_alpha), fc_weights)))
             
            # set up objective function and items to measure
            loss = loss_func(logits, y, y_, returns_, fc_final_layer_activation) + sum(penalties)
            train_step = train_step_func(loss, learning_rate)
            _ = tf.scalar_summary('loss', loss)
            
            correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            _ = tf.scalar_summary('accuracy', accuracy)

            for k, v in performance_funcs.items():
                performance_funcs[k] = v(logits, y, y_, returns_)
                _ = tf.scalar_summary(k, performance_funcs[k])                


            # record training statistics
            if logdir:
                summary_op = tf.merge_all_summaries()
                summary_writer = tf.train.SummaryWriter(logdir, graph_def=sess.graph_def)
            
            # train on batches
            start_time = time.time()
            init = tf.initialize_all_variables()
            sess.run(init)
            train_feed_dict={x: Xs_train, y_: ys_train, returns_: returns_train, conv_dropout_rate: 0., fc_dropout_rate: 0.}
            test_feed_dict={x: Xs_test, y_: ys_test, returns_: returns_test, conv_dropout_rate: 0., fc_dropout_rate: 0.}
            for i in xrange(iterations):
                bXs, bys, brets = get_batch(Xs_train, ys_train, returns_train, batch_size)
                batch_train_feed_dict = {x: bXs, y_: bys, returns_: brets, conv_dropout_rate: train_conv_dropout_rate, fc_dropout_rate: train_fc_dropout_rate}
                _, train_loss_value = sess.run([train_step, loss], feed_dict=batch_train_feed_dict)
                if verbosity and i % verbosity == 0 and i > 0:

                    test_loss_value = sess.run(loss, feed_dict=test_feed_dict) if Xs_test.any() else np.nan
                    
                    duration = time.time() - start_time
                    msg = 'step {0:>7}:\ttrain loss: {1:.5f}\ttest loss: {2:.5f}\t({3:.2f} sec)\n\t\t'
                    msg = msg.format(i, train_loss_value, test_loss_value, duration)
                    if performance_funcs:
                        for k, v in performance_funcs.items():
                            for j in (('train', batch_train_feed_dict), ('test', test_feed_dict)):
                                res = sess.run(v, feed_dict=j[1])
                                msg += '{0} {1}: {2:.5f}\t'.format(j[0], k, res)
                            msg += '\n\t\t'
                    print msg

                    if logdir:
                        summary_str = sess.run(summary_op, feed_dict=batch_train_feed_dict)
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
                predictions[i[0]] = {'labels': preds_res[0], 'weights': preds_res[1]} 
                stats[i[0]] = {'accuracy': stats_res[0], 'loss': stats_res[1]}
            
            # msg = 'train accuracy:\t{0:.2f}\ttest accuracy:\t{1:.2f}'
            # print(msg.format(stats['train']['accuracy'], stats['test']['accuracy']))
            msg = 'train loss:\t{0:.5f}\ttest loss:\t{1:.5f}'
            print(msg.format(stats['train']['loss'], stats['test']['loss']))

            return predictions, stats