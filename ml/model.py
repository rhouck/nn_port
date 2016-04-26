import math
import time

import numpy as np
import tensorflow as tf


def get_batch(Xs, ys, returns, batch_size):
    inds = np.random.choice(Xs.shape[0], batch_size, replace=True)
    return Xs[inds,:], ys[inds,:], returns[inds,:]

def get_penalties(items, name, alpha):
    """returns penalties for a given set of weights or biases"""
    penalties = tf.nn.l2_loss(items) * alpha
    name = 'l2_{0}_pen'.format(name)
    #_ = tf.histogram_summary(name, penalties)
    return penalties

def calc_loss(logits, y, y_, returns_, fc_final_layer_activation):
    if fc_final_layer_activation is None:
        name = 'squared_error'
        loss_func = tf.square(y - y_, name=name)
    elif fc_final_layer_activation.__name__ == 'softmax':
        name ='softmax_xentropy'
        loss_func = tf.nn.softmax_cross_entropy_with_logits(logits, y_, name=name)
    elif fc_final_layer_activation.__name__ == 'sigmoid':
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

def create_fc_layer(input, name, out_size, activation, alpha, dropout_rate):
    with tf.name_scope(name):
        in_size = int(input._shape[1])
        weights_dim =[in_size, out_size]
        stddev = 1.0 / math.sqrt(float(in_size))
        weights = tf.Variable(tf.random_normal(weights_dim, stddev=stddev), name='weights')
        biases = tf.Variable(tf.zeros([out_size]), name='biases')
        #_ = tf.histogram_summary('weights', weights)
        #_ = tf.histogram_summary('biases', biases)

        logits = tf.matmul(input, weights) + biases
        response = activation(logits) if activation else logits
        if dropout_rate: 
            response = tf.nn.dropout(response, 1. - dropout_rate)

        penalties = []
        if alpha:
            i = ((weights, 'weights'),)# (biases, 'biases'))
            penalties = map(lambda x: get_penalties(x[0], x[1], alpha), i)
        
        return response, penalties

def add_fc_layer_and_penalties(inp, *layer_def_args):
    """feeds model layers into subsequent model layers while 
    collecting penalty objects to separately add to loss function"""
    model, existing_penalties = inp             
    create_fc_layer_inps = [model,] +  list(layer_def_args)
    model, new_penalties = create_fc_layer(*create_fc_layer_inps)                    
    penalties = existing_penalties + new_penalties
    return model, penalties

def create_conv_layer(input, name, out_depth, activation):
    with tf.name_scope(name):
        width = input._shape[2]._value
        inp_depth = input._shape[3]._value
        stddev = 1.0 / math.sqrt(float(inp_depth))
        initial_weights = tf.random_normal([1, width, inp_depth, out_depth], stddev=stddev)
        weights = tf.Variable(initial_weights, name='weights')
        biases = tf.Variable(tf.zeros([out_depth]), name='biases')
        #_ = tf.histogram_summary('weights', weights)
        #_ = tf.histogram_summary('biases', biases)
        
        conv2d = tf.nn.conv2d(input, weights, strides=[1, 1, 1, 1], padding='VALID')
        return activation(conv2d + biases)

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

def train_nn(Xs, ys, returns, structure, iterations, batch_size, learning_rate, 
             penalty_alpha=0., dropout_rate=0., logdir=None, verbosity=100, 
             conv_layer_activation=tf.nn.relu,
             fc_hidden_layer_activation=tf.sigmoid, 
             fc_final_layer_activation=tf.nn.softmax,
             loss_func=calc_loss,
             performance_funcs={}):
    """train model on train set, test on train test set
    Xs and ys: lists contaiing train set and optionally a test set
    structure: list contianing convlayers depth and hidden layers depth
    returns: dict of model predictions, dict of stats
    """

    # input validation and formatting
    def split_train_test(array):
        array_train = array[0]
        array_test = array[1] if len(array) > 1 else np.array([])
        return array_train, array_test
    
    Xs_train, Xs_test = split_train_test(Xs)
    ys_train, ys_test = split_train_test(ys)
    returns_train, returns_test = split_train_test(returns)
    
    def validate_dtype(array):
        return array.dtype == np.float32

    for i in (Xs_train, Xs_test):
        if i.any() and not validate_dtype(i):
            raise TypeError('Xs must be numpy float32 type')
    
    if structure and  all(isinstance(i, list) for i in structure):
        conv_struct = structure[0]
        fc_struct = structure[1]
    else:
        conv_struct = None
        fc_struct = structure

    
    with tf.Graph().as_default():

        with tf.Session() as sess:           

            # setup placeholders
            y_ = tf.placeholder(tf.float32, [None, ys_train.shape[1]])
            returns_ = tf.placeholder(tf.float32, [None, returns_train.shape[1]])
            Xs_shape = [None] + list(Xs_train.shape[1:])
            x = tf.placeholder(tf.float32, Xs_shape)
            
            def combine_layers(create_layer_func, layer_defs, inputs):
                return reduce(lambda inp, ld: create_layer_func(inp, *ld), layer_defs, inputs)

            # define and create convolution layers if needed
            if conv_struct:
                conv_layer_defs = define_layers('conv_layer', conv_struct, conv_layer_activation)
                x_4d = tf.reshape(x, [-1, Xs_shape[1], Xs_shape[2], 1])
                final_conv_layer = combine_layers(create_conv_layer, conv_layer_defs, x_4d)
                init_fc_layer = flatten_conv_layer(final_conv_layer)
            else:
                init_fc_layer = x
            
            # define fully connected hidden layers and final softmax layer
            fc_layer_defs = define_layers('fully_connected_hidden', fc_struct, fc_hidden_layer_activation, penalty_alpha, dropout_rate)
            fc_layer_defs.append(['fully_connected_final_layer', ys_train.shape[1], None, penalty_alpha, dropout_rate])
            logits, penalties = combine_layers(add_fc_layer_and_penalties, fc_layer_defs, (init_fc_layer, []))
            y = fc_final_layer_activation(logits) if fc_final_layer_activation else logits
            _ = tf.histogram_summary('y', y)
             
            # set up objective function and items to measure
            loss = loss_func(logits, y, y_, returns_, fc_final_layer_activation) + sum(penalties)
            train_step = tracked_train_step(loss, learning_rate)
            _ = tf.scalar_summary('loss', loss)
            
            correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            _ = tf.scalar_summary('accuracy', accuracy)
            
            # record training statistics
            if logdir:
                summary_op = tf.merge_all_summaries()
                summary_writer = tf.train.SummaryWriter(logdir, graph_def=sess.graph_def)
            
            # train on batches
            start_time = time.time()
            init = tf.initialize_all_variables()
            sess.run(init)
            train_feed_dict={x: Xs_train, y_: ys_train, returns_: returns_train}
            test_feed_dict={x: Xs_test, y_: ys_test, returns_: returns_test}
            for i in xrange(iterations):
                bXs, bys, brets = get_batch(Xs_train, ys_train, returns_train, batch_size)
                _, train_loss_value = sess.run([train_step, loss], feed_dict={x: bXs, y_: bys, returns_: brets})
                if verbosity and i % verbosity == 0:
                    test_loss_value = sess.run(loss, feed_dict=test_feed_dict) if Xs_test.any() else np.nan
                    duration = time.time() - start_time
                    
                    msg = 'step {0:>7}:\t'.format(i)
                    for j in (('train', train_feed_dict), ('test', test_feed_dict)):
                        pred = sess.run(y, j[1])
                        for k, v in performance_funcs.items():
                            msg += '{0} {1}: {2:.2f}\t'.format(j[0], k, v(pred))
                    msg += 'train loss: {0:.4f}\ttest loss: {1:.4f}\t({2:.2f} sec)'
                    print(msg.format(train_loss_value, test_loss_value, duration))
                    
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
                predictions[i[0]] = {'labels': preds_res[0], 'weights': preds_res[1]} 
                stats[i[0]] = {'accuracy': stats_res[0], 'cross_entropy': stats_res[1]}
            
            msg = 'train accuracy:\t{0:.2f}\ttest accuracy:\t{1:.2f}'
            print(msg.format(stats['train']['accuracy'], stats['test']['accuracy']))
            msg = 'train loss:\t{0:.2f}\ttest loss:\t{1:.2f}'
            print(msg.format(stats['train']['cross_entropy'], stats['test']['cross_entropy']))

            return predictions, stats