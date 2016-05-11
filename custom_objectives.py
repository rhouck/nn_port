import numpy as np
import tensorflow as tf


def calc_batch_ir(logits, y, y_, returns_):
    q = 252.
    returns = tf.reduce_sum(tf.mul(returns_, y), 1)    
    returns_mean = tf.reduce_mean(returns)
    sum_squares = tf.reduce_sum(tf.square(tf.sub(returns, returns_mean)))
    count = tf.to_float(tf.shape(returns)[0])
    ann_std_dev = tf.sqrt(sum_squares / count) * np.sqrt(q)
    ann_returns_mean = returns_mean * q
    return ann_returns_mean / ann_std_dev

def calc_gearing(logits, y, y_, returns_):
    abs_holdings = tf.abs(y) / 2.
    return tf.reduce_mean(tf.reduce_sum(abs_holdings, 1))

def sigmoid_ir(logits, y, y_, returns_, activation):
    with tf.name_scope('sigmoid_ir'):
        
        ir = calc_batch_ir(logits, y, y_, returns_)
        sigmoid_ir = tf.sigmoid(-.3 * ir) * .5
        
        net_holdings = tf.reduce_mean(tf.reduce_sum(y, 1))
        holdings_diff = tf.abs(0. - net_holdings) * 1.
        
        gearing = calc_gearing(logits, y, y_, returns_)
        gearing_diff = tf.abs(1. - gearing) * 1.
        
        return sigmoid_ir + holdings_diff + gearing_diff