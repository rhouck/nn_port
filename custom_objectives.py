import numpy as np
import tensorflow as tf


def calc_batch_ir(logits, y, y_, returns_, return_per_days):
    q = 252. / return_per_days
    returns = tf.reduce_sum(tf.mul(returns_, y), 1)    
    returns_mean = tf.reduce_mean(returns)
    sum_squares = tf.reduce_sum(tf.square(tf.sub(returns, returns_mean)))
    count = tf.to_float(tf.shape(returns)[0])
    ann_std_dev = tf.sqrt(sum_squares / count) * np.sqrt(q)
    ann_returns_mean = returns_mean * q
    return ann_returns_mean / ann_std_dev

def calc_net_holdings(logits, y, y_, returns_):
    return tf.reduce_sum(y, 1)

def calc_gearing(logits, y, y_, returns_):
    abs_holdings = tf.abs(y) / 2.
    return tf.reduce_sum(abs_holdings, 1)
    # zeros = tf.zeros(tf.to_int32(tf.shape(y)), dtype=tf.float32)
    # pos_holdings = tf.maximum(y, zeros)
    # return tf.reduce_sum(pos_holdings, 1)

def calc_discount(metric, mean, gain):
    if mean:
        sel_metric = tf.reduce_mean(metric)
    else:
        sel_metric = tf.reduce_max(metric)
    #lim_1 = tf.tanh(sel_metric)
    lim_1 = sel_metric / (tf.abs(sel_metric) + gain)
    return 1. - lim_1
    
def sigmoid_ir(logits, y, y_, returns_, activation, return_per_days, sigmoid_gain, holdings_gain, gearing_gain, tilt_gain, mean):
    with tf.name_scope('sigmoid_ir'):

        net_holdings = calc_net_holdings(logits, y, y_, returns_)
        gearing = calc_gearing(logits, y, y_, returns_)
        #gearing_error = tf.abs(1. - gearing)
        gearing_ratio = tf.abs(net_holdings) / gearing
        tilts = tf.reduce_mean(y, 0)

        net_holdings_discount = calc_discount(net_holdings, mean, holdings_gain)
        #gearing_discount = calc_discount(gearing_error, mean, gearing_gain)  
        gearing_discount = calc_discount(gearing_ratio, mean, gearing_gain)
        tilt_discount = calc_discount(tilts, mean, tilt_gain)
        
        ir = calc_batch_ir(logits, y, y_, returns_, return_per_days)
        ir_scaled = ir * net_holdings_discount * gearing_discount * tilt_discount
        return tf.sigmoid(sigmoid_gain * -ir_scaled)