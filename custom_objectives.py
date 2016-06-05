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

def calc_summary(metric, mean):
    if mean:
        return  tf.reduce_mean(metric)
    return tf.reduce_max(metric)

def calc_discount(metric, gain):
    lim_1 = metric / (tf.abs(metric) + gain)
    return 1. - lim_1
    
def sigmoid_ir(logits, y, y_, returns_, activation, return_per_days, sigmoid_gain, holdings_gain, gearing_gain, tilt_gain, mean):
    with tf.name_scope('sigmoid_ir'):
        net_holdings = tf.abs(calc_net_holdings(logits, y, y_, returns_))
        gearing = calc_gearing(logits, y, y_, returns_)
        tilts = tf.abs(tf.reduce_mean(y, 0))
        
        inp = [net_holdings, gearing, tilts]
        net_holdings, gearing, tilts = map(lambda x: calc_summary(x, mean), inp)
        gearing_ratio = tf.maximum(net_holdings, tilts) / gearing
        inp = ((net_holdings, holdings_gain), (tilts, tilt_gain), (gearing_ratio, gearing_gain))
        discounts = map(lambda x: calc_discount(x[0], x[1]), inp)
        discounts = reduce(lambda x, y: x * y, discounts)

        ir = calc_batch_ir(logits, y, y_, returns_, return_per_days)
        ir_scaled = ir * discounts
        return tf.sigmoid(sigmoid_gain * -ir_scaled)