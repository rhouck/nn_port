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
    
def sigmoid_ir(logits, y, y_, returns_, activation, return_per_days, gain, holdings_gain, gearing_alpha):
    with tf.name_scope('sigmoid_ir'):

        # max net holdings penalty
        max_net_holdings = tf.reduce_max(tf.abs(calc_net_holdings(logits, y, y_, returns_)))
        holdings_lim_1 = max_net_holdings / (tf.abs(max_net_holdings) + holdings_gain)
        holdings_penalty = 1. - holdings_lim_1
        #holdings_penalty = 1. - tf.minimum(.99, max_net_holdings * holdings_penalty_alpha)
        ir = calc_batch_ir(logits, y, y_, returns_, return_per_days)
        ir_scaled = ir * holdings_penalty
        sigmoid_ir = tf.sigmoid(-gain * ir_scaled)
        
        gearing = calc_gearing(logits, y, y_, returns_)
        avg_gearing = tf.reduce_mean(gearing)
        gearing_diff = tf.square(1. - avg_gearing) * gearing_alpha
        
        return 2. * sigmoid_ir + gearing_diff