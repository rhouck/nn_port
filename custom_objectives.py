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
    
def sigmoid_ir(logits, y, y_, returns_, activation, return_per_days, gain, holdings_gain, gearing_alpha, mean):
    with tf.name_scope('sigmoid_ir'):

        net_holdings = calc_net_holdings(logits, y, y_, returns_)
        
        if mean:        
            # get average positive net holdings
            zeros = tf.zeros(tf.to_int32(tf.shape(net_holdings)), dtype=tf.float32)
            max_net_holdings = tf.reduce_mean(tf.maximum(net_holdings, zeros))
        else:
            # get max positive net holdings
            max_net_holdings = tf.maximum(0., tf.reduce_max(net_holdings))
        
        # get average non-zero holdings
        #max_net_holdings = tf.reduce_mean(tf.abs(net_holdings))
        
        # get max non-zero holdings
        #max_net_holdings = tf.reduce_max(tf.abs(net_holdings))
        
        holdings_lim_1 = max_net_holdings / (tf.abs(max_net_holdings) + holdings_gain)
        holdings_penalty = 1. - holdings_lim_1
        
        ir = calc_batch_ir(logits, y, y_, returns_, return_per_days)
        
        ir_scaled = ir * holdings_penalty
        sigmoid_ir = tf.sigmoid(-gain * ir_scaled)
        
        gearing = calc_gearing(logits, y, y_, returns_)
        avg_gearing = tf.reduce_mean(gearing)
        gearing_diff = tf.square(1. - avg_gearing) * gearing_alpha
        
        return 2. * sigmoid_ir + gearing_diff