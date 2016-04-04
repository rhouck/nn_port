import unittest
import datetime

import numpy as np
import pandas as pd

import model_inputs as mi
import model as md
import tensorflow as tf
from gen_data import *


class TestModelInputs(unittest.TestCase):

    def test_validation_doesnt_remove_or_distord_data_unnecessarilly(self):
        self.assertTrue(False)

class TestDataGeneration(unittest.TestCase):
    
    def setUp(self):
        np.random.seed(0)
        self.dti = pd.DatetimeIndex(start='2000-1-1', freq='B', periods=1000)

    def check_kernel_predictive_accuracy(self, Xs, ys, true_weights):
        """apply kernel to Xs and return correct prediction accuracy"""
        weights_df = pd.DataFrame(np.array([true_weights,]*Xs.shape[1]), index=Xs.major_axis, columns=Xs.minor_axis)
        scaled = Xs.multiply(weights_df)
        
        def get_max_ind(df):
            x = list(df.sum(axis=1).values)
            return x.index(max(x))

        df = pd.DataFrame({'xs': pd.Series([get_max_ind(scaled[item]) for item in scaled.items], index=ys.index),
                           'ys': ys.apply(lambda x: list(x).index(max(x)), axis=1)})
        matches = df.apply(lambda x: x[0]==x[1], axis=1)
        return float(matches.sum()) / matches.shape[0]

    def test_kernel_has_perfect_pred_power_with_no_noise(self):
        Xs, ys, tw = gen_2d_random_Xs_onehot_ys_from_random_kernel(self.dti, 15, 10, 0.)
        self.assertEquals(self.check_kernel_predictive_accuracy(Xs, ys, tw), 1.)

    def test_adding_noise_reduces_kernel_predicted_power(self):    
        accs = []
        for i in (100., 10., 1., 0.):
            Xs, ys, tw = gen_2d_random_Xs_onehot_ys_from_random_kernel(self.dti, 15, 10, i)
            accs.append(self.check_kernel_predictive_accuracy(Xs, ys, tw))
        self.assertEquals(sorted(accs), accs)

class TestModel(unittest.TestCase):
    
    def setUp(self):
        np.random.seed(0)
        self.dti = pd.DatetimeIndex(start='2000-1-1', freq='B', periods=1000)
        self.ys_probs = gen_random_probs(self.dti, 10)
        self.ys_labels = gen_random_onehot(self.dti, 10)
        Xs = gen_random_normal(self.dti, 20)
        self.Xs, _ = mi.validate_and_format_Xs_ys(Xs, self.ys_probs)
        Xs_conv, ys_labels_conv, _ = gen_2d_random_Xs_onehot_ys_from_random_kernel(self.dti, 5, 10)
        self.Xs_conv, self.ys_labels_conv = mi.validate_and_format_Xs_ys(Xs_conv, ys_labels_conv)

    def test_input_output_are_same_shape(self):
        inp = self.ys_labels.values.astype(np.float32)
        probs, labels, _ = md.train_nn_softmax(inp, inp, [], 50, 10, .1)
        self.assertEquals(probs.shape, inp.shape)
        self.assertEquals(labels.shape[0], inp.shape[0])

    def test_random_inputs_creates_equal_weights(self):
        Xs = self.Xs.values
        for ys in (self.ys_probs.values, self.ys_labels.values):
            probs, _, _ = md.train_nn_softmax(Xs, ys, [], 1000, 100, .1)
            probs = pd.DataFrame(probs)
            equal_weight = 1. / probs.shape[1]
            dist_from_equal = (probs.mean() - equal_weight).abs()
            short_dist = (dist_from_equal < .05)
            self.assertTrue(short_dist.all())

    def test_single_softmax_learns_onehot_w_perfect_foresight(self):
        inp = self.ys_labels.values.astype(np.float32)
        _, _, stats = md.train_nn_softmax(inp, inp, [], 500, 100, .1)
        self.assertTrue(stats['accuracy'] > .99)

    def test_hidden_layers_softmax_learns_onehot_w_perfect_foresight(self):
        inp = self.ys_labels.values.astype(np.float32)
        _, _, stats  = md.train_nn_softmax(inp, inp, [10,], 1000, 100, .1)
        self.assertTrue(stats['accuracy'] > .99)

    def test_single_softmax_learns_opt_weights_w_perfect_foresight(self):
        ys = pd.read_csv('tests/test_data/opt_weights_20.csv', index_col=0, parse_dates=['Date',])
        probs, _, _ = md.train_nn_softmax(ys.values.astype(np.float32), ys.values, [], 2000, 1000, .4)
        probs = pd.DataFrame(probs, columns=ys.columns, index=ys.index)
        self.assertTrue(probs.stack().corr(ys.stack()) > .95) 

    def test_noise_input_leads_to_stable_label_pred_based_on_max_val_freqs(self):
        ys = pd.read_csv('tests/test_data/opt_weights_20.csv', index_col=0, parse_dates=['Date',])
        Xs = gen_random_normal(ys.index, 20).values.astype(np.float32)
        probs, _, _ = md.train_nn_softmax(Xs, ys.values, [], 1000, 100, .1)
        probs = pd.DataFrame(probs, columns=ys.columns, index=ys.index)
        prob_ranks = probs.rank(axis=1).mean(axis=0).sort_values(ascending=False).index
        cols = list(ys.columns)
        cols = dict(map(lambda x: (cols.index(x), x), cols))
        actual_ranks = ys.apply(lambda x: list(x).index(max(x)), axis=1).map(lambda x: cols[x]).value_counts().index
        a = range(len(actual_ranks))
        b = list(map(lambda x: list(actual_ranks).index(x), list(prob_ranks)))
        self.assertEquals(a[0], b[0])
        rank_corr = pd.DataFrame([a, b]).T.corr(method='spearman').iloc[0,1]
        self.assertTrue(rank_corr > .5)
        
    def test_regularization_decreases_ability_to_fit_train_set(self):
        ys = self.ys_labels.values.astype(np.float32)
        res = []
        for penalty_alpha in (0., .5, 2.):
            _, _, stats = md.train_nn_softmax(ys, ys, [2], 1000, 500, .1, penalty_alpha=penalty_alpha)
            res.append(1. - stats['accuracy'])
        self.assertTrue(all(res[i] <= res[i+1] for i in xrange(len(res)-1)))

    def test_fc_model_trains_w_mult_structue_input_types(self):
        inp = self.ys_labels
        Xs, ys = mi.validate_and_format_Xs_ys(inp, inp)
        for structure in ([[],[]], [], [2]):
            try:
                _, _, _ = md.train_nn_softmax(Xs.values, ys.values, structure, 100, 100, .1)
            except:
                raise Exception('train model didnt like input: {0}'.format(structure))

    def test_conv_layer_always_covers_entire_input_row(self):
        Xs_shape = [None] + list(self.Xs_conv.values.shape[1:])
        x = tf.placeholder(tf.float32, Xs_shape)
        x_4d = tf.reshape(x, [-1, Xs_shape[1], Xs_shape[2], 1])
        for conv_struct in ([2,], [2,4]):    
            layer_defs = map(lambda x: ('name', x), conv_struct)          
            layer = reduce(lambda inp, ld: md.create_conv_layer(inp, ld[0], ld[1]), layer_defs, x_4d)
            init_fc_layer = md.flatten_conv_layer(layer)
            exp_shape = Xs_shape[1] * conv_struct[-1]
            act_shape = init_fc_layer._shape[1]._value
            self.assertEquals(exp_shape, act_shape)
    
    # def test_conv_layer_outperforms_when_inputs_have_shared_structure(self):
    #     ys = self.ys_labels_conv
    #     Xs = mi.flatten_panel(self.Xs_conv)
    #     _, _, stats = md.train_nn_softmax(Xs.values, ys.values, [2], 1000, 100, .1)
    #     print stats



