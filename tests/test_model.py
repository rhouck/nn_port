import unittest
import datetime

import numpy as np
import pandas as pd

import model_inputs as mi
import model as md
import tensorflow as tf
from gen_data import *


class TestFCModel(unittest.TestCase):
    
    def setUp(self):
        np.random.seed(0)
        self.dti = pd.DatetimeIndex(start='2000-1-1', freq='B', periods=1000)
        self.ys_probs = gen_random_probs(self.dti, 10)
        self.ys_labels = gen_random_onehot(self.dti, 10)
        Xs = gen_random_normal(self.dti, 20)
        self.Xs, _ = mi.validate_and_format_Xs_ys(Xs, self.ys_probs)

    def test_input_output_are_same_shape(self):
        inp = self.ys_labels.astype(np.float32).values
        preds, _ = md.train_nn_softmax([inp], [inp], [], 50, 10, .1)
        probs = preds['train']['weights']
        labels = preds['train']['labels']
        self.assertEquals(probs.shape, inp.shape)
        self.assertEquals(labels.shape[0], inp.shape[0])

    def test_random_inputs_creates_equal_weights(self):
        Xs = self.Xs.values
        for ys in (self.ys_probs.values, self.ys_labels.values):
            preds, _ = md.train_nn_softmax([Xs], [ys], [], 1000, 100, .1)
            probs = preds['train']['weights']
            probs = pd.DataFrame(probs)
            equal_weight = 1. / probs.shape[1]
            dist_from_equal = (probs.mean() - equal_weight).abs()
            short_dist = (dist_from_equal < .05)
            self.assertTrue(short_dist.all())

    def test_single_softmax_learns_onehot_w_perfect_foresight(self):
        inp = self.ys_labels.astype(np.float32).values
        _, stats = md.train_nn_softmax([inp], [inp], [], 500, 100, .1)
        self.assertTrue(stats['train']['accuracy'] > .99)

    def test_hidden_layers_softmax_learns_onehot_w_perfect_foresight(self):
        inp = self.ys_labels.astype(np.float32).values
        _, stats  = md.train_nn_softmax([inp], [inp], [10,], 1000, 100, .1)
        self.assertTrue(stats['train']['accuracy'] > .99)

    def test_single_softmax_learns_opt_weights_w_perfect_foresight(self):
        ys = pd.read_csv('tests/test_data/opt_weights_20.csv', index_col=0, parse_dates=['Date',])
        inp = ys.astype(np.float32).values
        preds, _ = md.train_nn_softmax([inp], [inp], [], 2000, 1000, .4)
        probs = preds['train']['weights']
        probs = pd.DataFrame(probs, columns=ys.columns, index=ys.index)
        self.assertTrue(probs.stack().corr(ys.stack()) > .95) 

    def test_noise_input_leads_to_stable_label_pred_based_on_max_val_freqs(self):
        ys = pd.read_csv('tests/test_data/opt_weights_20.csv', index_col=0, parse_dates=['Date',])
        Xs = gen_random_normal(ys.index, 20).astype(np.float32)     
        preds, _ = md.train_nn_softmax([Xs.values], [ys.values], [], 1000, 100, .1)
        probs = preds['train']['weights']
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
        inp = self.ys_labels.astype(np.float32).values
        res = []
        for penalty_alpha in (0., .5, 2.):
            _, stats = md.train_nn_softmax([inp], [inp], [2], 1000, 500, .1, penalty_alpha=penalty_alpha)
            res.append(1. - stats['train']['accuracy'])
        self.assertTrue(all(res[i] <= res[i+1] for i in xrange(len(res)-1)))

    def test_fc_model_trains_w_mult_structue_input_types(self):
        inp = self.ys_labels
        Xs, ys = mi.validate_and_format_Xs_ys(inp, inp)
        for structure in ([[],[]], [], [2]):
            try:
                _, _, = md.train_nn_softmax([Xs.values], [ys.values], structure, 100, 100, .1)
            except:
                raise Exception('train model didnt like input: {0}'.format(structure))

    def tests_results_dif_and_non_nan_for_train_and_stats(self):
        Xs = self.Xs.astype(np.float32)
        ys = self.ys_labels.astype(np.float32)
        train, test = mi.split_inputs_by_date(Xs, ys, datetime.date(2003,1,1), buffer_days=0)
        Xs_train, ys_train = train
        Xs_test, ys_test = test
        X_inps = [Xs_train.values, Xs_test.values]
        y_inps = [ys_train.values, ys_test.values]
        _, stats = md.train_nn_softmax(X_inps, y_inps, [], 500, 500, .1)
        for i in ('accuracy', 'cross_entropy'):
            self.assertNotEqual(stats['train'][i], stats['test'][i])
    
    
class TestConvModel(unittest.TestCase):
    
    def setUp(self):
        np.random.seed(0)
        self.dti = pd.DatetimeIndex(start='2000-1-1', freq='B', periods=1000)
        Xs_conv, ys_labels_conv, self.true_weights = gen_2d_random_Xs_onehot_ys_from_random_kernel(self.dti, num_classes=15, num_features=10, noise_sigma=10.)
        self.Xs_conv, self.ys_labels_conv = mi.validate_and_format_Xs_ys(Xs_conv, ys_labels_conv)

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

    def test_model_approx_max_potential_accuracy_in_test_set(self):
        max_accuracy = check_kernel_predictive_accuracy(self.Xs_conv, self.ys_labels_conv, self.true_weights)
        _, stats = md.train_nn_softmax([self.Xs_conv.values], [self.ys_labels_conv.values], [[1],[]], 1000, 100, .5)
        self.assertTrue(abs(stats['train']['accuracy'] - max_accuracy) < .1)

    def test_conv_layer_outperforms_in_test_set_when_inputs_have_shared_structure(self):
        ys = self.ys_labels_conv
        Xs = self.Xs_conv
        train, test = mi.split_inputs_by_date(Xs, ys, datetime.date(2003,1,1), buffer_days=0)
        Xs_train, ys_train = train
        Xs_test, ys_test = test
        Xs_train_f = mi.flatten_panel(Xs_train).astype(np.float32).values
        Xs_test_f = mi.flatten_panel(Xs_test).astype(np.float32).values
        Xs_train = Xs_train.astype(np.float32).values
        Xs_test = Xs_test.astype(np.float32).values  
        ys_train = ys_train.values
        ys_test = ys_test.values
        _, conv_stats = md.train_nn_softmax([Xs_train, Xs_test], [ys_train, ys_test], [[1],[]], 1000, 100, .1)
        _, fc_stats = md.train_nn_softmax([Xs_train_f, Xs_test_f], [ys_train, ys_test], [[],[]], 1000, 100, .1)
        self.assertTrue(conv_stats['train']['cross_entropy'] > fc_stats['train']['cross_entropy'])
        self.assertTrue(conv_stats['test']['cross_entropy'] < fc_stats['test']['cross_entropy'])