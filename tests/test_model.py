import unittest
import datetime

import numpy as np
import pandas as pd

import model_inputs as mi
import model as md


class TestModel(unittest.TestCase):
    
    def setUp(self):
        np.random.seed(0)
        dti = pd.DatetimeIndex(start='2000-1-1', freq='B', periods=1000)
        self.Xs = self.gen_random_normal(dti, 20)
        self.ys_probs = self.gen_random_probs(dti, 10)
        self.ys_labels = self.gen_random_onehot(dti, 10)
    
    def gen_random_normal(self, date_index, width):
        length = date_index.shape[0]
        return pd.DataFrame(np.random.randn(length, width), index=date_index) 

    def gen_random_probs(self, date_index, width):
        length = date_index.shape[0]
        df = pd.DataFrame(np.random.rand(length, width), index=date_index) 
        return df.div(df.sum(axis=1), axis=0)

    def gen_random_onehot(self, date_index, width):
        df = self.gen_random_probs(date_index, width)
        df_maxs = df.apply(lambda x: list(x).index(max(x)), axis=1)
        return pd.get_dummies(df_maxs)

    def test_input_output_are_same_shape(self):
        inp = self.ys_labels.values
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
        inp = self.ys_labels.values
        _, _, stats = md.train_nn_softmax(inp, inp, [], 500, 100, .1)
        self.assertTrue(stats['accuracy'] > .99)

    def test_hidden_layers_softmax_learns_onehot_w_perfect_foresight(self):
        inp = self.ys_labels.values
        _, _, stats  = md.train_nn_softmax(inp, inp, [10,], 1000, 100, .1)
        self.assertTrue(stats['accuracy'] > .99)

    def test_single_softmax_learns_opt_weights_w_perfect_foresight(self):
        ys = pd.read_csv('tests/test_data/opt_weights_20.csv', index_col=0, parse_dates=['Date',])
        probs, _, _ = md.train_nn_softmax(ys.values, ys.values, [], 2000, 1000, .4)
        probs = pd.DataFrame(probs, columns=ys.columns, index=ys.index)
        self.assertTrue(probs.stack().corr(ys.stack()) > .95)        