import unittest

import numpy as np
from lifelines.datasets import load_rossi
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt

from rafael.usecases import Standadization, KaplanMeier, Output
from rafael.logger import setup_logger

setup_logger()


def _standardization(usecase, Xs):
    sum_, count_ = list(zip(*map(lambda x: usecase.local_col_sum(x), Xs)))
    mean_, _ = usecase.global_mean(sum_, count_)
    Xs, ssq_, count_ = list(zip(*map(lambda x: usecase.local_ssq(x, mean_), Xs)))
    var_, delete_ = usecase.global_var(ssq_, count_)
    Xs = list(map(lambda x: usecase.local_standardize(x, var_, delete_), Xs))
    return Xs, var_

def _std_km(usecase, Xs, ys, n_std, var):
    grouped_y = []
    for i in range(len(Xs)):
        y_, n_std_ = usecase.local_group_by_std(Xs[i], n_std, ys[i])
        grouped_y.append(y_)
    n_std = n_std_

    fitted_, logrank_stats_ = usecase.global_fit_model(grouped_y)
    return fitted_, logrank_stats_, n_std


class KaplanMeierTestCase(unittest.TestCase):
    def setUp(self):
        self.std = Standadization()
        self.km = KaplanMeier()
        self.output = Output()
        
        self.data = load_rossi()
        self.cont_features = ['age', 'prio']
        self.cat_features = ['mar', 'race']
        self.y = self.data.loc[:, ['week', 'arrest']].to_numpy()

        self.n_chunk = 4
        self.chunk_size = self.data.shape[0]//self.n_chunk
        self.ys = [self.y[i*self.chunk_size:(i+1)*self.chunk_size, :] for i in range(self.n_chunk)]
    
    def test_standardization(self):
        X = self.data.loc[:, self.cont_features].to_numpy()
        Xs = [X[i*self.chunk_size:(i+1)*self.chunk_size, :] for i in range(self.n_chunk)]
        
        ansX = np.concatenate(Xs)
        ansX = (ansX-np.mean(ansX, axis=0)) / np.std(ansX, axis=0)
        
        Xs, _ = _standardization(self.std, Xs)
        res = np.concatenate(Xs)
        
        np.testing.assert_array_almost_equal(res, ansX, decimal=2)
        
    def test_km_group_by_soft_std(self):
        # This test performs on soft filtering,
        # thus, not dealing with missing sample condition.
        X = self.data.loc[:, self.cont_features].to_numpy()
        Xs = [X[i*self.chunk_size:(i+1)*self.chunk_size, :] for i in range(self.n_chunk)]
        n_std = 0.2
        
        ansX = (X-np.mean(X, axis=0)) / np.std(X, axis=0)
        kmf = KaplanMeierFitter()
        for i in range(ansX.shape[1]):
            ans_ax = plt.subplot(111)
            upper_mask = ansX[:, i] >= n_std
            lower_mask = ansX[:, i] <= -n_std
            kmf.fit(self.y[upper_mask][:, 0], self.y[upper_mask][:, 1], label=self.cont_features[i])
            kmf.plot_survival_function(ax=ans_ax, figsize=(8,6))

            kmf.fit(self.y[lower_mask][:, 0], self.y[lower_mask][:, 1], label=self.cont_features[i])
            kmf.plot_survival_function(ax=ans_ax, figsize=(8,6))
            plt.savefig(f'/tmp/ans.{self.cont_features[i]}.png', dpi=200)
            plt.close()
        
        Xs, var = _standardization(self.std, Xs)
        fitted, logrank_stats, n_std = _std_km(self.km, Xs, self.ys, n_std, var)
        self.output.kaplan_meier_results(self.cont_features, fitted, logrank_stats, n_std, '/tmp')
        
    def test_km_group_by_hard_std(self):
        # This test performs on hard filtering,
        # thus, dealing with missing sample condition.
        X = self.data.loc[:, self.cont_features].to_numpy()
        Xs = [X[i*self.chunk_size:(i+1)*self.chunk_size, :] for i in range(self.n_chunk)]
        n_std = 2
        
        Xs, var = _standardization(self.std, Xs)
        fitted, logrank_stats, n_std = _std_km(self.km, Xs, self.ys, n_std, var)
        self.output.kaplan_meier_results(self.cont_features, fitted, logrank_stats, n_std, '/tmp')
        
    def test_km_group_by_std_with_multi_weights(self):
        X = self.data.loc[:, self.cont_features].to_numpy()
        Xs = [X[i*self.chunk_size:(i+1)*self.chunk_size, :] for i in range(self.n_chunk)]
        n_std = [0.2, 2]
        
        Xs, var = _standardization(self.std, Xs)
        fitted, logrank_stats, n_std = _std_km(self.km, Xs, self.ys, n_std, var)
        self.output.kaplan_meier_results(self.cont_features, fitted, logrank_stats, n_std, '/tmp')