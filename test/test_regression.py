import os
import shutil
import unittest

import numpy as np
import scipy.stats as stats
import pandas as pd
try:
    from sklearn.linear_model import LinearRegression as sklearn_linear
    from sklearn.linear_model import LogisticRegression as sklearn_logistic
    scikit_learn_exists = True
except ImportError:
    scikit_learn_exists = False
from lifelines.datasets import load_rossi
from lifelines import CoxPHFitter

from rafael.usecases import LinearRegression, LogisticRegression, CoxPHRegression
from rafael.logger import setup_logger

setup_logger()

def _linear_t_stats(sse, XtX, dof, beta):
    mse = sse / dof
    vars = mse * np.linalg.inv(XtX).diagonal()
    std = np.sqrt(vars)
    t_stat = beta / std
    return t_stat


class LinearRegressionTestCase(unittest.TestCase):
    def setUp(self):
        n_samples, n_features = 20, 8
        self.X = np.random.randn(n_samples, n_features)
        self.y = np.random.randn(n_samples,)
        
        # Add bias and solve Xty=XtX.beta
        bX = np.concatenate([np.ones((self.X.shape[0], 1)), self.X], axis=1)
        self.beta = np.linalg.solve(
            bX.T @ bX,
            bX.T @ self.y
        )
        eps = self.y - bX @ self.beta
        
        # Calculate the statistics
        dof = bX.shape[0]-bX.shape[1]
        self.t_stats = _linear_t_stats(
            sse=eps @ eps,
            XtX=bX.T @ bX,
            dof=dof,
            beta=self.beta
        )
        self.pvals = 2 * (1 - stats.t.cdf(np.abs(self.t_stats), dof))
        self.regression = LinearRegression()
    
    def test_linear_regression(self):
        # calculate XtX and Xty
        XtX, Xty, X = self.regression.local_calculate_covariances(
            self.X, self.y
        )

        # calculate beta
        beta = self.regression.global_fit_model([XtX,], [Xty,])
        
        # calculate SSE and number of observations
        sse, n_obs = self.regression.local_sse_and_obs(X, self.y, beta)

        # calculate t-statistics and p-values
        t_stats, pvals = self.regression.global_stats([sse,], [n_obs,])
        
        np.testing.assert_array_almost_equal(self.beta, beta, decimal=4)
        np.testing.assert_array_almost_equal(self.t_stats, t_stats, decimal=4)
        np.testing.assert_array_almost_equal(self.pvals, pvals, decimal=4)
        
        if scikit_learn_exists:
            model = sklearn_linear().fit(self.X, self.y)
            np.testing.assert_array_almost_equal(self.beta[1:], model.coef_, decimal=4)
            np.testing.assert_array_almost_equal(self.beta[0], model.intercept_, decimal=4)


def _converged(prev_loglikelihood, loglikelihood, threshold=1e-4):
    if np.isnan(prev_loglikelihood).all():
        return False
    else:
        delta_loglikelihood = np.abs(prev_loglikelihood - loglikelihood)
        return True if (delta_loglikelihood < threshold).all() else False

def _logistic_t_stats(beta, inv_hessian):
    std = np.sqrt(inv_hessian.diagonal())
    t_stat = beta/std
    return t_stat


class LogisticRegressionTestCase(unittest.TestCase):
    def setUp(self):
        n_samples, n_features = 20, 8
        self.X = np.random.randn(n_samples, n_features)
        self.y = np.random.randint(2, size=n_samples)
        self.max_iterations = 50
        current_iteration = 1
        
        # Add bias and initialize parameters
        bX = np.concatenate([np.ones((self.X.shape[0], 1)), self.X], axis=1)
        beta = np.zeros((n_features+1,))
        prev_loglikelihood = np.nan
        eps = np.finfo(float).eps
        
        while current_iteration < self.max_iterations:
            pred_y = 1 / (1 + np.exp(-(bX @ beta)))
            gradient = bX.T @ (self.y - pred_y)
            hessian = np.multiply(bX.T, (pred_y * (1 - pred_y)).T) @ bX
            loglikelihood = np.sum(
                self.y * np.log(pred_y + eps) + 
                (1 - self.y) * np.log(1 - pred_y + eps)
            )
            beta += np.linalg.solve(hessian, gradient)

            print(f"setUp I: {current_iteration} | L: {loglikelihood:.3f}")
            if _converged(prev_loglikelihood, loglikelihood):
                break
            else:
                prev_loglikelihood = loglikelihood
                current_iteration += 1

        self.beta = beta
        self.t_stats = _logistic_t_stats(beta, np.linalg.inv(hessian))
        self.pvals = 1 - stats.chi2.cdf(np.square(self.t_stats), 1)
        self.regression = LogisticRegression()
        
    def test_auto_binarize(self):
        X = np.random.randn(5, 2)
        y = np.array([0, -1, 0, 0, -1])
        ans = np.array([1, 0, 1, 1, 0])
        result = self.regression.local_init_params(X, y)[1]
        np.testing.assert_array_equal(ans, result)
        
    def test_logistic_regression(self):
        # Initialize parameters
        X, y, gradient, hessian, loglikelihood, current_iteration = self.regression.local_init_params(self.X, self.y)
        print(f"Test I: {current_iteration} | L: {loglikelihood:.3f}")
        jump_to = None
        prev_loglikelihood = np.nan
        prev_beta = None
        
        while True:
            # update beta
            beta, prev_beta, prev_loglikelihood, inv_hessian, jump_to = self.regression.global_params(
                [gradient, ], [hessian, ], [loglikelihood, ], 
                current_iteration, self.max_iterations,
                prev_loglikelihood, prev_beta
            )

            if jump_to == 'global_stats':
                break
            else:
                # update gradient, hessian, loglikelihood
                gradient, hessian, loglikelihood, current_iteration, jump_to = self.regression.local_iter_params(
                    X, y, beta, current_iteration
                )
                print(f"Test I: {current_iteration} | L: {loglikelihood:.3f}")
        
        t_stats, pvals, beta = self.regression.global_stats(beta, inv_hessian)
        
        np.testing.assert_array_almost_equal(self.beta, beta, decimal=4)
        np.testing.assert_array_almost_equal(self.t_stats, t_stats, decimal=4)
        np.testing.assert_array_almost_equal(self.pvals, pvals, decimal=4)
        
        if scikit_learn_exists:
            model = sklearn_logistic(
                penalty=None,
                max_iter=self.max_iterations
            ).fit(self.X, self.y)
            np.testing.assert_array_almost_equal(self.beta[1:], model.coef_[0], decimal=3)
            np.testing.assert_array_almost_equal(self.beta[0], model.intercept_[0], decimal=3)
            
            
class CoxPHRegressionTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_rossi()
        cls.data.columns = ['time', 'event']+list(cls.data.columns[2:])
        training_size = 400
        cls.ans = CoxPHFitter()
        cls.ans.fit(cls.data.iloc[0:training_size, :], event_col='event', duration_col='time')
        cls.keep_feature_cols = list(cls.data.columns[2:])
        
        # Generate testing data
        cls.test_dir = '/tmp/_test_cox'
        cls.file_paths = []
        if not os.path.exists(cls.test_dir):
            os.mkdir(cls.test_dir)
        idx = np.random.choice(training_size, training_size, replace=False)
        cls.n_chunk = 4
        chunk_size = len(idx)//cls.n_chunk
        for i in range(cls.n_chunk):
            X_ = cls.data.iloc[i*chunk_size:(i+1)*chunk_size, 2:]
            y_ = cls.data.iloc[i*chunk_size:(i+1)*chunk_size, 0:2]
            data_ = pd.concat([X_, y_], axis=1)
            filepath = f'{cls.test_dir}/client{i+1}.clinical'
            data_.to_csv(filepath, index=None)
            cls.file_paths.append(filepath)
            
        cls.dccox = CoxPHRegression()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_dir)
        
    def test_missing_time_and_event(self):
        fake = pd.DataFrame(np.random.randn(4,5), columns=['time']+[f'f{i}' for i in range(4)])
        filepath = f'{self.test_dir}/fake.clinical'
        fake.to_csv(filepath, index=None)
        
        try:
            self.dccox.local_load_metadata(filepath, ['f1', 'f2'])
        except AssertionError as msg:
            assert str(msg) == "Missing columns ['event']."
        
    def test_drop_missings(self):
        fake = np.array(
            [
                [12, 0, np.nan, 1, 2, 3],
                [13, 1, 4, 5, 6, np.nan],
                [11, 1, 7, np.nan, 8, 9],
                [9, 0, 10, 11, np.nan, 12]
            ]
        )
        fake = pd.DataFrame(fake, columns=['time','event']+[f'f{i}' for i in range(4)])
        filepath = f'{self.test_dir}/fake2.clinical'
        fake.to_csv(filepath, index=None)
        
        X, y, keep_feature_cols, _ = self.dccox.local_load_metadata(filepath, ['f0', 'f1', 'f2', 'f3'])
        np.testing.assert_array_equal(X, np.array([]).reshape(-1,4))
        np.testing.assert_array_equal(y, np.array([]).reshape(-1,2))
        
        Xanc = self.dccox.global_create_Xanc(len(['f0', 'f1', 'f2', 'f3']))
        F, X_tilde, Xanc_tilde, sum_ = self.dccox.local_create_proxy_data(X, Xanc, y)
        
        assert F is None
        assert X_tilde == [None]
        assert Xanc_tilde == [None]
        
    def test_dccox(self):
        # Generate the global anchor matrix
        Xanc = self.dccox.global_create_Xanc(len(self.keep_feature_cols))
        
        # Generate projected proxy matrix 
        Xs_tilde, Xancs_tilde, Fs, ys, sums = [], [], [], [], []
        for i in range(self.n_chunk):
            X, y, keep_feature_cols, _ = self.dccox.local_load_metadata(self.file_paths[i], self.keep_feature_cols)
            F, X_tilde, Xanc_tilde, sum_ = self.dccox.local_create_proxy_data(X, Xanc, y)
            Xs_tilde.append(X_tilde)
            Xancs_tilde.append(Xanc_tilde)
            Fs.append(F)
            ys.append(y)
            sums.append(sum_)
        # Perform cox ph regression
        coef, coef_var, baseline_hazard, mean = self.dccox.global_fit_model(Xs_tilde, Xancs_tilde, ys, sums)
        
        # Recover survival function
        surv_func = self.dccox.local_recover_survival(keep_feature_cols, coef[0][0], coef_var[0][0], baseline_hazard, mean, Fs[0])
    
        # baseline hazard
        pd.testing.assert_frame_equal(surv_func.baseline_hazard, self.ans.baseline_hazard_)
        
        # coef
        np.testing.assert_array_almost_equal(surv_func.coef.to_numpy(), self.ans.summary['coef'].to_numpy())
        
        # var-cov matrix
        np.testing.assert_array_almost_equal(np.diag(surv_func.coef_var), np.diag(self.ans.variance_matrix_))
        
        # all statistical variables
        np.testing.assert_array_almost_equal(surv_func.summary.iloc[:,0:10], self.ans.summary.iloc[:,[i for i in range(11) if i != 7]])
        
        # cumulative hazard
        pd.testing.assert_frame_equal(surv_func.predict_cumhazard(self.data), self.ans.predict_cumulative_hazard(self.data))
        
        # survival probability
        pd.testing.assert_frame_equal(surv_func.predict_survival(self.data), self.ans.predict_survival_function(self.data))
        
        # expected survival days
        np.testing.assert_array_almost_equal(surv_func.predict_expectation(self.data), self.ans.predict_expectation(self.data).values)
        
    def test_dccox_with_missing(self):
        # Generate missing data
        sim_missing = self.data.copy()
        sim_missing['mar'] = np.nan
        filepath = f'{self.test_dir}/missing.clinical'
        sim_missing.to_csv(filepath, index=None)
        
        # Generate the global anchor matrix
        Xanc = self.dccox.global_create_Xanc(len(self.keep_feature_cols))
        
        # Generate projected proxy matrix 
        Xs_tilde, Xancs_tilde, Fs, ys, sums = [], [], [], [], []
        file_paths = self.file_paths+[filepath]
        for i in range(self.n_chunk+1):
            X, y, keep_feature_cols, _ = self.dccox.local_load_metadata(file_paths[i], self.keep_feature_cols)
            F, X_tilde, Xanc_tilde, sum_ = self.dccox.local_create_proxy_data(X, Xanc, y)
            Xs_tilde.append(X_tilde)
            Xancs_tilde.append(Xanc_tilde)
            Fs.append(F)
            ys.append(y)
            sums.append(sum_)
        # Perform cox ph regression
        coef, coef_var, baseline_hazard, mean = self.dccox.global_fit_model(Xs_tilde, Xancs_tilde, ys, sums)
        
        # Recover survival function
        surv_func = self.dccox.local_recover_survival(keep_feature_cols, coef[0][0], coef_var[0][0], baseline_hazard, mean, Fs[0])
    
        # baseline hazard
        pd.testing.assert_frame_equal(surv_func.baseline_hazard, self.ans.baseline_hazard_)
        
        # coef
        np.testing.assert_array_almost_equal(surv_func.coef.to_numpy(), self.ans.summary['coef'].to_numpy())
        
        # var-cov matrix
        np.testing.assert_array_almost_equal(np.diag(surv_func.coef_var), np.diag(self.ans.variance_matrix_))
        
        # all statistical variables
        np.testing.assert_array_almost_equal(surv_func.summary.iloc[:,0:10], self.ans.summary.iloc[:,[i for i in range(11) if i != 7]])
        
        # cumulative hazard
        pd.testing.assert_frame_equal(surv_func.predict_cumhazard(self.data), self.ans.predict_cumulative_hazard(self.data))
        
        # survival probability
        pd.testing.assert_frame_equal(surv_func.predict_survival(self.data), self.ans.predict_survival_function(self.data))
        
        # expected survival days
        np.testing.assert_array_almost_equal(surv_func.predict_expectation(self.data), self.ans.predict_expectation(self.data).values)
