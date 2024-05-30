from functools import reduce

import numpy as np
import pandas as pd
import scipy as sp
from scipy.integrate import trapz
from lifelines import CoxPHFitter

from .block import BlockMatrix


class DCCoxDataProjector:
    def __init__(self, k=10, bs_prop=0.6, bs_times=20, alpha=0.05, step_size=0.5):
        """
        Project data for DC-Cox PH Regression.

        Parameters
        ----------
            k : int
                The latent dimension of Fdr. Notice that the ultimate dimension is min(k, len(S)),
                where the S is the number of nonzero singular values.
            bs_prop : float
                The proportion of the samples in a client used for bootstrapping for each time.
            bs_times : int
                The number of times to bootstrap.
            alpha : float
                The level in the confidence intervals.
                It is `alpha` parameter in `lifelines.fitters.coxph_fitter.CoxPHFitter`.
            step_size : float
                Deal with the fitting error, `delta contains nan value(s)`.

        See also
        --------
        `lifelines.fitters.coxph_fitter.CoxPHFitter`
        https://lifelines.readthedocs.io/en/latest/Examples.html#problems-with-convergence-in-the-cox-proportional-hazard-model
        DC-COX[https://doi.org/10.1016/j.jbi.2022.104264]
        """
        self.k = k
        self.bs_prop = bs_prop
        self.bs_times = bs_times
        self.fitter_params = {"alpha": alpha}
        self.fitting_params = {
            "duration_col": "duration",
            "event_col": "event",
            "fit_options": {"step_size": step_size},
        }

    @property
    def F(self):
        return self.__F

    @property
    def X_tilde(self):
        return self.__X_tilde

    @property
    def Xanc_tilde(self):
        return self.__Xanc_tilde

    def _compute_Fbs(self, X, events, durations):
        """
        Apply the Cox PH model to X, events and durations to obtain coefficients.

        Parameters
        ----------
            X : np.array
                The feature matrix with shape (n_samples, n_features).
            events : np.array
                The one-hot event vector with shape (n_samples,).
            durations : np.array
                The one-hot duration vector with shape (n_samples,).

        Returns
        -------
            Fbs : np.array
                The concatenated betas from each bootstrap regression with shape (n_features, n_times).
        """
        coefs = []
        n_samples = X.shape[0]
        X = pd.DataFrame(X)
        y = pd.DataFrame({"duration": durations, "event": events})
        data = pd.concat([X, y], axis=1)
        for _ in range(self.bs_times):
            idx = np.random.choice(
                n_samples, int(self.bs_prop * n_samples), replace=False
            )
            model = CoxPHFitter(**self.fitter_params)
            model.fit(data.iloc[idx, :], **self.fitting_params)
            coef_ = model.summary["coef"].to_numpy()
            coefs.append(coef_.reshape(1, -1))
        Fbs = np.concatenate(coefs).T
        return Fbs

    def _compute_Fdr(self, X):
        """
        Compute the PCs (samples as features).

        Parameters
        ----------
            X : np.array
                The feature matrix with shape (n_samples, n_features).

        Returns
        -------
            Fdr : np.array
                The PCs with shape (n_features, min(k, len(non-zero singular values))).
        """
        X_shift = X.T - np.mean(X, axis=1)  # Make the row mean to 0
        _, S, Vh = np.linalg.svd(X_shift)
        k = min(self.k, len(S))
        return X_shift @ Vh.T[:, :k]

    @staticmethod
    def _create_random_nonsingular(m_tilde):
        E = np.random.rand(m_tilde, m_tilde)
        Ex = np.sum(np.abs(E), axis=1)
        np.fill_diagonal(E, Ex)
        return E

    def _compute_F(self, X, events, durations):
        # n_features x mbs(bs_times)
        Fbs = self._compute_Fbs(X, events, durations)
        # n_features x mdr(min(k, len(non-zero singular values)))
        Fdr = self._compute_Fdr(X)
        # n_features x m tilde(mbs+mdr)
        F_ = np.concatenate([Fbs, Fdr], axis=1)
        # m tilde x m tilde
        E = self._create_random_nonsingular(F_.shape[1])
        # n_features x m tilde
        self.__F = F_ @ E

    def project(self, X, Xanc, events, durations):
        """
        Perform the linear transformation.
        X tilde = X @ F, where X shape is (nc, md).
        Xanc tilde = Xanc @ F, where Xanc shape is (r, md).
        F = [Fbs, Fdr], where Fbs shape is (md, bs_times)
        and Fdr shape is (md, min(k, len(non-zero singular values))).

        Parameters
        ----------
            X : np.array
                The feature matrix with shape (nc, md),
                where nc is the number of samples in Xc,: and md is the number of features in X:,d.
            Xanc : np.array
                The global anchor matrix with shape (r, md).
                where r is the pseudo-number of samples of Xanc and md is the number of features in X:,d.
            events : np.array
                The one-hot event vecotr with shape (nc,).
            durations : np.array
                The time vector with shape (nc,).
        """
        self._compute_F(X, events, durations)
        # n_samples x m tilde
        self.__X_tilde = X @ self.__F
        # r x m tilde
        self.__Xanc_tilde = Xanc @ self.__F


class DCCoxRegressor:
    def __init__(self, alpha=0.05, step_size=0.5):
        """
        Perform DC-Cox PH Regression.

        Parameters
        ----------
            alpha : float
                The level in the confidence intervals.
                It is `alpha` parameter in `lifelines.fitters.coxph_fitter.CoxPHFitter`.
            step_size : float
                Deal with the fitting error, `delta contains nan value(s)`.

        See also
        --------
        `lifelines.fitters.coxph_fitter.CoxPHFitter`
        https://lifelines.readthedocs.io/en/latest/Examples.html#problems-with-convergence-in-the-cox-proportional-hazard-model
        DC-COX[https://doi.org/10.1016/j.jbi.2022.104264]
        """
        self.fitter_params = {"alpha": alpha}
        self.fitting_params = {
            "duration_col": "duration",
            "event_col": "event",
            "fit_options": {"step_size": step_size},
        }

    @property
    def coef(self):
        return self.__coef

    @property
    def coef_var(self):
        return self.__coef_var

    @property
    def baseline_hazard(self):
        return self.__baseline_hazard

    @property
    def Gs(self):
        return self.__Gs

    @staticmethod
    def _compute_target_matrix(Xancs_tilde: BlockMatrix):
        """
        Compute the target matrix Gs.
        [Xancs_tilde_1, ..., Xancs_tilde_c] = U * S @ Vh.
        Gc = pinv(Xanc_tilde_c) @ U @ C, where C is the nonsingular matrix.

        Parameters
        ----------
            Xancs_tilde : BlockMatrix
                The `BlockMatrix` with shape (c, d).

        Returns
        -------
            Gs : BlockMatrix
                The `BlockMatrix(axis=0)`.
                For each G in BlockMatrix, the shape is (mc tilde, m hat).
        """
        # r x (c * m tilde), where m tilde equals to the sum of Fbs.shape[1]+Fdr.shape[1] for all c.
        # e.g. c1d1 = 20+4, c1d2 = 20+3, m tilde = 47
        Xancs_ = np.concatenate(
            [Xancs_tilde[c, :] for c in range(Xancs_tilde.shape[0])], axis=1
        )

        # U: (r x m hat), m hat equals to len(S)
        U, S, _ = np.linalg.svd(Xancs_)
        P = U[:, : len(S)] * S

        Gs = []
        for c in range(Xancs_tilde.shape[0]):
            # Gc: m tilde x m hat
            Gc = np.linalg.pinv(Xancs_tilde[c, :]) @ P
            idxs = np.cumsum([0] + Xancs_tilde.dsizes)
            # Gcd: mc tilde x m hat
            Gs.append([Gc[idxs[i] : idxs[i + 1], :] for i in range(len(idxs) - 1)])

        return BlockMatrix(Gs, axis=0)

    @staticmethod
    def _drop_low_var_cols(X_hat, Gs):
        vars_ = np.var(X_hat, axis=0)
        keeps_ = vars_ > 0.01
        X_hat = X_hat[:, keeps_]
        for c in range(Gs.shape[0]):
            for d in range(Gs.shape[1]):
                Gs.blocks[c][d] = Gs.blocks[c][d][:, keeps_]
        return X_hat, Gs

    def _cox_regression(self, Xs_tilde, Gs, durations, events):
        """
        Perform Cox-PH regression on X hat.
        X hat = concat([Xc_tilde @ Gc]).
        CoxPHFitter().fit(X hat, y).

        Parameters
        ----------
            Xs_tilde : BlockMatrix
                The proxy data with shape (c, d).
            Gs : BlockMatrix
                The `BlockMatrix(axis=0)`.
                For each G in BlockMatrix, the shape is (mc tilde, m hat).
            durations : int
                The duration time vector with shape (n,).
            event : int
                The one-hot event vector with shape (n,).
                e.g. 1 for dead, 0 for survived.
        """
        X_hat = np.concatenate(
            [Xs_tilde[c, :] @ Gs[c, :] for c in range(Xs_tilde.shape[0])]
        )
        X_hat, Gs = self._drop_low_var_cols(X_hat, Gs)
        X_hat = pd.DataFrame(X_hat)
        y = pd.DataFrame({"duration": durations, "event": events})
        model = CoxPHFitter(**self.fitter_params)
        model.fit(pd.concat([X_hat, y], axis=1), **self.fitting_params)
        self.__baseline_hazard = model.baseline_hazard_
        self.__coef = model.summary["coef"].to_numpy()
        self.__coef_var = model.variance_matrix_.to_numpy()
        self.__Gs = Gs

    def fit(self, Xs_tilde: BlockMatrix, Xancs_tilde: BlockMatrix, durations, events):
        """
        Calculates the target matrix G and
        performs the Cox-PH regression.
        .. math::
            \begin{align*}
            &\left[\tilde{X}_1^{\mathrm{anc}}, \tilde{X}_2^{\mathrm{anc}},...,\tilde{X}_c^{\mathrm{anc}}\right] \\
            & \approx U_{\hat{m}} \Sigma_{\hat{m}}  V_{\hat{m}}^{T}
            G_i = (\tilde{X}^{\mathrm{anc}}_i)^\dagger U_{\hat{m}}C
            \begin{equation}
            \hat{X} = \begin{bmatrix}
            \hat{X}_1 \\
            \hat{X}_2 \\
            \vdots \\
            \hat{X}_n
            \end{bmatrix} = \begin{bmatrix}
            \tilde{X}_1 \mathbf{G}_1 \\
            \tilde{X}_2 \mathbf{G}_2 \\
            \vdots \\
            \tilde{X}_{\hat{m}} \mathbf{G}_{\hat{m}}
            \end{bmatrix} \in \mathbb{R}^{n \times \hat{m}}
            \end{equation}
            \end{align*}
        """
        Gs = self._compute_target_matrix(Xancs_tilde)
        self._cox_regression(Xs_tilde, Gs, durations, events)
        return self


class CoxSurvivalFunction:
    def __init__(
        self,
        coef: pd.DataFrame,
        coef_var,
        baseline_hazard: pd.DataFrame,
        mean,
        alpha=0.05,
    ):
        self.__coef = coef
        self.__coef_var = coef_var
        self.__baseline_hazard = baseline_hazard
        self.__mean = mean
        self.alpha = alpha

    @property
    def baseline_cumhazards(self):
        """
        Baseline cumulative hazards.
        .. math::
            H\left(t\right)=\int_0^t h\left(z\right)dz
        """
        return pd.DataFrame(
            {
                "baseline cumhazards": [
                    self.cumhazard_at(t) for t in self.__baseline_hazard.index
                ]
            },
            index=self.__baseline_hazard.index,
        )

    @property
    def baseline_survival(self):
        """
        Baseline survivals.
        .. math::
            S\left(t\right)=\exp\left(-H\left(t\right)\right)
        """
        return pd.DataFrame(
            {
                "baseline survival": [
                    self.survival_at(t) for t in self.__baseline_hazard.index
                ]
            },
            index=self.__baseline_hazard.index,
        )

    @property
    def baseline_hazard(self):
        """
        Baseline hazards.
        .. math::
            h_0\left(t\right)
        """
        return self.__baseline_hazard

    @property
    def coef(self):
        """
        Coefficients.
        .. math::
            \beta
        """
        return self.__coef

    @property
    def coef_var(self):
        """
        Variance-covariance matrix.
        .. math::
            \mathrm{var}\left(\beta\right)
        """
        return self.__coef_var

    @property
    def SE(self):
        """
        Stanadard error.
        .. math::
            \sqrt{\mathrm{diag}\left(\mathrm{var}\left(\beta\right)\right)}
        """
        coef_var_ = (
            np.diag(self.__coef_var) if self.__coef_var.ndim == 2 else self.__coef_var
        )
        return pd.Series(np.sqrt(coef_var_), index=self.__coef.index)

    @property
    def stats(self):
        """
        t-statistics
        .. math::
            \frac{\beta}{\mathrm{SE}}
        """
        return self.__coef / self.SE

    @property
    def pvals(self):
        """
        P-values.
        .. math::
            1-F\left(|t|^2,\ 1\right)
        """
        pvals_ = 1 - sp.stats.chi2.cdf(np.square(self.stats), 1)
        return pd.Series(pvals_, index=self.__coef.index)

    @property
    def hazard_ratio(self):
        """
        Hazard ratios.
        .. math::
            \frac{h\left(t, \mathbf{x}+\mathbf{e}_i\right)}{h\left(t, \mathbf{x}\right)} = \exp(\beta_i)
        """
        return pd.Series(np.exp(self.__coef), index=self.__coef.index)

    @property
    def CI(self):
        """
        Confidence intervals.
        .. math::
            \begin{align*}
            \beta \pm Z_{\alpha /2}\mathrm{SE}
            \exp\left(\beta \pm Z_{\alpha /2}\mathrm{SE}\right)
            \end{align*}
        """
        alpha_stats_ = sp.stats.norm.ppf(1 - (self.alpha / 2))
        upper_ = self.__coef + alpha_stats_ * self.SE
        lower_ = self.__coef - alpha_stats_ * self.SE
        ci_name = f"{100*(1-self.alpha)}% CI"
        ci_result = {
            f"coef lower {ci_name}": lower_,
            f"coef upper {ci_name}": upper_,
            f"HR lower {ci_name}": np.exp(lower_),
            f"HR upper {ci_name}": np.exp(upper_),
        }
        return pd.DataFrame(ci_result, index=self.__coef.index)

    @property
    def summary(self):
        """
        Statistical summary.
        """
        df1 = pd.DataFrame(
            {
                "coef": self.__coef,
                "HR exp(coef)": self.hazard_ratio,
                "SE": self.SE,
            }
        )
        df2 = pd.DataFrame(
            {
                "z": self.stats,
                "p-value": self.pvals,
                "-log2(p)": -np.log2(self.pvals),
                "-log10(p)": -np.log10(self.pvals),
            }
        )
        return pd.concat([df1, self.CI, df2], axis=1)

    def predict_log_partial_hazard(self, X):
        """
        Log-partial hazard.
        .. math::
            \mathbf{X}\beta
        """
        X = X.loc[:, self.__coef.index]
        X -= self.__mean
        return X @ self.__coef

    def predict_partial_hazard(self, X):
        """
        Partial hazard.
        .. math::
            \exp\left(\mathbf{X}\beta\right)
        """
        return np.exp(self.predict_log_partial_hazard(X))

    def hazard_at(self, t):
        """
        Hazard at time t.
        .. math::
            h_0\left(t\right)
        """
        # h0(t)
        # TODO: Add Interpolation
        return self.__baseline_hazard.loc[t][0]

    def cumhazard_at(self, t):
        """
        Cumulative hazard at time t.
        .. math::
            H\left(t\right)=\int_0^t h\left(z\right)dz
        """
        # H(t)
        # TODO: Add Interpolation
        times = [t_ for t_ in self.__baseline_hazard.index if t_ <= t]
        hazards_ = [self.hazard_at(t_) for t_ in times]
        return reduce(lambda h1, h2: h1 + h2, hazards_)

    def survival_at(self, t):
        """
        Survival at time t.
        .. math::
            S\left(t\right)=\exp\left(-H\left(t\right)\right)
        """
        # S(t)
        return np.exp(-self.cumhazard_at(t))

    def predict_hazard_at(self, t, X):
        """
        Predict hazard at time t.
        .. math::
            h\left(t,\ \mathbf{X}\right) = h_0\left(t\right)\exp\left(\mathbf{X}\beta\right)
        """
        return self.hazard_at(t) * self.predict_partial_hazard(X)

    def predict_cumhazard_at(self, t, X):
        """
        Predict cumulative hazard at time t.
        .. math::
            H\left(t,\ \mathbf{X}\right)=\int_0^{t}h\left(z, \mathbf{X}\right)dz
        """
        return self.cumhazard_at(t) * self.predict_partial_hazard(X)

    def predict_survival_at(self, t, X):
        """
        Predict survival probability at time t.
        .. math::
            S\left(t,\ \mathbf{X}\right) = \exp\left(-H\left(t,\ \mathbf{X}\right)\right)
        """
        return np.exp(-self.predict_cumhazard_at(t, X))

    def _predict_times(self, func_name, X):
        pred_ = []
        pred_func = getattr(self, func_name)
        for t in self.__baseline_hazard.index:
            pred_.append(np.array(pred_func(t, X)).reshape(1, -1))
        return pd.DataFrame(np.concatenate(pred_), index=self.__baseline_hazard.index)

    def predict_hazard(self, X):
        """
        Predict hazard at a series of times.
        .. math::
            h\left(t,\ \mathbf{X}\right) = h_0\left(t\right)\exp\left(\mathbf{X}\beta\right)
        """
        return self._predict_times("predict_hazard_at", X)

    def predict_cumhazard(self, X):
        """
        Predict cumulative hazard at a series of times.
        .. math::
            H\left(t,\ \mathbf{X}\right)=\int_0^{t}h\left(z, \mathbf{X}\right)dz
        """
        return self._predict_times("predict_cumhazard_at", X)

    def predict_survival(self, X):
        """
        Predict survival probability at a series of times.
        .. math::
            S\left(t,\ \mathbf{X}\right) = \exp\left(-H\left(t,\ \mathbf{X}\right)\right)
        """
        return self._predict_times("predict_survival_at", X)

    def predict_expectation(self, X):
        """
        Predict expectation of survival time.
        .. math::
            \mathbb{E}\left[T\right] = \int_0^{\mathrm{inf}}S\left(t,\ \mathbf{X}\right)dt
        """
        survivals_ = self.predict_survival(X)
        return trapz(survivals_.T, survivals_.index)
