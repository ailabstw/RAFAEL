import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

"""
This script is used to generate the ground truth answers,
not including the comparison between rafael and lifelines.
"""

def read_expressions(*files, meta_cols=None):
    expressions = []
    for f in files:
        exp = pd.read_csv(f)
        
        if meta_cols is not None:
            exp = exp.drop(columns=meta_cols)
            
        expressions.append(exp)
    
    return pd.concat(expressions)

def fit(
        upper_time: np.array, 
        upper_event: np.array,
        upper_label: str,
        lower_time: np.array, 
        lower_event: np.array,
        lower_label: str,
        feature_name: str,
        n_std: float,
        save_dir: str=None
    ):
    model = KaplanMeierFitter()
    
    ax = plt.subplot(111)
    
    # Lower fitting
    model.fit(lower_time, lower_event, label=lower_label)
    model.plot_survival_function(ax=ax, figsize=(8,6))
    lower_med_surv_t = model.median_survival_time_
    
    # Upper fitting
    model.fit(upper_time, upper_event, label=upper_label)
    model.plot_survival_function(ax=ax, figsize=(8,6))
    upper_med_surv_t = model.median_survival_time_
    
    # Logrank test
    test_res = logrank_test(
        durations_A=upper_time,
        durations_B=lower_time,
        event_observed_A=upper_event,
        event_observed_B=lower_event
    ).summary
    stats, pval = test_res['test_statistic'][0], test_res['p'][0]
    
    # Plot information
    ax.text(
        1.05, .5,
        f'Number of observations\n'
        f'  $\geq +{n_std}\sigma$={len(upper_time)}\n'
        f'  $\leq -{n_std}\sigma$={len(lower_time)}\n'
        f'\n'
        f'Median Survival time:\n'
        f'  $\geq +{n_std}\sigma$={upper_med_surv_t}\n'
        f'  $\leq -{n_std}\sigma$={lower_med_surv_t}\n'
        f'\n'
        f'Log-rank test:\n'
        f'  statistic={stats:.3f}\n'
        f'  p-value={pval:.3f}',
        fontsize=10, 
        verticalalignment='center',
        horizontalalignment='left',
        transform=ax.transAxes
    )
    ax.set_title(feature_name)
    ax.set_xlabel('timeline')
    ax.set_ylabel('Survival Probability')
    
    handles, labels = ax.get_legend_handles_labels()
    handles = handles[::-1]
    labels = labels[::-1]
    # ax.legend(handles, labels)
    ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.95))
    
    plt.tight_layout()
    
    save_dir = os.path.dirname(__file__) if save_dir is None else save_dir
    plt.savefig(os.path.join(save_dir, f'km.ans.{feature_name}.{n_std}std.png'))
    
    plt.close()

def standardize(X):
    return np.array((X-np.mean(X, axis=0)) / np.std(X, axis=0, ddof=1))

def main(X, y, n_std):
    if isinstance(n_std, float):
        n_std = [n_std] * X.shape[1]
    
    stdzX = standardize(X)
    y = np.array(y)
    
    for i in range(stdzX.shape[1]):
        upper_mask = stdzX[:, i] >= n_std[i]
        lower_mask = stdzX[:, i] <= -n_std[i]
        
        if len(y[lower_mask]) == 0 or len(y[upper_mask]) == 0:
            continue
        
        print(f'{X.columns[i]} upper n: {len(y[upper_mask])}, lower n: {len(y[lower_mask])}')
        
        fit(
            upper_time=y[upper_mask][:, 0],
            upper_event=y[upper_mask][:, 1],
            upper_label=f'$\geq +{+n_std[i]}\sigma$',
            lower_time=y[lower_mask][:, 0],
            lower_event=y[lower_mask][:, 1],
            lower_label=f'$\leq -{n_std[i]}\sigma$',
            feature_name=X.columns[i],
            n_std=n_std[i],
        )

if __name__ == '__main__':
    exp_files = [f'/volume/jianhung-fa/rafael/data/client{i+1}/GSE62564-{i+1}.csv' for i in range(3)]
    
    expression = read_expressions(meta_cols=['sample-id'], *exp_files)

    X = expression.drop(columns=['time', 'event'])
    y = expression.loc[:, ['time', 'event']]
    
    main(X, y, 1.0)
