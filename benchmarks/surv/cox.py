import os
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter

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

def fit(data: pd.DataFrame):
    assert 'time' in data.columns, "Duration column `time` is missing."
    assert 'event' in data.columns, "Event column `event` is missing."
    
    model = CoxPHFitter()
    model.fit(data, event_col='event', duration_col='time')
    
    return model

def save_statistic_summary(model: CoxPHFitter, save_dir: str=None):
    save_dir = os.path.dirname(__file__) if save_dir is None else save_dir
    model.summary.to_csv(os.path.join(save_dir, 'cox.ans.summary.csv'))

def save_coef_fig(model: CoxPHFitter, save_dir: str=None):
    save_dir = os.path.dirname(__file__) if save_dir is None else save_dir
    model.plot()
    plt.savefig(os.path.join(save_dir, 'cox.ans.coef.png'))


if __name__ == "__main__":
    exp_files = [f'/volume/jianhung-fa/rafael/data/client{i+1}/GSE62564-{i+1}.csv' for i in range(3)]
    
    expression = read_expressions(meta_cols=['sample-id'], *exp_files)
    
    # Select features
    # expression = expression.loc[:, ['time', 'event', 'MYCN', 'BRCA1', 'MDM2', 'ALK']]
    
    model = fit(expression)
    
    save_statistic_summary(model)
    save_coef_fig(model)
