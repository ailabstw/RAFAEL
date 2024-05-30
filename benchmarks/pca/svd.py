import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

"""
This script is used to generate the ground truth answers,
not including the comparison between rafael and numpy.
"""

def read_expressions(*files, meta_cols=None):
    expressions = []
    for f in files:
        exp = pd.read_csv(f)
        
        if meta_cols is not None:
            exp = exp.drop(columns=meta_cols)
            
        expressions.append(exp)
    
    return pd.concat(expressions)

def svd(data: pd.DataFrame):
    data = data.to_numpy()
    
    U, S, Vh = np.linalg.svd(data, full_matrices=False)
    
    return U, S, Vh.T

def to_df(U: np.array, S: np.array, V: np.array, row_index=None, col_index=None):
    U = pd.DataFrame(
        U,
        index=row_index,
        columns=[f'Eigenvec{i+1}' for i in range(U.shape[1])]
    )
    
    S = pd.DataFrame(S).T
    S.columns = [f'Sinval{i+1}' for i in range(S.shape[1])]
    
    V = pd.DataFrame(
        V,
        index=col_index,
        columns=[f'Eigenvec{i+1}' for i in range(V.shape[1])]
    )
    return U, S, V

def save_as_table(U: pd.DataFrame, S: pd.DataFrame, V: pd.DataFrame, save_dir=None):
    save_dir = os.path.dirname(__file__) if save_dir is None else save_dir
    
    u_file = os.path.join(save_dir,'row.ans.eigenvec.csv')
    s_file = os.path.join(save_dir,'s.ans.sinval.csv')
    v_file = os.path.join(save_dir, 'col.ans.eigenvec.csv')
    
    U.to_csv(u_file, index=None)
    S.to_csv(s_file, index=None)
    V.to_csv(v_file, index=None)
    
def _plot(Eigenvec: pd.DataFrame, idx: int, save_dir: str, filename: str):
    plt.figure(figsize=(7, 6), dpi=200)
    sns.scatterplot(
        Eigenvec,
        x=f'Eigenvec{idx+1}',
        y=f'Eigenvec{idx+2}',
        s=8,
        alpha=0.5
    )
    
    plt.tight_layout()
    
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()
    
def save_fig(U: pd.DataFrame, V: pd.DataFrame, save_dir=None):
    save_dir = os.path.dirname(__file__) if save_dir is None else save_dir
    
    k = U.shape[1]
    
    for i in range(k-1):
        _plot(U, i, save_dir, f'svd.ans.row.eigenvec{i+1}.eigenvec{i+2}.png')
        _plot(V, i, save_dir, f'svd.ans.col.eigenvec{i+1}.eigenvec{i+2}.png')


if __name__ == '__main__':
    exp_files = [f'/volume/jianhung-fa/rafael/data/client{i+1}/GSE62564-{i+1}-exp.csv' for i in range(3)]
    
    expression = read_expressions(meta_cols=['Unnamed: 0'], *exp_files)
    
    # Select features
    # expression = expression.loc[:, ['time', 'event', 'MYCN', 'BRCA1', 'MDM2', 'ALK']]
    
    U, S, V = svd(expression)
    
    # Select top k
    k = 20
    U, S, V = U[:, :k], S[:k], V[:, :k]
    
    U, S, V = to_df(U, S, V)
    save_as_table(U, S, V)
    
    # Select top k
    k = 4
    U, V = U.iloc[:, :k], V.iloc[:, :k]
    
    save_fig(U, V)
