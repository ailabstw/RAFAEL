import os

import pandas as pd
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt

"""
This script is used to generate the ground truth answers,
not including the comparison between rafael and sklearn.
"""

def read_expressions(*files, meta_cols=None):
    expressions = []
    for f in files:
        exp = pd.read_csv(f)
        
        if meta_cols is not None:
            exp = exp.drop(columns=meta_cols)
            
        expressions.append(exp)
    
    return pd.concat(expressions)

def pca(data: pd.DataFrame, k: int):
    data = data.to_numpy()
    
    pca_ = PCA(n_components=k, svd_solver='randomized')
    row_pc = pca_.fit_transform(data)
    
    pca_ = PCA(n_components=k, svd_solver='randomized')
    col_pc = pca_.fit_transform(data.T)
    
    return row_pc, col_pc

def to_df(row_pc: pd.DataFrame, col_pc: pd.DataFrame, row_index=None, col_index=None):
    row_pc = pd.DataFrame(
        row_pc,
        index=row_index,
        columns=[f'PC{i+1}' for i in range(row_pc.shape[1])]
    )
    
    col_pc = pd.DataFrame(
        col_pc,
        index=col_index,
        columns=[f'PC{i+1}' for i in range(col_pc.shape[1])]
    )
    return row_pc, col_pc

def save_as_table(row_pc: pd.DataFrame, col_pc: pd.DataFrame, save_dir=None):
    save_dir = os.path.dirname(__file__) if save_dir is None else save_dir
    
    row_pc_file = os.path.join(save_dir, 'row.ans.PC.csv')
    col_pc_file = os.path.join(save_dir, 'col.ans.PC.csv')
    
    row_pc.to_csv(row_pc_file, index=None)
    col_pc.to_csv(col_pc_file, index=None)
    
def _plot(PCs: pd.DataFrame, idx: int, save_dir: str, filename: str):
    plt.figure(figsize=(6, 6), dpi=200)
    sns.scatterplot(
        PCs,
        x=f'PC{idx+1}',
        y=f'PC{idx+2}',
        s=8,
        alpha=0.5
    )
    
    plt.tight_layout()
    
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()
    
def save_fig(row_pc: pd.DataFrame, col_pc: pd.DataFrame, save_dir=None):
    save_dir = os.path.dirname(__file__) if save_dir is None else save_dir
    
    k = row_pc.shape[1]
    
    for i in range(k-1):
        _plot(row_pc, i, save_dir, f'pca.ans.row.PC{i+1}.PC{i+2}.png')
        _plot(col_pc, i, save_dir, f'pca.ans.col.PC{i+1}.PC{i+2}.png')


if __name__ == '__main__':
    exp_files = [f'/volume/jianhung-fa/rafael/data/client{i+1}/GSE62564-{i+1}-exp.csv' for i in range(3)]
    
    expression = read_expressions(meta_cols=['Unnamed: 0'], *exp_files)
    
    # Select features
    # expression = expression.loc[:, ['time', 'event', 'MYCN', 'BRCA1', 'MDM2', 'ALK']]
    
    # Select top k
    k = 20
    
    row_pc, col_pc = pca(expression, k)
    
    row_pc, col_pc = to_df(row_pc, col_pc)
    save_as_table(row_pc, col_pc)
    
    # Select top k
    k = 4
    row_pc, col_pc = row_pc.iloc[:, :k], col_pc.iloc[:, :k]
    
    save_fig(row_pc, col_pc)
