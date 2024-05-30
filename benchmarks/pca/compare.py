import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def read_latent_vec(filename: str):
    return pd.read_csv(filename)

def read_latent_vecs(*filenames):
    return pd.concat([pd.read_csv(f) for f in filenames]).reset_index(drop=True)

def read_sinval(filename: str):
    return pd.read_csv(filename)

def compare_pc(ans: pd.DataFrame, result: pd.DataFrame, kind: str, k: int, save_dir: str=None):
    save_dir = os.path.dirname(__file__) if save_dir is None else save_dir
    
    print(
        f'Answer:\n'
        f'{ans}\n'
        f'\n'
        f'RAFAEL Result:\n'
        f'{result}\n'
    )
    
    # Make sure they are in the same direction
    neg_idx = np.where(np.sign(ans) != np.sign(result))[1]
    result.iloc[:, neg_idx] *= -1
    
    ans['Source'] = 'Scikit-learn'
    result['Source'] = 'RAFAEL'
    
    plot_df = pd.concat([ans, result])
    
    for i in range(k-1):
        plt.figure(figsize=(8, 8), dpi=200)
        sns.scatterplot(
            data=plot_df,
            x=f'PC{i+1}',
            y=f'PC{i+2}',
            hue='Source',
            style='Source',
            markers={'Scikit-learn':'^', 'RAFAEL':'X'},
            size_order=('RAFAEL', 'Scikit-learn'),
            sizes=(60, 30),
            size='Source',
            palette='Set2',
            alpha=.9
        )

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'compare.{kind}.PC{i+1}.PC{i+2}.png'))
        plt.close()

def compare_eigenvec(ans: pd.DataFrame, result: pd.DataFrame, kind: str, save_dir: str=None):
    save_dir = os.path.dirname(__file__) if save_dir is None else save_dir

    print(
        f'Answer:\n'
        f'{ans}\n'
        f'\n'
        f'RAFAEL Result:\n'
        f'{result}\n'
    )

    cos_sim = np.round(
        abs(ans.to_numpy().T @ result.to_numpy()), 2
    )
    
    print('Cosine similarity:\n')
    print(cos_sim)
    
    cos_sim = pd.DataFrame(
        cos_sim,
        index=ans.columns,
        columns=ans.columns
    )
    
    plt.figure(figsize=(8, 6), dpi=200)
    ax = sns.heatmap(cos_sim, annot=True, cmap='viridis')
    fname = 'Sample' if kind == 'row' else 'Gene'
    ax.set_ylabel(f'NumPy SVD {fname} Eigenvectors')
    ax.set_xlabel(f'RAFAEL SVD {fname} Eigenvectors')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'compare.{kind}.eigenvec.png'))
    plt.close()
    
def compare_sinval(ans: pd.DataFrame, result: pd.DataFrame, save_dir: str=None):
    save_dir = os.path.dirname(__file__) if save_dir is None else save_dir

    print(
        f'Answer:\n'
        f'{ans}\n'
        f'\n'
        f'RAFAEL Result:\n'
        f'{result}\n'
    )
    
    plot_df = pd.concat([ans, result]).T
    plot_df.columns = ['NumPy Singular Value', 'RAFAEL Singular Value']
    
    plt.figure(figsize=(5, 5), dpi=200)
    plt.plot(
        [plot_df.min().min(), plot_df.max().max()],
        [plot_df.min().min(), plot_df.max().max()],
        '--',
        color='gray',
        alpha=.5
    )
    sns.scatterplot(
        data=plot_df,
        x='NumPy Singular Value',
        y='RAFAEL Singular Value',
        s=30,
        alpha=.9
    )
    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'compare.sinval.png'))
    plt.close()


if __name__ == '__main__':
    k = 4 
    
    """
    Comparing the PCs.
    
    The RAFAEL request:
    {
        "node_id": "1",
        "args": {
            "config": {
                "clients": [
                    "2",
                    "3",
                    "4"
                ],
                "file_path":[
                    "/volume/gwasfl/rafael/data/client1/GSE62564-1-exp.csv",
                    "/volume/gwasfl/rafael/data/client2/GSE62564-2-exp.csv",
                    "/volume/gwasfl/rafael/data/client3/GSE62564-3-exp.csv"
                ],
                "svd_save_dir":[
                    "/volume/gwasfl/rafael/svd1",
                    "/volume/gwasfl/rafael/svd2",
                    "/volume/gwasfl/rafael/svd3"
                ],
                "to_pc": true
            }
        },
        "api": "PCAfromTabular"
    }
    """
    compare_pc(
        read_latent_vec('row.ans.PC.csv'),
        read_latent_vecs(
            '/volume/jianhung-fa/rafael/svd1/row.pc.csv',
            '/volume/jianhung-fa/rafael/svd2/row.pc.csv',
            '/volume/jianhung-fa/rafael/svd3/row.pc.csv'
        ),
        'row',
        k
    )
    
    # TODO: do this comparison after deploying row-wise standardization
    # compare_pc(
    #     read_latent_vec('col.ans.PC.csv'),
    #     read_latent_vec('/volume/jianhung-fa/rafael/svd1/col.pc.csv'),
    #     'col',
    #     k
    # )
    
    """
    
    Comparing the eigenvectors and singular values.
    
    The RAFAEL request:
    {
        "node_id": "1",
        "args": {
            "config": {
                "clients": [
                    "2",
                    "3",
                    "4"
                ],
                "file_path":[
                    "/volume/gwasfl/rafael/data/client1/GSE62564-1-exp.csv",
                    "/volume/gwasfl/rafael/data/client2/GSE62564-2-exp.csv",
                    "/volume/gwasfl/rafael/data/client3/GSE62564-3-exp.csv"
                ],
                "svd_save_dir":[
                    "/volume/gwasfl/rafael/svd1",
                    "/volume/gwasfl/rafael/svd2",
                    "/volume/gwasfl/rafael/svd3"
                ]
            }
        },
        "api": "SVDfromTabular"
    }
    """
    compare_eigenvec(
        read_latent_vec('row.ans.eigenvec.csv'),
        read_latent_vecs(
            '/volume/jianhung-fa/rafael/svd1/row.eigenvec.csv',
            '/volume/jianhung-fa/rafael/svd2/row.eigenvec.csv',
            '/volume/jianhung-fa/rafael/svd3/row.eigenvec.csv',
        ),
        kind='row'
    )
    
    compare_eigenvec(
        read_latent_vec('col.ans.eigenvec.csv'),
        read_latent_vec('/volume/jianhung-fa/rafael/svd1/col.eigenvec.csv'),
        kind='col'
    )
    
    compare_sinval(
        read_sinval('s.ans.sinval.csv'),
        read_sinval('/volume/jianhung-fa/rafael/svd1/s.sinval.csv')
    )
