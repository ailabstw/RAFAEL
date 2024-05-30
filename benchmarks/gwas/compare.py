import os
from warnings import warn
import numpy as np
import scipy as sp
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from plotnine import (
    element_text,
    geom_point,
    ggplot,
    geom_bar,
    labs,
    facet_wrap,
    aes,
    ggsave,
    position_dodge,
    theme_classic,
    scale_fill_manual,
    theme,
)


def read_plink_eigenvec(file_path: str):
    return pd.read_csv(file_path, sep='\t')

def read_rafael_eigenvecs(file_paths: list):
    return pd.concat(
        [
            pd.read_csv(f, sep='\t')
            for f in file_paths
        ]
    )

def _drop_covars(eigenvec: pd.DataFrame, covars: list):
    return eigenvec.drop(columns=covars)

def _match_eigenvecs(plink: pd.DataFrame, rafael: pd.DataFrame, k=10):
    merged = pd.merge(plink, rafael, left_on=['#FID','IID'], right_on=['FID', 'IID']).drop(columns=['FID'])
    print(merged)
    print(plink.columns[2:])
    print(rafael.columns[2:])
    plink = merged.loc[:, plink.columns[2:]]
    rafael = merged.loc[:, rafael.columns[2:]]
    return plink, rafael

def compare_eigenvec(plink: pd.DataFrame, rafael: pd.DataFrame, save_dir: str=None):
    save_dir = os.path.dirname(__file__) if save_dir is None else save_dir
    
    # The direction of the eigenvecs may be different, use absolute values
    cos_sim = np.round(
        abs(plink.to_numpy().T @ rafael.to_numpy()), 2
    )
    
    print('Cosine similarity:\n')
    print(cos_sim)
    
    cos_sim = pd.DataFrame(
        cos_sim,
        index=rafael.columns,
        columns=rafael.columns
    )
    
    plt.figure(figsize=(6, 5), dpi=200)
    ax = sns.heatmap(cos_sim, annot=True, cmap='viridis', annot_kws={"size": 10})
    ax.set_ylabel(f'PLINK SVD Sample Eigenvectors')
    ax.set_xlabel(f'RAFAEL SVD Sample Eigenvectors')
    plt.tight_layout()
    plt.savefig(f'plink.rafael.compare.eigenvec.png')
    plt.savefig(os.path.join(save_dir, f'plink.rafael.compare.eigenvec.png'))
    
def read_glm(glm_path: str):
    if ".csv" in glm_path:
        glm = pd.read_csv(glm_path, sep = ",")
    else:
        glm = pd.read_csv(glm_path, sep = "\s+")

    if "TEST" in glm:
        glm = glm.loc[glm.TEST.isin(["Add","ADD"])]
    
    return glm

def format_glm(glm: pd.DataFrame, label: str=""):
    colname_cand = [
        "#CHROM", "CHR", "chr", "Chr", "chrom", "CHROM", "chromsome", "Chrom",
        "POS", "pos", "position", "Position", "physical position", "BP", "bp", "POSITION",
        "p", "P", "pvalue", "p-value", "PVALUE", "P-value", "Pvalue",
        "SNP", "snp", "id", "variant", "ID", "Variant", "Variants"
    ]

    # locate the expected columns
    col_idx = sorted([colname_cand.index(col) for col in glm.columns if col in colname_cand])
    cols = [colname_cand[idx] for idx in col_idx]
    assert len(col_idx) == 4, f"The GLM columns must have CHR, POS, P and ID.\nAvailable candidates:\n{colname_cand}"
    
    glm = glm.loc[:, cols]
    glm.columns = ["CHR","POS","P","SNP"]
    glm["Label"] = label

    # add unique position id for downstream evaluation
    glm["-LOG10P"] = -1 * np.log10(glm.P.values)
    glm.index = [f'{c}-{p}' for c, p in list(zip(glm["CHR"], glm["POS"]))]
    
    return glm

def eval_jci(plink: pd.DataFrame, rafael: pd.DataFrame, thres: float=5e-8):
    assert plink.index[0] != 0, "The index must be 'CHR-POS'."
    assert rafael.index[0]!= 0, "The index must be 'CHR-POS'."

    # evaluate the concordance of significant SNPs by jaccard index
    plink_sig = plink[plink["P"] < thres]
    rafael_sig = rafael[rafael["P"] < thres]
    
    intersection_sig = set(plink_sig.index).intersection(rafael_sig.index)
    union_sig = set(plink_sig.index).union(rafael_sig.index)
    
    jci = len(intersection_sig)/len(union_sig)
    
    print(
        f"PLINK significant SNPs (pval < {thres}):\n"
        f"{plink_sig}\n"
        f"RAFAEL significant SNPs (pval < {thres}):\n"
        f"{rafael_sig}\n"
        f"The JCI = {jci} ({len(intersection_sig)}/{len(union_sig)})"
    )
    return jci

def eval_pcc(plink: pd.DataFrame, rafael: pd.DataFrame):
    assert plink.index[0] != 0, "The index must be 'CHR-POS'."
    assert rafael.index[0]!= 0, "The index must be 'CHR-POS'."
    
    # prevent infinity errors
    plink_inf = plink[np.isinf(plink["-LOG10P"])]
    rafael_inf = rafael[np.isinf(rafael["-LOG10P"])]
    
    if len(plink_inf) > 0:
        warn(f"PLINK result contains -log10(P)=inf, these will be removed in pcc evaluation\n{plink_inf}\n")

    if len(rafael_inf) > 0:
        warn(f"RAFAEL result contains -log10(P)=inf, these will be removed in pcc evaluation\n{rafael_inf}\n")


    # use overlapped SNPs for pcc evaluation
    whole_intersection = list(set(plink.index).intersection(rafael.index))

    if len(whole_intersection) != len(plink) or len(whole_intersection) != len(rafael):
        warn(
            f"PLINK and RAFAEL have different number of SNPs.\n"
            f"PLINK total snps: {len(plink)}\n"
            f"RAFAEL total snps: {len(rafael)}\n"
            f"Intersection: {len(whole_intersection)}\n"
        )


    pcc_res = sp.stats.pearsonr(plink.loc[whole_intersection, "-LOG10P"], rafael.loc[whole_intersection, "-LOG10P"])
    print(f"PCC on -log10(P): {pcc_res}\n")
    
    return pcc_res[0]
    
def glm_comparison_barplot(plt_df: pd.DataFrame, save_dir: str=None):
    save_dir = os.path.dirname(__file__) if save_dir is None else save_dir

    p = (
        ggplot(plt_df)
        + geom_bar(
            aes(x="Metric", y="Value", fill="Scenario"),
            stat="identity",
            position="dodge",
            width=0.8,
        )
        + theme_classic()
        + facet_wrap("~Name", scales="free_y", nrow=1, shrink=False)
        + theme(
            legend_text=element_text(size=8),
            legend_title=element_text(size=10),
            legend_position="top",
        )
        + scale_fill_manual(
            values=["#5e457a", "#3e557f", "#42868a", "#6cbe7a", "#b0d85b", "#eee652"]
        )
        + labs(x="", y="")
    )

    ggsave(p, f"{os.path.join(save_dir, 'glm.comparison.bar.png')}", width=6.5, height=5, dpi=300)

def glm_comparison_scatterplot(plt_df: pd.DataFrame, save_dir: str=None):
    save_dir = os.path.dirname(__file__) if save_dir is None else save_dir

    p = (
        ggplot(plt_df)
        + geom_point(
            aes(x="Metric", y="Value", fill="Scenario"),
            position=position_dodge(width=0.4),
            size=5,
            stroke=0
        )
        + theme_classic()
        + facet_wrap("~Name", scales="free_y", nrow=1, shrink=False)
        + theme(
            legend_text=element_text(size=8),
            legend_title=element_text(size=10),
            legend_position="top",
        )
        + scale_fill_manual(
            values=["#5e457a", "#3e557f", "#42868a", "#6cbe7a", "#b0d85b", "#eee652"]
        )
        + labs(x="", y="")
    )

    ggsave(p, f"{os.path.join(save_dir, 'glm.comparison.scat.png')}", width=7, height=5, dpi=300)

def _compare_glm(*glm_pairs, sig_thres: float=5e-8, save_dir: str=None):
    save_dir = os.path.dirname(__file__) if save_dir is None else save_dir

    scenarios = {
        "I": "Balanced",
        "II": "Slightly Imbalanced",
        "III": "Moderately Imbalanced",
        "IV": "Highly Imbalanced",
        "V": "Severely Imbalanced",
        "VI": "Heterogeneous Confounding Factor"
    }
    
    jci_df = pd.DataFrame(
        {
            "Scenario": scenarios.values(),
            "Metric": ["Jaccard Index"] * len(scenarios),
            "Value": list(map(lambda glm: eval_jci(glm[0], glm[1], sig_thres), glm_pairs))
        }
    )
    
    pcc_df = pd.DataFrame(
        {
            "Scenario": scenarios.values(),
            "Metric": ["Pearson Correlation"] * len(scenarios),
            "Value": list(map(lambda glm: eval_pcc(glm[0], glm[1]), glm_pairs))
        }
    )
    
    plt_df = pd.concat([pcc_df, jci_df]).reset_index(drop=True)
    plt_df["Name"] = "Concordance Metrics"

    print(plt_df)

    glm_comparison_barplot(plt_df, save_dir)
    glm_comparison_scatterplot(plt_df, save_dir)
    
def compare_glm(ans_path: str, *res_paths, label: str="RAFAEL", save_dir=None):
    save_dir = os.path.dirname(__file__) if save_dir is None else save_dir
    
    ans = format_glm(read_glm(ans_path), label="PLINK")
    
    glm_pairs = []
    for path in res_paths:
        glm_pairs.append((ans, format_glm(read_glm(path), label=label)))
    
    _compare_glm(*glm_pairs, sig_thres=5e-8, save_dir=save_dir)
    
if __name__ == "__main__":
    # Comparison of Jaccard Index (JCI) for evaluating significant SNP overlap 
    # and Pearson Correlation Coefficient (PCC) for assessing the overall p-value distribution concordance, respectively.
    res_paths = []
    for scen in ["I", "II", "III", "IV", "V", "VI"]:
        res_paths.append(f"/volume/jianhung-fa/rafael/large_logistic_{scen}/gwas.glm")
    
    compare_glm(
        "/volume/jianhung-fa/TWB2_240418/DATA/GOUT/agg/agg.qc.pca.PHENO1.glm.logistic.hybrid",
        *res_paths
    )
