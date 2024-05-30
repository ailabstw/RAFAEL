import os
import pandas as pd

from gwasprs import gwasplot

"""
This script is used to generate the ground truth answers,
not including the comparison between rafael and plink.

This should be performed on the aggregated dataset.
"""

def _makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def quality_control(bfile_path, output_path, maf=0.05, geno=0.02, mind=0.02, hwe=5e-7):
    _makedirs(os.path.dirname(output_path))
    
    os.system(
        f'plink2 --bfile {bfile_path} \
                 --missing \
                 --hardy \
                 --freq \
                 --autosome \
                 --allow-extra-chr \
                 --out {output_path}'
    )
    
    os.system(
        f'plink2 --bfile {bfile_path} \
                 --maf {maf} \
                 --geno {geno} \
                 --hwe {hwe} \
                 --read-freq {output_path}.afreq \
                 --het \
                 --make-bed \
                 --autosome \
                 --allow-extra-chr \
                 --out {output_path}.tmp'
    )

    # I assume that in the previous command, "mind" considers all SNPs,
    # whereas in this command, "mind" considers only the filtered SNPs.
    # This results in different remaining samples, aligning with the implementation in RAFAEL.
    os.system(
        f'plink2 --bfile {output_path}.tmp \
                 --mind {mind} \
                 --make-bed \
                 --autosome \
                 --allow-extra-chr \
                 --out {output_path}'
    )
    
def ld_pruning(bfile_path, output_path, win_size=50, step=5, r2=0.2):
    _makedirs(os.path.dirname(output_path))

    os.system(
        f'plink2 --bfile {bfile_path} \
                 --indep-pairwise {win_size} {step} {r2} \
                 --bad-ld \
                 --out {output_path}'
    )
    
    os.system(
        f'plink2 --bfile {bfile_path} \
                 --extract {output_path}.prune.in \
                 --make-bed \
                 --autosome \
                 --allow-extra-chr \
                 --out {output_path}'
    )
    
def pca(bfile_path, output_path, k=10):
    _makedirs(os.path.dirname(output_path))

    os.system(
        f'plink2 --bfile {bfile_path} \
                 --pca {k} \
                 --out {output_path}'
    )
    
def _concat_cov(cov_path, pc_path, output_path):
    cov = pd.read_csv(cov_path, sep='\t')
    pc = pd.read_csv(pc_path, sep='\t')
    merged = pd.merge(left=cov, right=pc, left_on=['FID', 'IID'], right_on=['#FID', 'IID'], how='inner')
    merged.drop(columns=['#FID'], inplace=True)
    merged.to_csv(output_path, sep='\t', index=None)
    
def regression(bfile_path, cov_path, output_path):
    _makedirs(os.path.dirname(output_path))

    os.system(
        f'plink2 --bfile {bfile_path} \
                 --glm hide-covar cols=chrom,pos,ref,alt,ax,a1freq,nobs,orbeta,se,ci,tz,p \
                 --covar {cov_path} \
                 --covar-variance-standardize \
                 --out {output_path}'
    )
    
def plot_results(
    glm_path,
    chr_col="#CHROM",
    pos_col="POS",
    stat_col="P",
    sep="\t",
    out_preifx="",
    upper_thres=5e-8,
    lower_thres=1e-5,
    ):
    glm = gwasplot.read_glm(
        glm_path,
        chr_col,
        pos_col,
        stat_col,
        sep
    )
    glm, max_logp = gwasplot.format_glm(glm)

    # Manhattan
    manhattan_glm = gwasplot.prepare_manhattan(glm)
    gwasplot.plot_manhattan(
        *manhattan_glm,
        f"{out_preifx}.manhattan.png",
        lower_thres,
        upper_thres,
        max_logp
    )

    # QQ
    qq_glm = gwasplot.prepare_qq(glm)
    gwasplot.plot_qq(qq_glm, f"{out_preifx}.qq.png")
    
def parse_args():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("-b", "--bfile", type=str, required=True, help="The path to bfile")
    parser.add_argument("-o", "--out", type=str, required=True, help="The output path and prefix")
    parser.add_argument("-c", "--covar", type=str, required=True, help="The path to covariate file")
    parser.add_argument("--maf", type=float, default=0.05, help="The minor allele frequency threshold for QC")
    parser.add_argument("--geno", type=float, default=0.02, help="The missing genotype threshold for QC")
    parser.add_argument("--mind", type=float, default=0.02, help="The missing individual threshold for QC")
    parser.add_argument("--hwe", type=float, default=5e-7, help="Hardy-Weinberg equilibrium exact test p-value threshold for QC")
    parser.add_argument("--win", type=int, default=50, help="window size in variant count for LD pruning")
    parser.add_argument("--step", type=int, default=5, help="variant count to shift the window at the end of each step for LD pruning")
    parser.add_argument("--r2", type=float, default=0.2, help="The r2 threshold for LD pruning")
    parser.add_argument("-k", "--pca", type=int, default=10, help="The first n eigenvectors to be calculated")

    args = parser.parse_args()

    return args
    
if __name__ == "__main__":
    args = parse_args()
    
    quality_control(
        bfile_path=args.bfile,
        output_path=f"{args.out}.qc",
        maf=args.maf,
        geno=args.geno,
        mind=args.mind,
        hwe=args.hwe
    )
    
    ld_pruning(
        bfile_path=f"{args.out}.qc",
        output_path=f"{args.out}.ld",
        win_size=args.win,
        step=args.step,
        r2=args.r2
    )
    
    pca(
        bfile_path=f"{args.out}.ld",
        output_path=f"{args.out}.pca",
        k=args.pca
    )
    
    _concat_cov(
        cov_path=args.covar,
        pc_path=f"{args.out}.pca.eigenvec",
        output_path=f"{args.out}.cov"
    )
    
    regression(
        bfile_path=f"{args.out}.qc",
        cov_path=f"{args.out}.cov",
        output_path=f"{args.out}.reg"
    )

    try:
        plot_results(
            glm_path=f"{args.out}.reg.PHENO1.glm.logistic.hybrid",
            out_preifx=args.out
        )
    except:
        plot_results(
            glm_path=f"{args.out}.reg.PHENO1.glm.linear",
            out_preifx=args.out
        )
