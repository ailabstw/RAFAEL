from warnings import warn
from typing import List, Tuple

import pandas as pd
import numpy as np
import numpy.typing as npt

from .utils import bash, PLINK2_PATH
from .hwe import read_hardy, cal_hwe_pvalue_vec
from .reader import FamReader, CovReader
from .gwasdata import format_cov


def cal_qc_client(
    bfile_path: str,
    output_path: str,
    snp_list: List[str],
    het_bin: int,
    het_range: Tuple[float, float],
):
    warn("cal_qc_client is deprecated.", DeprecationWarning, stacklevel=2)
    if len(snp_list) > 0:
        write_snp_list(f"{output_path}.common_snp_list", snp_list)

    calculate_allele_freq_hwe(bfile_path, output_path, snp_list)
    calculate_homo_het_count(bfile_path, output_path, snp_list)

    allele_count = read_hardy(output_path)
    n_obs = get_obs_count(f"{output_path}.vmiss")
    het_hist, het = get_histogram(f"{output_path}.het", het_bin, het_range)

    return allele_count, het_hist, het, n_obs


def qc_stats(bfile_path: str, output_path: str, snp_list: List[str]):
    if len(snp_list) > 0:
        write_snp_list(f"{output_path}.common_snp_list", snp_list)

    calculate_allele_freq_hwe(bfile_path, output_path, snp_list)
    calculate_homo_het_count(bfile_path, output_path, snp_list)

    allele_count = read_hardy(output_path)
    n_obs = get_obs_count(f"{output_path}.vmiss")

    return allele_count, n_obs


def calculate_allele_freq_hwe(bfile_path: str, output_path: str, snp_list: List[str]):
    extract_arg = (
        f"--extract {output_path}.common_snp_list" if len(snp_list) > 0 else ""
    )
    plink_cmd = f"{PLINK2_PATH} --bfile {bfile_path} \
                                {extract_arg} \
                                --rm-dup force-first \
                                --allow-extra-chr \
                                --freq \
                                --hardy \
                                --missing \
                                --out {output_path}"
    bash(plink_cmd)


def calculate_homo_het_count(bfile_path: str, output_path: str, snp_list: List[str]):
    # read-freq for sample size < 50
    extract_arg = (
        f"--extract {output_path}.common_snp_list" if len(snp_list) > 0 else ""
    )
    plink_cmd = f"{PLINK2_PATH} --bfile {bfile_path} \
                                {extract_arg} \
                                --rm-dup force-first \
                                --allow-extra-chr \
                                --het \
                                --read-freq {output_path}.afreq \
                                --out {output_path}"
    bash(plink_cmd)


def write_snp_list(output_path, snp_list):
    output = open(output_path, "w")
    snp_list = list(map(lambda snp: snp + "\n", snp_list))
    output.writelines(snp_list)
    output.close()


def get_histogram(het_path, bin, range):
    het = pd.read_csv(het_path, sep=r"\s+")
    het_hist, _ = np.histogram(het.F, bins=bin, range=range)
    return het_hist, het.F.values


def get_obs_count(vmiss_path):
    vmiss = pd.read_csv(vmiss_path, sep=r"\s+")
    n_obs = vmiss.OBS_CT.max()
    return n_obs


# aggregator use het to filter ind
def filter_ind(
    HET,
    het_mean: float,
    heta_std: float,
    HETER_SD: float,
    sample_list: npt.NDArray[np.byte],
):
    HET = np.abs((HET - het_mean) / heta_std)
    # currently het is not filtered
    remove_idx = np.where(HET > HETER_SD)[0]
    remove_list = [sample_list[i] for i in remove_idx]

    return remove_list


# edge create bed after qc
def create_filtered_bed(
    bfile_path: str,
    filtered_bfile_path: str,
    keep_snps: List[str],
    mind: float,
    keep_inds: List[Tuple[str, str]] = [],
):
    ind_list_output = open(f"{filtered_bfile_path}.ind_list", "w")
    keep_inds = list(map(lambda ind: f"{ind[0]}\t{ind[1]}\n", keep_inds))
    ind_list_output.writelines(keep_inds)
    ind_list_output.close()

    assert len(keep_snps) > 0
    write_snp_list(f"{filtered_bfile_path}.snp_list", keep_snps)

    plink_cmd = f"{PLINK2_PATH} --bfile {bfile_path} \
                                --mind {mind} \
                                --extract {filtered_bfile_path}.snp_list \
                                --keep {filtered_bfile_path}.ind_list \
                                --hardy \
                                --allow-extra-chr \
                                --rm-dup force-first \
                                --make-bed \
                                --out {filtered_bfile_path}"
    output, err = bash(plink_cmd)
    return filtered_bfile_path


def create_filtered_covariates(filtered_bfile_path, cov_path):
    fam = FamReader(filtered_bfile_path).read()
    cov = CovReader(cov_path).read()
    filtered_cov = format_cov(cov, fam)
    filtered_cov_path = filtered_bfile_path + ".cov"
    filtered_cov.to_csv(filtered_cov_path, index=None, sep="\t")
    return filtered_cov_path


def filter_snp(
    allele_count: np.ndarray,
    snp_id: np.ndarray,
    sample_count: int,
    save_path: str,
    geno: float,
    hwe: float,
    maf: float,
):
    obs_count = allele_count.sum(axis=1)

    # FREQ
    obs_maf = (allele_count[:, 0] * 2 + allele_count[:, 1]) / (2 * obs_count)

    # HWE
    obs_hwe_pval = cal_hwe_pvalue_vec(
        allele_count[:, 1], allele_count[:, 0], allele_count[:, 2]
    )

    # MISSING
    obs_missing_rate = 1 - (obs_count / sample_count)

    # SUMMARY
    obs_summary = pd.DataFrame(
        {"ID": snp_id, "MISSING": obs_missing_rate, "HWE": obs_hwe_pval, "MAF": obs_maf}
    )

    obs_summary.loc[obs_summary.MAF > 0.5, "MAF"] = (
        1 - obs_summary.loc[obs_summary.MAF > 0.5].MAF
    )
    obs_summary["PASS"] = False
    obs_summary.MISSING = obs_summary.MISSING.round(6)
    obs_summary.MAF = obs_summary.MAF.round(6)
    geno_filter = obs_summary.MISSING <= geno
    hwe_filter = obs_summary.HWE >= hwe
    maf_filter = obs_summary.MAF >= maf
    obs_summary.loc[geno_filter & hwe_filter & maf_filter, "PASS"] = True
    snp_id = obs_summary[obs_summary.PASS].ID.values

    obs_summary.to_csv(f"{save_path}", index=False)

    return snp_id


def cal_het_sd(het_hist: np.ndarray, het_range: Tuple[float, float], het_bin: int):
    bin_edges = np.linspace(het_range[0], het_range[1], num=het_bin + 1)
    margin = bin_edges[1] - bin_edges[0]
    bin_edges = (bin_edges + margin)[:het_bin]

    het_sum = np.sum(het_hist)
    het_mean = np.sum(het_hist * bin_edges) / het_sum
    het_std = np.sqrt(np.sum(((bin_edges - het_mean) ** 2) * het_hist) / (het_sum - 1))
    return het_std, het_mean
