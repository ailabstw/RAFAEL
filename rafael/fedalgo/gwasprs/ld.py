from typing import List, Literal
import numpy as np
from numpy.typing import NDArray
import logging

from .utils import bash, PLINK2_PATH


def read_snp_list(file_path: str):
    with open(file_path, "r") as f:
        return list(map(lambda x: x.strip(), f.readlines()))


def write_snp_list(file_path: str, snps: List[str]):
    with open(file_path, "w") as f:
        f.write("\n".join(snps))


def prune_ld(
    bfile_path: str,
    out_path: str,
    win_size: int,
    step: int,
    r2: float,
    extra_arg: str = "",
) -> NDArray:

    plink_cmd = f"{PLINK2_PATH} --bfile {bfile_path} \
                                --allow-extra-chr \
                                --autosome \
                                --indep-pairwise {win_size} {step} {r2} \
                                {extra_arg} \
                                --out {out_path}"

    out, _ = bash(plink_cmd)
    for i in out:
        logging.debug(i)

    snp_list = read_snp_list(f"{out_path}.prune.in")

    return np.array(snp_list)


def match_snp_sets(
    snp_lists: List[NDArray], method: Literal["intersect", "union"]
) -> NDArray:
    
    snp_sets = [set(snps) for snps in snp_lists]
    
    if method == "intersect":
        snp_set = set.intersection(*snp_sets)
    elif method == "union":
        snp_set = set.union(*snp_sets)
    else:
        raise NotImplementedError
    
    return np.array(list(snp_set))


def extract_snps(bfile_path: str, out_path: str, snp_list: NDArray) -> str:
    plink_cmd = f"{PLINK2_PATH} --bfile {bfile_path} \
                                --allow-extra-chr \
                                --extract {out_path}.snp_list \
                                --make-bed \
                                --out {out_path}"

    write_snp_list(f"{out_path}.snp_list", snp_list.tolist())

    out, _ = bash(plink_cmd)
    for i in out:
        logging.debug(i)

    return out_path
