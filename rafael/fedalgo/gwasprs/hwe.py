import pandas as pd
from ctypes import c_double, c_int32

from .setup import setup_plink_hwp


PLINK_HWP = setup_plink_hwp()


def read_hardy(out_path: str):
    hwe = pd.read_csv(f"{out_path}.hardy", sep = r"\s+")
    hwe.AX = hwe.AX.astype(str)
    hwe.A1 = hwe.A1.astype(str)
    hwe["AA"] = hwe.TWO_AX_CT
    hwe["aa"] = hwe.HOM_A1_CT
    hwe.loc[hwe.A1 < hwe.AX, "AA"] = hwe.HOM_A1_CT
    hwe.loc[hwe.A1 < hwe.AX, "aa"] = hwe.TWO_AX_CT
    allele_count = hwe[["AA", "HET_A1_CT", "aa"]].values
    return allele_count


def cal_hwe_pvalue(het, hom1, hom2):
    pvalue_list = []
    for i in range(len(het)):
        pvalue = PLINK_HWP.HweP_py(int(het[i]), int(hom1[i]), int(hom2[i]), 0)
        pvalue_list.append(pvalue)

    return pvalue_list


def series_tolist(het, n):
    het = het.tolist()
    het = (c_int32 * n)(*het)
    return het

def cal_hwe_pvalue_vec(het, hom1, hom2):
    n = len(het)
    het = series_tolist(het, n)
    hom1 = series_tolist(hom1, n)
    hom2 = series_tolist(hom2, n)
    pvalue_list = [0.] * n
    pvalue_list = (c_double * n)(*pvalue_list)

    PLINK_HWP.HweP_vec_py(het, hom1, hom2, pvalue_list, n, 0)
    pvalue_list = pvalue_list[:]
    return pvalue_list
