import logging
import traceback

import numpy as np
import pandas as pd

from rafael.fedalgo.gwasprs.gwasdata import create_unique_snp_id


def compare_qc(plink_prefix: str, rafael_qc_path: str):
    # Read RAFAEL QC data
    rafael_data = pd.read_csv(rafael_qc_path)
    rafael_data = rafael_data.rename(columns={"ID": "NEW_ID"})

    # Read PLINK QC data
    missing_data = pd.read_csv(f"{plink_prefix}.vmiss", sep=r"\s+")[["F_MISS", "ID"]]
    freq_data = pd.read_csv(f"{plink_prefix}.afreq", sep=r"\s+")[["ALT_FREQS", "ID"]]
    hardy_data = pd.read_csv(f"{plink_prefix}.hardy", sep=r"\s+")
    hardy_data = hardy_data.rename(columns={"#CHROM": "CHR", "AX": "A2"})
    hardy_data["NEW_ID"], _ = create_unique_snp_id(hardy_data, to_byte=False)
    hardy_data = hardy_data[["P", "NEW_ID", "ID"]]

    # Merge PLINK QC data
    plink_data = hardy_data.merge(freq_data, on="ID", how="inner")
    plink_data = plink_data.merge(missing_data, on="ID", how="inner")

    # Merge PLINK and RAFAEL data
    merged_data = plink_data.merge(rafael_data, on="NEW_ID", how="inner")

    # Logging information
    logging.info(f"{len(plink_data.index)} SNPs in PLINK QC")
    logging.info(f"{len(rafael_data.index)} SNPs in RAFAEL QC")
    logging.info(f"{len(merged_data.index)} SNPs matched")

    # Calculate correlations
    hwe_cor = merged_data[["P", "HWE"]].corr("pearson").iloc[0, 1]
    miss_cor = merged_data[["MISSING", "F_MISS"]].corr("pearson").iloc[0, 1]
    freq_cor = merged_data[["ALT_FREQS", "MAF"]].corr("pearson").iloc[0, 1]

    # Log correlation results
    logging.info(f"HWE correlation: {hwe_cor}")
    logging.info(f"Missing correlation: {miss_cor}")
    logging.info(f"Frequency correlation: {freq_cor}")

    try:
        assert len(plink_data.index) == len(rafael_data.index)
        assert hwe_cor > 0.999
        assert miss_cor > 0.999
        assert freq_cor > 0.999

    except AssertionError:
        traceback.print_exc()
        logging.warning("This is merged data\n{}".format(merged_data))
        logging.warning("PLINK-only SNPs\n{}".format(plink_data[~plink_data.NEW_ID.isin(merged_data.NEW_ID)]))
        logging.warning("RAFAEL-only SNPs\n{}".format(rafael_data[~rafael_data.NEW_ID.isin(merged_data.NEW_ID)]))
        raise AssertionError


def compare_glm(plink_glm: str, rafael_glm: str, skip_count: bool = False, use_original_id: bool = True):
    """
    Compare results between PLINK and RAFAEL for Generalized Linear Model.
    """
    # Choose ID column based on the flag
    id_col = "rsID" if use_original_id else "ID"

    # Read and match data
    if 'linear' in plink_glm:
        plink_data = pd.read_csv(plink_glm, sep=r"\s+")[["ID", "P", "T_STAT", "BETA", "OBS_CT"]]

    elif 'logistic' in plink_glm:
        plink_data = pd.read_csv(plink_glm, sep=r"\s+")[["ID", "P", "Z_STAT", "OR", "OBS_CT"]]
        plink_data["OR"] = np.log(plink_data["OR"])
        plink_data.columns = ["ID", "P", "T_STAT", "BETA", "OBS_CT"]

    else:
        raise ValueError(f"Unknown GLM type: {plink_glm}")
    
    rafael_data = pd.read_csv(rafael_glm, sep="\t")[[id_col, "P", "T_STAT", "BETA", "CNT"]]
    rafael_data.columns = ["ID", "P", "T_STAT", "BETA", "OBS_CT"]

    # Logging information
    logging.info(f"{len(plink_data.index)} SNPs in PLINK GLM")
    logging.info(f"{len(rafael_data.index)} SNPs in RAFAEL GLM")

    # Merge data
    merged_data = plink_data.merge(rafael_data, on="ID", suffixes=["_plink", "_rafael"])
    logging.info(f"{len(merged_data.index)} SNPs matched")
    merged_data = merged_data.dropna(axis=0)

    # Ensure positive values for certain columns
    merged_data["T_STAT_plink"] = merged_data["T_STAT_plink"].abs()
    merged_data["T_STAT_rafael"] = merged_data["T_STAT_rafael"].abs()
    merged_data["BETA_plink"] = merged_data["BETA_plink"].abs()
    merged_data["BETA_rafael"] = merged_data["BETA_rafael"].abs()

    # Calculate correlations
    cnt_diff_sum = (merged_data["OBS_CT_plink"] - merged_data["OBS_CT_rafael"]).sum()
    p_cor = merged_data[["P_plink", "P_rafael"]].corr("pearson").iloc[0, 1]
    tstat_cor = merged_data[["T_STAT_plink", "T_STAT_rafael"]].corr("pearson").iloc[0, 1]
    beta_cor = merged_data[["BETA_plink", "BETA_rafael"]].corr("pearson").iloc[0, 1]

    # Log correlation results
    logging.info(f"Count difference: {cnt_diff_sum}")
    logging.info(f"P-value correlation: {p_cor}")
    logging.info(f"T-stat correlation: {tstat_cor}")
    logging.info(f"Beta correlation: {beta_cor}")

    try:
        if not skip_count:
            assert len(plink_data.index) == len(rafael_data.index)
            assert cnt_diff_sum == 0
        assert beta_cor > 0.999
        assert tstat_cor > 0.999
        assert p_cor > 0.999

    except AssertionError:
        traceback.print_exc()
        logging.warning(f"This is merged data\n{merged_data}")
        logging.warning(f"PLINK-only SNPs\n{plink_data[~plink_data.ID.isin(merged_data.ID)]}")
        logging.warning(f"RAFAEL-only SNPs\n{rafael_data[~rafael_data.ID.isin(merged_data.ID)]}")
        raise AssertionError
