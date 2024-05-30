import argparse
import time
import datetime
from multiprocessing import Pool

from gwasprs.gwasdata import GWASDataIterator
from rafael.usecases import BasicBfileQC, QuantitativeGWAS, Output
from rafael.logger import setup_logger
from rafael.configurations import load_yml
from rafael.utils import flatten


def main():
    args = parse_args()
    config_path, log_path = args.config, args.log
    config = load_yml(config_path)["config"]
    logger = setup_logger(log_path)

    bfile_path = config["bfile_path"]
    block_size = config["block_size"]
    nprocess = config["num_core"]

    # QC case
    qc = BasicBfileQC()

    t = time.perf_counter_ns()
    snp_list, sample_list, snp_id_table = qc.local_get_metadata(
        bfile_path, config["cov_path"],
        config.get('pheno_path', None),
        config.get('pheno_name', 'pheno'),
        autosome_only=config.get('autosome_only', True)
    )

    elapsed = time.perf_counter_ns() - t
    logger.info(f"local_get_metadata elapsed: {elapsed} ns")

    t = time.perf_counter_ns()
    snp_list = qc.global_match_snps(snp_list)
    elapsed = time.perf_counter_ns() - t
    logger.info(f"global_match_snps elapsed: {elapsed} ns")

    t = time.perf_counter_ns()
    allele_count, sample_count = qc.local_qc_stats(
        snp_list, config["qc_output_path"], snp_id_table, bfile_path)
    elapsed = time.perf_counter_ns() - t
    logger.info(f"local_qc_stats elapsed: {elapsed} ns")

    t = time.perf_counter_ns()
    snp_list = qc.global_qc_stats(allele_count, sample_count, snp_list,
        config["qc_output_path"], config["geno"], config["hwe"], config["maf"])
    elapsed = time.perf_counter_ns() - t
    logger.info(f"global_qc_stats elapsed: {elapsed} ns")

    t = time.perf_counter_ns()
    filtered_bed_path, filtered_cov_path = qc.local_filter_bfile(snp_list,
        config["qc_output_path"], config["cov_path"],
        sample_list, snp_id_table, bfile_path, config["mind"])
    elapsed = time.perf_counter_ns() - t
    logger.info(f"local_filter_bfile elapsed: {elapsed} ns")


    # Linear regression case
    linear = QuantitativeGWAS()

    t = time.perf_counter_ns()

    # TODO: refactor into strategy pattern
    logger.info(f"Load bed file {filtered_bed_path}.bed for GWAS.")
    loader = GWASDataIterator(
            bfile_path=filtered_bed_path,
            cov_path=filtered_cov_path,
            style="snp",
            snp_step=10000,
        )
    logger.info(f"There are {loader.bedreader.n_snp} SNPs and {loader.bedreader.n_sample} samples.")
    pool = Pool(nprocess)
    outputs = pool.map(linear.local_load_chunk_gwasdata, loader)
    pool.close()

    # Extract outputs and prepare the inputs
    genotype = flatten.list_of_arrays(list(zip(*outputs))[0], axis=1)
    phenotype = outputs[0][2]
    covariates = outputs[0][1]
    snp_info = flatten.list_of_frames(list(zip(*outputs))[4], axis=0)
    del outputs

    elapsed = time.perf_counter_ns() - t
    logger.info(f"local_load_gwasdata elapsed: {elapsed} ns")

    t = time.perf_counter_ns()
    XtX1, Xty1, n_models = linear.local_calculate_covariances(genotype, covariates, phenotype, block_size, nprocess)
    elapsed = time.perf_counter_ns() - t
    logger.info(f"local_calculate_covariances elapsed: {elapsed} ns")

    t = time.perf_counter_ns()
    beta = linear.global_fit_model([XtX1, ], [Xty1, ], n_models)
    elapsed = time.perf_counter_ns() - t
    logger.info(f"global_fit_model elapsed: {elapsed} ns")

    t = time.perf_counter_ns()
    sse, n_obs = linear.local_sse_and_obs(beta, phenotype, genotype, covariates, block_size, nprocess)
    elapsed = time.perf_counter_ns() - t
    logger.info(f"local_sse_and_obs elapsed: {elapsed} ns")

    t = time.perf_counter_ns()
    t_stat, pvals = linear.global_stats([sse, ], [n_obs, ])
    elapsed = time.perf_counter_ns() - t
    logger.info(f"global_stats elapsed: {elapsed} ns")

    output = Output()
    beta = beta.view().reshape(n_models, -1)[:,1]
    t_stat = t_stat.view()[:,1]
    pvals = pvals.view()[:,1]

    output.regression_results(
        snp_info, t_stat, pvals, beta, n_obs, config["regression_save_dir"]
    )

    logger.info(f"Complete computation.")


def parse_args():
    start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="configuration file path.", type=str)
    parser.add_argument("--log", help="log file path.", default=f"rafael-{start_time}.log", type=str)
    return parser.parse_args()


if __name__ == '__main__':
    main()
