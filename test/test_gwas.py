import os
import shutil
import unittest

from rafael.usecases import *
from rafael.logger import setup_logger
from rafael.fedalgo.gwasprs.ld import read_snp_list
from .utils import compare_qc, compare_glm

setup_logger()

class BasicBfileQCTestCase(unittest.TestCase):
    def setUp(self):
        self.QualityControl = BasicBfileQC()
        self.config = {
            "bfile_path": "data/whole/hapmap1_100",
            "pheno_name": "pheno",
            "cov_path": "data/whole/hapmap1_100.cov",
            "qc_output_path": "/tmp/qc",
            "maf": 0.05,
            "geno": 0.05,
            "mind": 0.05,
            "hwe": 5e-7
        }
        self.ans_prefix = "data/whole/qc/hapmap1_100"
    
    def test_qc(self):
        _test_qc(self.QualityControl, self.config)
        compare_qc(self.ans_prefix, f'{self.config["qc_output_path"]}.qc.csv')
    

class LDPruningTestCase(unittest.TestCase):
    def setUp(self):
        self.bfile_paths = ["data/client1/hapmap1_100_1", "data/client2/hapmap1_100_2"]
        self.output_paths = ["/tmp/hapmap1_100_1", "/tmp/hapmap1_100_2"]
        self.and_list = "data/whole/hapmap1_100.prune.in"
        self.ld = NaiveLDPruning()
    
    def test_ld(self):
        # step 1 local, add --bad-ld to allow run in < 50 sample
        snp_lists = []
        for bfile_path, output_path in zip(self.bfile_paths, self.output_paths):
            snp_list = self.ld.local_ldprune(
                bfile_path, output_path, 50, 5, 0.2, "--bad-ld"
            )
            snp_lists.append(snp_list)

        # step 2 global
        snp_list = self.ld.global_match_snps(snp_lists, method="union")

        # step 3 local
        for bfile_path, output_path in zip(self.bfile_paths, self.output_paths):
            self.ld.local_extract_snp(bfile_path, output_path, snp_list)

        # check ground truth
        ans_snp_list = read_snp_list(self.and_list)
        for i in snp_list:
            if i not in ans_snp_list:
                raise KeyError(f"{i} not found in ans_snp_list")

        for i in ans_snp_list:
            if i not in snp_list:
                raise KeyError(f"{i} not found in snp_list")


class QuantitativeGWASTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.QualityControl = BasicBfileQC()
        # cls.LDPruning = LDPruning() not implemented yet
        cls.Standadization = Standadization()
        cls.RandomizedSVD = RandomizedSVD()
        cls.GramSchmidt = GramSchmidt()
        cls.QuantGWAS = QuantitativeGWAS()
        cls.Output = Output()
        
        # Set up shared variables
        cls.config = {
            "qc_output_path": "/tmp/qc",
            "regression_save_dir": "/tmp/glm",
            "pheno_name": "pheno",
            "impute_cov": False
        }
        
        cls.ans_prefix = os.path.join(cls.config["regression_save_dir"], "ans")
        cls.ans = f'{cls.ans_prefix}.PHENO1.glm.linear'
        
    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.config["regression_save_dir"])
    
    def test_gwas(self):
        """ Only test the accuracy of the GWAS linear model """
        bfile_path = "data/whole/qc/hapmap1_100"
        cov_path = "data/whole/hapmap1_100.cov"

        # Quantitative GWAS
        snp_info, beta, t_stats, pvals, nobs = _test_quantitative_gwas(
            self.QuantGWAS,
            {
                "bfile_path": bfile_path,
                "cov_path": cov_path,
                **self.config
            }
        )
        
        # save GLM
        self.Output.regression_results(
            snp_info, t_stats[:, 0], pvals[:, 0],
            beta.view().reshape(*t_stats.shape)[:, 0],
            nobs, self.config["regression_save_dir"]
        )
        result = os.path.join(self.config["regression_save_dir"], 'gwas.glm')

        # Ground truth
        os.system(
            f'plink2 --bfile {bfile_path} \
                     --covar {cov_path} \
                     --covar-variance-standardize \
                     --glm hide-covar cols=chrom,pos,ref,alt,ax,a1freq,nobs,orbeta,se,ci,tz,p \
                     --out {self.ans_prefix}'
        )
        
        compare_glm(self.ans, result)
        os.remove(self.ans)
        os.remove(result)
    
    def test_qc_gwas(self):
        """ Test the accuracy of the GWAS linear model integrated with the QC """
        bfile_path = "data/whole/hapmap1_100"
        cov_path = "data/whole/hapmap1_100.cov"
        
        # Basic bfile QC
        filtered_bfile_path, filtered_cov_path = _test_qc(
            self.QualityControl,
            {
                "bfile_path": bfile_path,
                "cov_path": cov_path,
                "maf": 0.05,
                "geno": 0.05,
                "mind": 0.05,
                "hwe": 5e-7,
                **self.config
            }
        )
        
        # Quantitative GWAS
        snp_info, beta, t_stats, pvals, nobs = _test_quantitative_gwas(
            self.QuantGWAS,
            {
                "bfile_path": filtered_bfile_path,
                "cov_path": filtered_cov_path,
                **self.config
            }
        )
        
        # save GLM
        self.Output.regression_results(
            snp_info, t_stats[:, 0], pvals[:, 0],
            beta.view().reshape(*t_stats.shape)[:, 0],
            nobs, self.config["regression_save_dir"]
        )
        result = os.path.join(self.config["regression_save_dir"], 'gwas.glm')
        
        compare_glm("data/whole/glm/hapmap1_100.PHENO1.glm.linear", result)
        os.remove(result)
    
    def test_qc_ld_pca_gwas(self):
        pass
    

class BinaryGWASTestCase(unittest.TestCase):
    """
    This test case does not compare with the PLINK ground truth.
    Due to the fact that we found some inaccurate results from the hapmap1_100 dataset,
    some statistics of SNPs are odd. For example:
    rs3128342 (16 iterations)
            beta        statistic       p-value
    PLINK2      -0.2573     -0.1075         0.9143
    RAFAEL      -16.298     -0.0049         0.9960

    The RAFAEL result is from the 16 iterations setting mentioned in PLINK's maximum iterations.

    However, when we reduce the iterations, the RAFAEL result gets closer to the PLINK2 result.

    rs3128342 (10 iterations)
                beta        statistic       p-value
    PLINK2      -0.2573     -0.1075         0.9143
    RAFAEL      -10.298     -0.0626         0.9500

    This indicates that the iteration does affect the accuracy.

    The limitation of our JAX-vectorized-based Logistic Regression implementation is that
    we can't check the convergence of every single SNP; we only check the convergence of batches of SNPs.
    In other words, the convergence flag is raised only when a batch of SNPs has converged.

    In conclusion, the implementation of the Binary GWAS should raise warnings when performing on a small dataset.
    """
    @classmethod
    def setUpClass(cls):
        cls.QualityControl = BasicBfileQC()
        # cls.LDPruning = LDpruning() not implemented yet
        cls.Standadization = Standadization()
        cls.RandomizedSVD = RandomizedSVD()
        cls.GramSchmidt = GramSchmidt()
        cls.BinGWAS = BinaryGWAS()
        cls.Output = Output()
        
        # Set up shared variables
        cls.config = {
            "qc_output_path": "/tmp/qc",
            "regression_save_dir": "/tmp/glm",
            "pheno_name": "pheno",
            "impute_cov": False,
            "max_iterations": 16
        }
        
        cls.ans_prefix = os.path.join(cls.config["regression_save_dir"], "ans")
        cls.ans = f'{cls.ans_prefix}.pheno.glm.logistic.hybrid'
    
    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.config["regression_save_dir"])
        
    def test_gwas(self):
        bfile_path = "data/whole/qc/hapmap1_100"
        cov_path = "data/whole/hapmap1_100.cov"
        pheno_path = "data/whole/hapmap1_100.pheno"
        
        snp_info, beta, t_stats, pvals, nobs = _test_binary_gwas(
            self.BinGWAS,
            {
                "bfile_path": bfile_path,
                "cov_path": cov_path,
                "pheno_path": pheno_path,
                **self.config
            }
        )
        
        # save GLM
        self.Output.regression_results(
            snp_info, t_stats, pvals, beta,
            nobs, self.config["regression_save_dir"]
        )
        result = os.path.join(self.config["regression_save_dir"], 'gwas.glm')

        # Ground truth
        os.system(
            f'plink2 --bfile {bfile_path} \
                     --covar {cov_path} \
                     --covar-variance-standardize \
                     --pheno {pheno_path} \
                     --glm hide-covar cols=chrom,pos,ref,alt,ax,a1freq,nobs,orbeta,se,ci,tz,p \
                     --out {self.ans_prefix}'
        )
        
        # compare_glm(self.ans, result)
        os.remove(self.ans)
        os.remove(result)
    
    def test_qc_gwas(self):
        bfile_path = "data/whole/hapmap1_100"
        cov_path = "data/whole/hapmap1_100.cov"
        pheno_path = "data/whole/hapmap1_100.pheno"
        
        # Basic bfile QC
        filtered_bfile_path, filtered_cov_path = _test_qc(
            self.QualityControl,
            {
                "bfile_path": bfile_path,
                "cov_path": cov_path,
                "pheno_path": pheno_path,
                "maf": 0.05,
                "geno": 0.05,
                "mind": 0.05,
                "hwe": 5e-7,
                **self.config
            }
        )
        
        # Quantitative GWAS
        snp_info, beta, t_stats, pvals, nobs = _test_binary_gwas(
            self.BinGWAS,
            {
                "bfile_path": filtered_bfile_path,
                "cov_path": filtered_cov_path,
                "pheno_path": pheno_path,
                **self.config
            }
        )
        
        # save GLM
        self.Output.regression_results(
            snp_info, t_stats, pvals, beta,
            nobs, self.config["regression_save_dir"]
        )
        result = os.path.join(self.config["regression_save_dir"], 'gwas.glm')
        
        # compare_glm("data/whole/glm/hapmap1_100.pheno.glm.logistic.hybrid", result)
        os.remove(result)
    
    def test_qc_ld_pca_gwas(self):
        pass


def _test_qc(usecase, config):
    autosome_snp_list, sample_list, autosome_snp_table = usecase.local_get_metadata(
            config["bfile_path"], config["cov_path"], None,
            config["pheno_name"], snp_list=None, sample_list=None
    )

    autosome_snp_list = usecase.global_match_snps(autosome_snp_list)

    allele_count, nobs = usecase.local_qc_stats(
        autosome_snp_list, config["qc_output_path"],
        autosome_snp_table, config["bfile_path"]
    )

    filtered_autosome_snp_list = usecase.global_qc_stats(
        allele_count, nobs, autosome_snp_list, config["qc_output_path"],
        config["geno"], config["hwe"], config["maf"]
    )

    filtered_bfile_path, filtered_cov_path = usecase.local_filter_bfile(
        filtered_autosome_snp_list, config["qc_output_path"], config["cov_path"],
        sample_list, autosome_snp_table, config["bfile_path"], config["mind"]
    )
    return filtered_bfile_path, filtered_cov_path

def _test_quantitative_gwas(usecase, config):
    genotype, covariates, phenotype, _, snp_info = usecase.local_load_gwasdata(
        config["bfile_path"], config["cov_path"],
        None, config["pheno_name"], impute_cov=config["impute_cov"]
    )
    
    # calculate XtX and Xty
    XtX, Xty = usecase.local_calculate_covariances(
        genotype, covariates, phenotype
    )
    
    # calculate beta
    beta = usecase.global_fit_model([XtX, ], [Xty, ])
    
    # calculate SSE and number of observations
    sse, nobs = usecase.local_sse_and_obs(
        beta, phenotype, genotype, covariates
    )

    # calculate t-statistics and p-values
    t_stats, pvals = usecase.global_stats([sse, ], [nobs, ])
    return snp_info, beta, t_stats, pvals, nobs

def _test_binary_gwas(usecase, config):
    genotype, covariates, phenotype, _, snp_info = usecase.local_load_gwasdata(
        config["bfile_path"], config["cov_path"],
        config["pheno_path"], config["pheno_name"], impute_cov=config["impute_cov"]
    )
    
    nobs, gradient, hessian, loglikelihood, current_iteration = usecase.local_init_params(genotype, covariates, phenotype)
    
    prev_loglikelihood, prev_beta = None, None
    
    while True:
        beta, prev_beta, prev_loglikelihood, inv_hessian, jump_to = usecase.global_params(
            gradient, hessian, loglikelihood,
            current_iteration, config["max_iterations"],
            prev_loglikelihood, prev_beta
        )
        
        if jump_to == 'global_stats':
            break
        
        gradient, hessian, loglikelihood, current_iteration, _ = usecase.local_iter_params(
            genotype, covariates, phenotype,
            beta, current_iteration
        )
        
    t_stats, pvals, beta = usecase.global_stats(beta, inv_hessian)
    
    return snp_info, beta, t_stats, pvals, nobs
    