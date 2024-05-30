from abc import ABCMeta
import logging
import os
from itertools import repeat
from multiprocessing import Pool
from typing import List, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import issparse
import jax.numpy as jnp
import jax.scipy as jsp
from jax import vmap, jit
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.cm import Accent

from .fedalgo import gwasprs
from .fedalgo.gwasprs import gwasplot
from .fedalgo.gwasprs.array import ArrayIterator
from .fedalgo import survival
from .utils import flatten


class UseCase(metaclass=ABCMeta):
    def __init__(self):
        pass

    @classmethod
    def return_variables(cls, method_name: str):
        raise Exception(f"{method_name} undefined.")


class BasicBfileQC(UseCase):
    """Perform basic QC from given arguments MAF, HWE, GENO, and MIND."""

    def __init__(self):
        super().__init__()

    @classmethod
    def return_variables(cls, method_name):
        if method_name == "local_get_metadata":
            return ["autosome_snp_list", "sample_list", "autosome_snp_table"]
        elif method_name == "global_match_snps":
            return ["autosome_snp_list"]
        elif method_name == "local_qc_stats":
            return ["allele_count", "n_obs"]
        elif method_name == "global_qc_stats":
            return ["filtered_autosome_snp_list"]
        elif method_name == "local_filter_bfile":
            return ["filtered_bfile_path", "filtered_cov_path"]
        else:
            super().return_variables(method_name)

    def local_get_metadata(
        self, bfile_path, cov_path, pheno_path, pheno_name, **kwargs
    ):
        """
        Get the metadata for each snp in the given bfile.

        Parameters
        ----------
            bfile_path : str
                The path to the bfile.
            cov_path : str
                The path to the covariate file.
            pheno_path : str
                The path to the phenotype file.
            pheno_name : str
                The column name of the phenotype.
            **kwargs : dict, optional
                Additional arguments for the GWASData.custom method.
        Returns
        -------
            autosome_snp_list : list
                The list of snp ids (format: CHR:POS:A1:A2).
            sample_list : list
                The list of sample ids. See GWASData.drop_missing_samples.
            autosome_snp_table : dict
                The dictionary mapping snp id (format: CHR:POS:A1:A2) to rsID.
        """
        autosome_snp_list, sample_list, autosome_snp_table = gwasprs.get_qc_metadata(
            bfile_path,
            cov_path,
            pheno_path,
            pheno_name,
            kwargs.get("autosome_only", True),
        )
        return autosome_snp_list, sample_list, autosome_snp_table

    def global_match_snps(self, autosome_snp_list):
        """
        Select the shared SNPs among all edges.

        Parameters
        ----------
            autosome_snp_list : list of list
                The lists of snp ids (format: CHR:POS:A1:A2) from edges.
        Returns
        -------
            autosome_snp_list : list
                The list of snp ids shared among all edges.
        """
        if isinstance(autosome_snp_list, list) and isinstance(
            autosome_snp_list[0], list
        ):
            autosome_snp_list = gwasprs.aggregations.Intersect()(*autosome_snp_list)
        logging.info(f"There are {len(autosome_snp_list)} snp matched among all edges.")
        return autosome_snp_list

    def local_qc_stats(
        self, autosome_snp_list, qc_output_path, autosome_snp_table, bfile_path
    ):
        """
        Calculate the allele count and number of observations for each snp.

        Parameters
        ----------
            autosome_snp_list : list
                The list of shared snp ids (format: CHR:POS:A1:A2) among all edges.
            qc_output_path : str
                The path to the output file for the QC report.
            autosome_snp_table : dict
                The dictionary mapping snp id (format: CHR:POS:A1:A2) to rsID.
            bfile_path : str
                The path to the bfile for the QC.
        Returns:
            allele_count : np.array
                The array (n_SNP, 3) storing allele counts for each snp.
            n_obs : int
                The number of samples in the edge.
        """
        autosome_rsID_list = list(
            map(lambda id: autosome_snp_table[id], autosome_snp_list)
        )
        allele_count, n_obs = gwasprs.qc.qc_stats(
            bfile_path, qc_output_path, autosome_rsID_list
        )
        return np.array(allele_count), int(n_obs)

    def global_qc_stats(
        self, allele_count, n_obs, autosome_snp_list, qc_output_path, geno, hwe, maf
    ):
        """
        Calculate the filtered snp list.

        Parameters
        ----------
            allele_count : list of np.array
                Allele counts for each snp in shape (n_edge, n_SNP, 3).
            n_obs : list of int
                The numbers of samples from edges.
            autosome_snp_list : list
                The list of shared snp ids among all edges.
            qc_output_output : str
                The path to the output file for the QC report.
            geno : float
                Filters out all SNPs with
                missing call rates exceeding the provided value.
            hwe : float
                Filters out all SNPs having Hardy-Weinberg equilibrium test
                p-value below the provided threshold.
            maf : float
                Filters out all variants with
                minor allele frequency below the provided threshold.
        Returns
        -------
            filtered_autosome_snp_list : list
                The list of snp ids (format: CHR:POS:A1:A2) passing the QC.
        """
        if isinstance(allele_count, (list, tuple)):
            allele_count = gwasprs.aggregations.SumUp()(*allele_count)
            n_obs = gwasprs.aggregations.SumUp()(*n_obs)

        qc_report = f"{qc_output_path}.qc.csv"
        filtered_snps = gwasprs.qc.filter_snp(
            allele_count, autosome_snp_list, n_obs, qc_report, geno, hwe, maf
        )
        filtered_autosome_snp_list = filtered_snps.tolist()
        return filtered_autosome_snp_list

    def local_filter_bfile(
        self,
        filtered_autosome_snp_list,
        qc_output_path,
        cov_path,
        sample_list,
        autosome_snp_table,
        bfile_path,
        mind,
    ):
        """
        Filter the bfile based on the filtered snp list.

        Parameters
        ----------
            filtered_autosome_snp_list : list
                The list of snp ids (format: CHR:POS:A1:A2) passing the QC.
            qc_output_output : str
                The path to the output file for the QC report.
            sample_list : list
                The list of sample ids. See GWASData.drop_missing_samples.
            autosome_snp_table : dict
                The dictionary mapping snp id (format: CHR:POS:A1:A2) to rsID.
            bfile_path : str
                The path to the bfile.
            mind : float
                Filters out all samples with
                missing call rates exceeding the provided value.
        Returns
        -------
            filtered_bfile_path : str
                The path to the filtered bfile.
            filtered_cov_path : str
                The path to the filtered covariate file.
        """
        autosome_rsID_list = list(
            map(lambda id: autosome_snp_table[id], filtered_autosome_snp_list)
        )

        filtered_bfile_path = gwasprs.qc.create_filtered_bed(
            bfile_path, qc_output_path, autosome_rsID_list, mind, sample_list
        )
        
        if cov_path is None:
            filtered_cov_path = None
        
        else:
            filtered_cov_path = gwasprs.qc.create_filtered_covariates(
                filtered_bfile_path, cov_path
            )
        return filtered_bfile_path, filtered_cov_path
    
    
class CovarProcessor(UseCase):
    def __init__(self):
        super().__init__()

    @classmethod
    def return_variables(cls, method_name):
        if method_name == "local_get_covariates":
            return ["covariates", "covar_names", "sample_ids"]
        elif method_name == "local_update_covariates":
            return ["cov_path"]
        else:
            super().return_variables(method_name)
            
    def local_get_covariates(self, bfile_path, cov_path, pheno_path, pheno_name):
        """
        Read covariates and drop samples with missing values.
        
        Parameters
        ----------
            bfile_path : str
                The path to the bfile.
            cov_path : str
                The path to the covariate file.
            pheno_path : str
                The path to the phenotype file.
            pheno_name : str
                The column name of the phenotype.
        
        Returns
        -------
            covariates : np.array
                The covariate matrix.
            covar_names : list
                The names of the covariates.
            sample_ids : pd.DataFrame
                The pandas DataFrame contains two columns, FID and IID.
        """
        if cov_path is None:
            return None, None, None
        
        fam, cov = gwasprs.gwasdata.format_sample_metadata(
            bfile_path, cov_path, pheno_path, pheno_name
        )
        keep_idx = gwasprs.gwasdata.index_non_missing_samples(fam, cov)
        covariates, _ = gwasprs.gwasdata.subset_samples(keep_idx, cov, list_is_idx=True)
        sample_ids = covariates.iloc[:, :2]
        covar_names = list(covariates.columns)[2:]
        
        return covariates.iloc[:, 2:].values, covar_names, sample_ids
    
    def local_update_covariates(
        self,
        cov_path: str,
        covar_matrix: NDArray,
        covar_names: List[str],
        cov_sample_ids: pd.DataFrame,
        pc_path: str,
        pc_sample_ids: pd.DataFrame,
        save_dir: str
        ):
        """
        Update the covariates, including standardized covariates and global PCs.
        
        Parameters
        ----------
            cov_path : str
                The path to the covariate file.
            covar_matrix : np.array
                The standardized covariates.
            covar_names : list
                The names of the covariates.
            cov_sample_ids : pd.DataFrame
                The pandas DataFrame contains two columns, FID and IID.
            pc_path : str
                The path to the pc file.
            pc_sample_ids : pd.DataFrame
                The pandas DataFrame contains two columns, FID and IID.
            save_dir : str
                The path to save updated covariates.
        
        Returns
        -------
            cov_path : str
                The path to the updated covariate file.
                
        Notes
        -----
        The covariate file is supposed to originate from the QC process, where samples are filtered.
        Similarly, the number of samples in the PC file should also be a result of the QC process,
        with LD-pruning not affecting the sample count.
        However, differences in sample counts between the covariate file and the PC file can occur
        because during covariate standardization after QC, samples with any missing values are dropped.
        This leads to a different set of samples in the PC file, which includes all samples after QC.
        """

        if covar_names is None:
            cov = None
        else:
            cov = pd.DataFrame(covar_matrix, columns=covar_names)
            cov = pd.concat([cov_sample_ids, cov], axis=1)
            
            # Meets the same length requirement in GWASData
            raw_cov = gwasprs.reader.CovReader(cov_path).read()
            raw_cov = raw_cov.loc[:, ['FID', 'IID']]
            cov = pd.merge(raw_cov, cov, on=['FID', 'IID'], how='outer')
            

        if os.path.exists(pc_path):
            pc = pd.read_csv(pc_path)
            pc_sample_ids = pd.DataFrame(pc_sample_ids, columns=["FID", "IID"])
            pc = pd.concat([pc_sample_ids, pc], axis=1)
            
            if cov is not None:
                assert len(cov) == len(pc),\
                    f"Unexpected number of samples, cov: {len(cov)} samples and pc: {len(pc)} samples."
                cov = pd.merge(cov, pc, on=['FID', 'IID'], how='inner')
            
            else:
                cov = pc


        if cov is not None:
            cov_path = os.path.join(save_dir,'stdz.merged.cov')
            cov.to_csv(f"{cov_path}", sep='\t', index=None)
            return cov_path
        
        else:
            return None


def add_bias_and_dropna(covariates, phenotype, genotype):
    logging.info("Adding bias (inner).")
    Xs = gwasprs.block.BlockDiagonalMatrix([])
    ys = []
    for g in ArrayIterator(genotype, axis=1):
        X = gwasprs.regression.add_bias(g, axis=1)
        X = gwasprs.array.concat((X, covariates))
        idx = np.logical_and(
            gwasprs.mask.isnonnan(X, axis=1),
            gwasprs.mask.isnonnan(np.expand_dims(phenotype, -1), axis=1),
        )
        Xs.append(X[idx])
        ys.append(phenotype[idx])
    logging.info("Adding bias done (inner).")
    return Xs, np.concatenate(ys)


def calculate_covariances(genotype, covariates, phenotype):
    logging.info("Calculating covariances (inner).")
    X, y = add_bias_and_dropna(covariates, phenotype, genotype)
    XtX = gwasprs.stats.blocked_unnorm_autocovariance(X)
    Xty = gwasprs.stats.blocked_unnorm_covariance(X, y)
    return XtX, Xty


def calculate_sse(genotype, covariates, phenotype, beta):
    logging.info("Calculating SSE (inner).")
    sse, n_obs = [], []
    beta = beta.view().reshape(
        -1,
    )
    X, y = add_bias_and_dropna(covariates, phenotype, genotype)
    n_obs = np.array([sh[0] for sh in X.blockshapes])
    client_model = gwasprs.regression.BlockedLinearRegression(
        beta=beta, nmodels=X.nblocks
    )
    sse = client_model.sse(X, y)
    return sse, n_obs


class QuantitativeGWAS(UseCase):
    """
    Linear regression for performing continuous phenotype GWAS
    under the federated scheme.
    This work is inspired by the work of sPLINK,
    see https://github.com/tum-aimed/splink.
    """

    def __init__(self):
        super().__init__()

    @classmethod
    def return_variables(cls, method_name: str):
        if method_name == "local_load_gwasdata":
            return ["genotype", "covariates", "phenotype", "sample_info", "snp_info"]
        elif method_name == "local_calculate_covariances":
            return ["XtX", "Xty", "n_model"]
        elif method_name == "global_fit_model":
            return ["beta"]
        elif method_name == "local_sse_and_obs":
            return ["sse", "n_obs"]
        elif method_name == "global_stats":
            return ["t_stat", "pval"]
        else:
            super().return_variables(method_name)

    def local_load_gwasdata(
        self, filtered_bfile_path, filtered_cov_path, pheno_path, pheno_name, **kwargs
    ):
        """
        Load gwasdata from the given bfile.

        Parameters
        ----------
            filtered_bfile_path : str
                The path to the bfile.
            filtered_cov_path : str
                The path to the covariate file.
            pheno_path : str
                The path to the phenotype file.
            pheno_name : str
                The column name of the phenotype.
            **kwargs : dict, optional
                Additional arguments for the GWASData.custom method.
        Returns
        -------
            genotype : np.array
                Genotype matrix in shape (n_sample, n_SNP).
            covariates : np.array
                Covariate matrix in shape (n_sample, n_covariate).
            phenotype : np.array
                Phenotype vector in shape (n_sample,).
        See also
        --------
        GWASData.subset
        GWASData.impute_covariates
        GWASData.add_unique_snp_id
        """
        logging.info("Loading bed file.")
        data = gwasprs.GWASData.read(
            filtered_bfile_path, filtered_cov_path, pheno_path, pheno_name
        )
        data.custom(add_unique_snp_id=True, **kwargs)
        genotype = data.genotype
        covariates = data.covariate.iloc[:, 2:].values
        phenotype = data.phenotype["PHENO1"].values
        sample_info = data.sample_id
        snp_info = data.snp
        return genotype, covariates, phenotype, sample_info, snp_info

    def local_load_chunk_gwasdata(self, chunk_data, **kwargs):
        if kwargs.get("pheno") is not None:
            chunk_data = gwasprs.GWASData(
                chunk_data.genotype,
                gwasprs.gwasdata.format_fam(chunk_data.phenotype, kwargs.get("pheno")),
                chunk_data.snp,
                chunk_data.covariate,
            )

        chunk_data.custom(add_unique_snp_id=True, **kwargs)
        genotype = chunk_data.genotype
        if chunk_data.covariate is None:
            covariates = None
        else:
            covariates = chunk_data.covariate.iloc[:, 2:].values
        phenotype = chunk_data.phenotype["PHENO1"].values
        sample_info = chunk_data.sample_id
        snp_info = chunk_data.snp

        return genotype, covariates, phenotype, sample_info, snp_info

    def local_calculate_covariances(
        self, genotype, covariates, phenotype, block_size, nprocess
    ):
        """
        Calculate block diagonal (auto)covariance matrix

        The shape of X matrix is (n_SNP, n_sample, 1(SNP)+n_covariate+1(bias)),
        however, the number of samples for each SNP may vary
        because samples with missing value will be filtered out.

        Parameters
        ----------
            genotype : np.array
                Genotype matrix in shape (n_sample, n_SNP).
            covariates : np.array
                Covariate matrix in shape (n_sample, n_covariate).
            phenotype : np.array
                Phenotype vector in shape (n_sample,).
            block_size : int
                Number of SNPs to run in a block in a process.
            nprocess : int
                Number of processes to run the task.
        Returns
        -------
            XtX : np.array
                Block diagonal covariance matrix
                in shape (n_SNP*(n_covariate+2), n_SNP*(n_covariate+2)).
            Xty : np.array
                Vector in shape (n_SNP*(n_covariate+2), 1).
        """
        logging.info("Calculating (auto)covariance matrix.")
        n_SNP = genotype.shape[1]
        split_idx = [
            min(start + block_size, n_SNP) for start in range(0, n_SNP, block_size)
        ]
        genotype = np.split(genotype, split_idx, axis=1)[:-1]
        num_blocks = len(genotype)
        pool = Pool(nprocess)
        results = pool.starmap(
            calculate_covariances,
            zip(
                genotype, repeat(covariates, num_blocks), repeat(phenotype, num_blocks)
            ),
        )
        pool.close()
        pool.join()
        XtX, Xty = list(zip(*results))

        return flatten.list_of_blocks(XtX), np.concatenate(Xty), n_SNP

    def global_fit_model(self, XtX, Xty, n_model, **kwargs):
        """
        Fit the linear regression model

        Parameters
        ----------
            XtX : list of np.array
                Block diagonal covariance matrix in shape
                (n_SNP*(n_covariate+2), n_SNP*(n_covariate+2)) from edges.
            Xty : list of np.array
                Vector in shape (n_SNP*(n_covariate+2), 1) from edges.
        Returns
        -------
            beta : np.array
                Coefficients in shape (n_SNP*(n_covariate+2), 1).
        """
        logging.info("Fitting linear regression model.")
        if isinstance(XtX, list) and isinstance(Xty, list):
            XtX = gwasprs.aggregations.SumUp()(*XtX)
            Xty = gwasprs.aggregations.SumUp()(*Xty)

        if issparse(XtX):
            XtX = XtX.todense()

        self.server_model = gwasprs.regression.BlockedLinearRegression(
            XtX=XtX, Xty=Xty, nmodels=n_model
        )
        beta = self.server_model.coef
        self.XtX = XtX
        return beta

    def local_sse_and_obs(
        self, beta, phenotype, genotype, covariates, block_size, nprocess
    ):
        """
        Calculate sum of square error and number of observations

        Parameters
        ----------
            beta : np.array
                Coefficients in shape (n_SNP*(n_covariate+2), 1).
            phenotype : np.array
                Phenotype vector in shape (n_sample,).
            genotype : np.array
                Genotype matrix in shape (n_sample, n_SNP).
            covariates : np.array
                Covariate matrix in shape (n_sample, n_covariate).
        Returns
        -------
            sse : np.array
                Sum of square error in shape (n_SNP*(n_covariate+2),).
            n_obs : np.array
                Number of observations in shape (n_SNP,).
        """
        logging.info("Calculating SSE and number of observations.")
        n_SNP = genotype.shape[1]
        split_idx = [
            min(start + block_size, n_SNP) for start in range(0, n_SNP, block_size)
        ]
        genotype = np.split(genotype, split_idx, axis=1)[:-1]
        beta = beta.view().reshape(n_SNP, -1)
        beta = np.split(beta, split_idx, axis=0)[:-1]
        num_blocks = len(genotype)
        pool = Pool(nprocess)
        results = pool.starmap(
            calculate_sse,
            zip(
                genotype,
                repeat(covariates, num_blocks),
                repeat(phenotype, num_blocks),
                beta,
            ),
        )
        pool.close()
        pool.join()
        sse, n_obs = list(zip(*results))

        return flatten.list_of_arrays(sse, axis=0), flatten.list_of_arrays(
            n_obs, axis=0
        )

    def global_stats(self, sse, n_obs):
        """
        Calculate t statistics and p-values

        Parameters
        ----------
            sse : list of np.array
                Sum of square error in shape (n_SNP*(n_covariate+2),).
            n_obs : list of np.array
                Number of observations in shape (n_SNP,).
        Returns
        -------
            t_stat : np.array
                t-statistics in shape (n_SNP, n_covariate+2).
            pval : np.array
                p-values in shape (n_SNP, n_covariate+2).
        """
        logging.info("Calculating t statistics.")
        if isinstance(sse, list) and isinstance(n_obs, list):
            sse = gwasprs.aggregations.SumUp()(*sse)
            n_obs = gwasprs.aggregations.SumUp()(*n_obs)

        dof = self.server_model.dof(n_obs)
        t_stat = self.server_model.t_stats(sse, self.XtX, dof)
        t_stat = np.reshape(t_stat, (dof.shape[0], -1))

        logging.info("Calculating p values for t statistics.")
        pval = gwasprs.stats.t_dist_pvalue(t_stat, np.expand_dims(dof, -1))
        return t_stat, pval


class LinearRegression(UseCase):
    """
    The federated linear regression implementation,
    which equals to the `sklearn.linear_model.LinearRegression()`.
    See also `test_regression.py`.
    """

    def __init__(self):
        super().__init__()

    @classmethod
    def return_variables(cls, method_name: str):
        if method_name == "local_calculate_covariances":
            return ["XtX", "Xty", "X"]
        elif method_name == "global_fit_model":
            return ["beta"]
        elif method_name == "local_sse_and_obs":
            return ["sse", "n_obs"]
        elif method_name == "global_stats":
            return ["t_stat", "pval"]
        else:
            super().return_variables(method_name)

    def local_calculate_covariances(self, X, y):
        """
        Calulates the covariance matrix

        Parameters
        ----------
            X : np.array
                The feature matrix with shape (n_samples, n_features).
            y : np.array
                The target vector with shape (n_samples,)
        Returns
        -------
            XtX : np.array
                The covariance matrix with shape (n_features+1, n_features+1).
            Xty : np.array
                The dot product of feature matrix and target vector with shape (n_features+1,).
        """
        assert not np.any(np.isnan(X)), "Feature matrix contains NaN values."
        assert not np.any(np.isnan(y)), "Target array contains NaN values."
        X = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
        XtX = np.dot(X.T, X)
        Xty = np.dot(X.T, y)
        return XtX, Xty, X

    def global_fit_model(self, XtX, Xty):
        """
        Fit the linear regression model

        Parameters
        ----------
            XtX : list of np.array
                The covariance matrix with shape (n_features+1, n_features+1).
            Xty : np.array
                The dot product of feature matrix and target vector with shape (n_features+1,).
        Returns
        -------
            beta : np.array
                Coefficients of the linear regression with shape (n_features+1,).
        """
        if isinstance(XtX, list) and isinstance(Xty, list):
            XtX = gwasprs.aggregations.SumUp()(*XtX)
            Xty = gwasprs.aggregations.SumUp()(*Xty)

        self.server_model = gwasprs.regression.LinearRegression(XtX=XtX, Xty=Xty)
        beta = self.server_model.coef
        self.XtX = XtX
        return beta

    def local_sse_and_obs(self, X, y, beta):
        """
        Calculate sum of square error and number of observations

        Parameters
        ----------
            X : np.array
                The feature matrix with shape (n_samples, n_features+1).
            y : np.array
                The target vector with shape (n_samples,)
            beta : np.array
                Coefficients of the linear regression with shape (n_features+1,).
        Returns
        -------
            sse : np.array
                Sum of square error in shape (n_features+1,).
            n_obs : int
                Number of observations.
        """
        client_model = gwasprs.regression.LinearRegression(beta=beta)
        sse = client_model.sse(X, y)
        n_obs = X.shape[0]
        return sse, n_obs

    def global_stats(self, sse, n_obs):
        """
        Calculate t statistics and p-values

        Parameters
        ----------
            sse : list of np.array
                Sum of square error in shape (n_features+1,).
            n_obs : list of int
                Number of observations.
        Returns
        -------
            t_stat : np.array
                t-statistics in shape (n_features+1,).
            pval : np.array
                p-values in shape (n_features+1,).
        """
        if isinstance(sse, list) and isinstance(n_obs, list):
            sse = gwasprs.aggregations.SumUp()(*sse)
            n_obs = gwasprs.aggregations.SumUp()(*n_obs)

        dof = self.server_model.dof(n_obs)
        t_stat = self.server_model.t_stats(sse, self.XtX, dof)
        pval = gwasprs.stats.t_dist_pvalue(t_stat, dof)
        return t_stat, pval


""" Logistic Helper Functions """


@jit
def preprocess_x_y(genotype, covariates, phenotype):
    genotype = jnp.expand_dims(genotype, axis=-1)
    X = jnp.concatenate((genotype, covariates), axis=1)
    mask = jnp.expand_dims(gwasprs.mask.isnonnan(X, axis=1), -1)
    X_mask = gwasprs.impute_with(X) * mask
    Y_mask = phenotype * jnp.squeeze(mask)

    # Set the max value to 1, else to 0
    max_value = jnp.max(Y_mask)
    Y_mask = jnp.where(Y_mask == max_value, 1, 0)

    return X_mask, Y_mask


@jit
def generate_Xy(GT, covariates, phenotype):
    return vmap(preprocess_x_y, (1, None, None), (0, 0))(GT, covariates, phenotype)


def converged(prev_loglikelihood, loglikelihood, threshold=1e-4):
    if jnp.isnan(prev_loglikelihood).all():
        return False
    else:
        delta_loglikelihood = jnp.abs(prev_loglikelihood - loglikelihood)
        return True if (delta_loglikelihood < threshold).all() else False


def to_list(*args):
    return [[arg] for arg in args]


class BinaryGWAS(UseCase):
    """
    Logistic regression for performing binary phenotype GWAS
    under the federated scheme.
    This work is inspired by the work of sPLINK,
    see https://github.com/tum-aimed/splink.
    """

    def __init__(self):
        super().__init__()

    @classmethod
    def return_variables(cls, method_name):
        if method_name == "local_load_gwasdata":
            return ["genotype", "covariates", "phenotype", "sample_info", "snp_info"]
        elif method_name == "local_init_params":
            return [
                "n_obs",
                "gradient",
                "hessian",
                "loglikelihood",
                "current_iteration",
            ]
        elif method_name == "global_params":
            return ["beta", "prev_beta", "prev_loglikelihood", "inv_hessian", "jump_to"]
        elif method_name == "local_iter_params":
            return [
                "gradient",
                "hessian",
                "loglikelihood",
                "current_iteration",
                "jump_to",
            ]
        elif method_name == "global_stats":
            return ["t_stat", "pval", "beta"]
        else:
            super().return_variables(method_name)

    def local_load_gwasdata(
        self, filtered_bfile_path, filtered_cov_path, pheno_path, pheno_name, **kwargs
    ):
        """
        Load gwasdata from the given bfile.

        Parameters
        ----------
            filtered_bfile_path : str
                The path to the bfile.
            filtered_cov_path : str
                The path to the covariate file.
            pheno_path : str
                The path to the phenotype file.
            pheno_name : str
                The column name of the phenotype.
            **kwargs : dict, optional
                Additional arguments for the GWASData.custom method.
        Returns
        -------
            genotype : np.array
                Genotype matrix in shape (n_sample, n_SNP).
            covariates : np.array
                Covariate matrix in shape (n_sample, n_covariate).
            phenotype : np.array
                Phenotype vector in shape (n_sample,).
        See also
        --------
        GWASData.subset
        GWASData.impute_covariates
        GWASData.add_unique_snp_id
        """
        logging.info("Loading bed file.")
        data = gwasprs.GWASData.read(
            filtered_bfile_path, filtered_cov_path, pheno_path, pheno_name
        )
        data.custom(add_unique_snp_id=True, **kwargs)
        genotype = data.genotype
        covariates = data.covariate.iloc[:, 2:].values
        phenotype = data.phenotype["PHENO1"].values
        sample_info = data.sample_id
        snp_info = data.snp
        return genotype, covariates, phenotype, sample_info, snp_info

    def local_load_chunk_gwasdata(self, chunk_data, **kwargs):
        if kwargs.get("pheno") is not None:
            chunk_data = gwasprs.GWASData(
                chunk_data.genotype,
                gwasprs.gwasdata.format_fam(chunk_data.phenotype, kwargs.get("pheno")),
                chunk_data.snp,
                chunk_data.covariate,
            )

        chunk_data.custom(add_unique_snp_id=True, **kwargs)
        genotype = jnp.array(chunk_data.genotype)
        if chunk_data.covariate is None:
            covariates = None
        else:
            covariates = jnp.array(chunk_data.covariate.iloc[:, 2:].values)
        phenotype = jnp.array(chunk_data.phenotype["PHENO1"].values)
        sample_info = chunk_data.sample_id
        snp_info = chunk_data.snp

        return genotype, covariates, phenotype, sample_info, snp_info

    def local_init_params(self, genotype, covariates, phenotype):
        """
        Prepare parameters, X, y, gradient, hessian
        and loglikelihood for the GWAS model.

        Parameters
        ----------
            genotype : np.array
                Genotype matrix in shape (n_sample, n_SNP).
            covariates : np.array
                Covariate matrix in shape (n_sample, n_covariate).
            phenotype : np.array
                Phenotype vector in shape (n_sample,).
        Returns
        -------
            n_obs : np.array
                Number of observations in shape (n_SNP,).
            gradient : np.array
                Gradient matrix in shape (n_SNP, n_covariate+2).
            hessian : np.array
                Hessian matrix in shape (n_SNP, n_covariate+2, n_covariate+2).
            loglikelihood : np.array
                Loglikelihood vector in shape (n_SNP,).
            current_iteration : int
                Current iteration, starts from 1.
        """
        # Initialize
        if covariates is None:
            covariates = jnp.ones((genotype.shape[0], 1))
        else:
            covariates = jnp.array(gwasprs.regression.add_bias(covariates))
        X, y = generate_Xy(genotype, covariates, phenotype)
        beta = jnp.zeros((genotype.shape[1], covariates.shape[1] + 1))
        n_obs = gwasprs.mask.nonnan_count(genotype)  # used in output file
        current_iteration = 0

        # Update the parameters
        model = gwasprs.regression.BatchedLogisticRegression(beta)
        gradient = model.gradient(X, y)
        hessian = model.hessian(X)
        loglikelihood = model.loglikelihood(X, y)
        current_iteration += 1
        return n_obs, gradient, hessian, loglikelihood, current_iteration

    def global_params(
        self,
        gradient,
        hessian,
        loglikelihood,
        current_iteration,
        max_iterations,
        prev_loglikelihood,
        prev_beta,
    ):
        """
        Update global parameters, gradient, hessian, loglikelihood and beta

        Parameters
        ----------
            gradient : np.array
                Gradient matrix in shape (n_SNP, n_covariate+2) from edges.
            hessian : np.array
                Hessian matrix in shape (n_SNP, n_covariate+2, n_covariate+2) from edges.
            loglikelihood : np.array
                Loglikelihood vector in shape (n_SNP,) from edges.
            current_iteration : int
                Current iteration, starts from 1.
            max_iterations : int
                Maximum number of iterations.
            prev_loglikelihood : np.array
                Previous loglikelihood vector in shape (n_SNP,).
            prev_beta : np.array
                Previous beta vector in shape (n_SNP, n_covariate+2).
        Returns
        -------
            beta : np.array
                Updated beta vector in shape (n_SNP, n_covariate+2).
            prev_beta : np.array
                Updated beta vector in shape (n_SNP, n_covariate+2).
            prev_loglikelihood : np.array
                Updated loglikelihood vector in shape (n_SNP,).
            inv_hessian : np.array
                If meeting the termination criterion,
                the inverse hessian (n_SNP, n_covariate+2, n_covariate+2).
                Otherwise, None.
            jump_to : str
                If meeting the termination criterion, 'global_stats'.
                Otherwise, 'local_iter_params'.
        """
        # Make sure the input parameters have the edge axis
        if len(jnp.array(gradient).shape) != 3:
            gradient, hessian, loglikelihood = to_list(gradient, hessian, loglikelihood)

        # Initialize the beta and loglikelihood
        if prev_beta is None:
            n_snp, n_feature, _ = jnp.array(hessian[0]).shape
            prev_beta = jnp.zeros((n_snp, n_feature))
            prev_loglikelihood = jnp.full((n_snp,), jnp.nan)

        # Aggregate
        gradient = gwasprs.aggregations.SumUp()(*gradient)
        hessian = gwasprs.aggregations.SumUp()(*hessian)
        loglikelihood = gwasprs.aggregations.SumUp()(*loglikelihood)

        # Update the beta and loglikelihood
        model = gwasprs.regression.BatchedLogisticRegression(prev_beta)
        beta = model.beta(gradient, hessian)

        # Check termination criterion
        if (
            converged(prev_loglikelihood, loglikelihood)
            or current_iteration == max_iterations
        ):
            inv_hessian = gwasprs.linalg.batched_inv(hessian)
            jump_to = "global_stats"
        else:
            inv_hessian = None
            jump_to = "local_iter_params"

        # Update previous parameters
        prev_beta = beta
        prev_loglikelihood = loglikelihood
        return beta, prev_beta, prev_loglikelihood, inv_hessian, jump_to

    def local_iter_params(
        self, genotype, covariates, phenotype, beta, current_iteration
    ):
        """
        Update local parameters from global beta

        Parameters
        ----------
            genotype : np.array
                Genotype matrix in shape (n_sample, n_SNP).
            covariates : np.array
                Covariate matrix in shape (n_sample, n_covariate).
            phenotype : np.array
                Phenotype vector in shape (n_sample,).
            beta : np.array
                Global beta matrix in shape (n_SNP, n_covariate+2).
            current_iteration : int
                Current iteration.
        Returns
        -------
            gradient : np.array
                Gradient matrix in shape (n_SNP, n_covariate+2).
            hessian : np.array
                Hessian matrix in shape (n_SNP, n_covariate+2, n_covariate+2).
            loglikelihood : np.array
                Loglikelihood vector in shape (n_SNP,).
            current_iteration : int
                Updated current iteration.
            jump_to : str
                'global_params' to update global parameters.
        """
        if covariates is None:
            covariates = jnp.ones((genotype.shape[0], 1))
        else:
            covariates = jnp.array(gwasprs.regression.add_bias(covariates))
        X, y = generate_Xy(genotype, covariates, phenotype)

        # Update the parameters
        model = gwasprs.regression.BatchedLogisticRegression(beta)
        gradient = model.gradient(X, y)
        hessian = model.hessian(X)
        loglikelihood = model.loglikelihood(X, y)
        current_iteration += 1
        jump_to = "global_params"
        return gradient, hessian, loglikelihood, current_iteration, jump_to

    def global_stats(self, beta, inv_hessian):
        """
        Calculates the global statistics

        Parameters
        ----------
            beta : np.array
                Global beta matrix in shape (n_SNP, n_covariate+2).
            inv_hessian : np.array
                Inverse hessian (n_SNP, n_covariate+2, n_covariate+2).
        Returns
        -------
            t_stat : np.array
                T-statistic vector in shape (n_SNP,).
            pval : np.array
                P-value vector in shape (n_SNP,).
            beta : np.array
                Global beta vector in shape (n_SNP,).
        """
        beta = beta.astype('float64')
        inv_hessian = inv_hessian.astype('float64')
        t_stat, pval = gwasprs.stats.batched_logistic_stats(beta, inv_hessian)
        return t_stat[:, 0], pval[:, 0], beta[:, 0]


class LogisticRegression(UseCase):
    """
    The federated logistic regression implementation,
    which equals to the `sklearn.linear_model.LogisticRegression(penalty=None)`.
    See also `test_regression.py`.
    """

    def __init__(self):
        super().__init__()

    @classmethod
    def return_variables(cls, method_name):
        if method_name == "local_load_gwasdata":
            return [
                "X",
                "y",
                "gradient",
                "hessian",
                "loglikelihood",
                "current_iteration",
            ]
        elif method_name == "local_init_params":
            return [
                "X",
                "y",
                "n_obs",
                "gradient",
                "hessian",
                "loglikelihood",
                "current_iteration",
            ]
        elif method_name == "global_params":
            return ["beta", "prev_beta", "prev_loglikelihood", "inv_hessian", "jump_to"]
        elif method_name == "local_iter_params":
            return [
                "gradient",
                "hessian",
                "loglikelihood",
                "current_iteration",
                "jump_to",
            ]
        elif method_name == "global_stats":
            return ["t_stat", "pval", "beta"]
        else:
            super().return_variables(method_name)

    def local_init_params(self, X, y):
        """
        Prepare parameters, X, y, gradient, hessian and loglikelihood.

        Parameters
        ----------
            X : np.array
                The feature matrix with shape (n_samples, n_features).
            y : np.array
                The target vector with shape (n_samples,)
        Returns
        -------
            X : np.array
                X matrix in shape (n_samples, n_features+1).
            y : np.array
                Vector in shape (n_samples,).
            gradient : np.array
                Gradient vector in shape (n_features+1,).
            hessian : np.array
                Hessian matrix in shape (n_features+1, n_features+1).
            loglikelihood : float
                Loglikelihood.
            current_iteration : int
                Current iteration, starts from 1.
        """
        assert not np.any(np.isnan(X)), "Feature matrix contains NaN values."

        # Check y
        assert not np.any(np.isnan(y)), "Target array contains NaN values."
        uniques = np.unique(y)
        assert len(uniques) == 2, "The number of unique values != 2."
        if uniques[0] != 0.0:
            logging.warn(
                f"Auto-binarize the values {uniques[0]} to 0, {uniques[1]} to 1."
            )
            y = np.where(y < np.mean(uniques), 0, 1)

        # Add bias
        X = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)

        # Initialize logistic regression
        beta = np.zeros((X.shape[1],))
        model = gwasprs.regression.LogisticRegression(beta)
        gradient = model.gradient(X, y)
        hessian = model.hessian(X)
        loglikelihood = model.loglikelihood(X, y)
        current_iteration = 1

        return X, y, gradient, hessian, loglikelihood, current_iteration

    def global_params(
        self,
        gradient,
        hessian,
        loglikelihood,
        current_iteration,
        max_iterations,
        prev_loglikelihood=None,
        prev_beta=None,
    ):
        """
        Update global parameters, gradient, hessian, loglikelihood and beta

        Parameters
        ----------
            gradient : list of np.array
                Gradient vector in shape (n_features+1,).
            hessian : list of np.array
                Hessian matrix in shape (n_features+1, n_features+1).
            loglikelihood : list of float
                Loglikelihood.
            current_iteration : int
                Current iteration, starts from 1.
            max_iterations : int
                Maximum number of iterations.
            prev_loglikelihood : float
                Loglikelihood.
            prev_beta : np.array
                Previous beta vector in shape (n_features+1,).
        Returns
        -------
            beta : np.array
                Updated beta vector in shape (n_features+1,).
            prev_beta : np.array
                Updated beta vector in shape (n_features+1,).
            prev_loglikelihood : float
                Updated loglikelihood.
            inv_hessian : np.array
                If meeting the termination criterion,
                the inverse hessian (n_features+1, n_features+1).
                Otherwise, None.
            jump_to : str
                If meeting the termination criterion, 'global_stats'.
                Otherwise, 'local_iter_params'.
        """
        if (
            isinstance(gradient, list)
            and isinstance(hessian, list)
            and isinstance(loglikelihood, list)
        ):
            gradient = gwasprs.aggregations.SumUp()(*gradient)
            hessian = gwasprs.aggregations.SumUp()(*hessian)
            loglikelihood = gwasprs.aggregations.SumUp()(*loglikelihood)

        if prev_beta is None:
            prev_beta = np.zeros((hessian.shape[0],))
            prev_loglikelihood = np.nan

        model = gwasprs.regression.LogisticRegression(prev_beta)
        beta = model.beta(gradient, hessian)

        if (
            converged(prev_loglikelihood, loglikelihood)
            or current_iteration == max_iterations
        ):
            inv_hessian = gwasprs.linalg.inv(hessian)
            jump_to = "global_stats"
        else:
            inv_hessian = None
            jump_to = "local_iter_params"

        prev_beta = beta
        prev_loglikelihood = loglikelihood
        return beta, prev_beta, prev_loglikelihood, inv_hessian, jump_to

    def local_iter_params(self, X, y, beta, current_iteration):
        """
        Update local parameters from global beta

        Parameters
        ----------
            X : np.array
                X matrix in shape (n_samples, n_features+1).
            y : np.array
                Vector in shape (n_samples,).
            beta : np.array
                Updated beta vector in shape (n_features+1,).
            current_iteration : int
                Current iteration.
        Returns
        -------
            gradient : np.array
                Gradient vector in shape (n_features+1,).
            hessian : np.array
                Hessian matrix in shape (n_features+1, n_features+1).
            loglikelihood : float
                Loglikelihood.
            current_iteration : int
                Updated current iteration.
            jump_to : str
                'global_params' to update global parameters.
        """
        model = gwasprs.regression.LogisticRegression(beta)
        gradient = model.gradient(X, y)
        hessian = model.hessian(X)
        loglikelihood = model.loglikelihood(X, y)
        current_iteration += 1
        jump_to = "global_params"
        return gradient, hessian, loglikelihood, current_iteration, jump_to

    def global_stats(self, beta, inv_hessian):
        """
        Calculates the global statistics

        Parameters
        ----------
            beta : np.array
                Global beta vector in shape (n_features+1,).
            inv_hessian : np.array
                Inverse hessian matrix (n_features+1, n_features+1).
        Returns
        -------
            t_stat : np.array
                T-statistic vector in shape (n_features+1,).
            pval : np.array
                P-value vector in shape (n_features+1,).
            beta : np.array
                Global beta vector in shape (n_features+1,).
        """
        t_stat, pval = gwasprs.stats.logistic_stats(beta, inv_hessian)
        return t_stat, pval, beta


class Standadization(UseCase):
    """
    Standadization under the federated scheme.
    """

    def __init__(self):
        super().__init__()

    @classmethod
    def return_variables(cls, method_name: str):
        if method_name == "local_load_gwasdata":
            return ["A", "sample_info", "snp_info"]
        elif method_name == "local_col_nansum":
            return ["col_sum", "row_count", "jump_to"]
        elif method_name == "local_col_sum":
            return ["col_sum", "row_count"]
        elif method_name == "local_imputed_mean":
            return ["A", "col_sum", "row_count", "jump_to"]
        elif method_name == "global_mean":
            return ["mean", "jump_to"]
        elif method_name == "local_ssq":
            return ["A", "ssq", "row_count"]
        elif method_name == "global_var":
            return ["var", "delete"]
        elif method_name == "local_standardize":
            return ["A"]
        else:
            super().return_variables(method_name)

    def local_load_gwasdata(
        self, filtered_bfile_path, filtered_cov_path, pheno_path, pheno_name, **kwargs
    ):
        """
        Load gwasdata from the given bfile.

        Parameters
        ----------
            filtered_bfile_path : str
                The path to the bfile.
            filtered_cov_path : str
                The path to the covariate file.
            pheno_path : str
                The path to the phenotype file.
            pheno_name : str
                The column name of the phenotype.
            **kwargs : dict, optional
                Additional arguments for the GWASData.custom method.
        Returns
        -------
            A : np.array
                Genotype matrix in shape (n_sample, n_SNP).
        See also
        --------
        GWASData.subset
        GWASData.impute_covariates
        GWASData.add_unique_snp_id
        """
        logging.info("Loading bed file.")
        data = gwasprs.GWASData.read(
            filtered_bfile_path, filtered_cov_path, pheno_path, pheno_name
        )
        data.custom(drop_missing_samples=False, add_unique_snp_id=True, **kwargs)
        A = data.genotype
        sample_info = data.sample_id
        snp_info = data.snp
        return A, sample_info, snp_info

    def local_col_nansum(self, A):
        """
        Calculate local nansum

        Parameters
        ----------
            A : np.array
                Matrix in shape (n_sample, n_feature).
        Returns
        -------
            col_sum : np.array
                Column sum vector in shape (n_feature,).
            row_count : np.array
                The number of samples without missing value for each SNP. # TODO: Check this.
            jump_to : str
                'global_mean' to calculate nanmean.
        """
        col_sum, row_count = gwasprs.stats.nansum(A)
        jump_to = "global_mean"
        return col_sum, row_count, jump_to

    def local_col_sum(self, A):
        """
        Calculate local sum of columns.

        Parameters
        ----------
            A : np.array
                Matrix in shape (n_sample, n_feature).
        Returns
        -------
            col_sum : np.array
                Column sum vector in shape (n_feature,).
            row_count : np.array
                The number of samples.
        """
        return A.sum(axis=0), A.shape[0]

    def local_imputed_mean(self, A, mean):
        """
        Calculate local mean from the imputed matrix

        Parameters
        ----------
            A : np.array
                Matrix in shape (n_sample, n_feature).
            mean : np.array
                Vector in shape (n_feature,).
        Returns
        -------
            A : np.array
                Mean-imputed matrix in shape (n_sample, n_feature).
            col_sum : np.array
                Column sum vector in shape (n_feature,).
            row_count : int
                Number of samples.
            jump_to : str
                'global_mean' to calculate global mean.
        """
        A = jnp.array(A)
        A = gwasprs.stats.impute_with_mean(A, mean)
        col_sum, row_count = gwasprs.stats.sum_and_count(A)
        jump_to = "global_mean"
        return A, col_sum, row_count, jump_to

    def global_mean(self, col_sum, row_count):
        """
        Calculate global mean

        Parameters
        ----------
            col_sum : list of np.array
                Column sum vector in shape (n_feature,) from edges.
            row_count : list of int
                Number of samples.
        Returns
        -------
            mean : np.array
                The global mean vector in shape (n_feature,).
            jump_to : str
                Jumps to 'local_imputed_mean' or 'local_ssq'.
                Depends on the previous step.
        """
        if isinstance(col_sum, (list, tuple)):
            col_sum = gwasprs.aggregations.SumUp()(*col_sum)
            row_count = gwasprs.aggregations.SumUp()(*row_count)
        mean = col_sum / row_count
        if (col_sum.astype(np.int32) == col_sum).all():
            jump_to = "local_imputed_mean"
        else:
            jump_to = "local_ssq"
        return mean, jump_to

    def local_ssq(self, A, mean):
        """
        Calculates the sum of square

        Parameters
        ----------
            A : np.array
                Mean-imputed matrix in shape (n_sample, n_feature).
            mean : np.array
                The global mean vector in shape (n_feature,).
        Returns
        -------
            A : np.array
                Matrix with 0 column means.
            ssq : np.array
                Sum of square vector in shape (n_feature,).
            row_count : int
                The number of all samples.
        """
        A = gwasprs.stats.make_mean_zero(A, mean)
        ssq = gwasprs.stats.sum_of_square(A)
        row_count = A.shape[0]
        return A, ssq, row_count

    def global_var(self, ssq, row_count):
        """
        Calculate global variance

        Parameters
        ----------
            ssq : list of np.array
                Sum of square vector in shape (n_feature,) from edges.
            row_count : int
                The number of all samples in edges.
        Returns
        -------
            var : np.array
                The global variance vector in shape (n_feature,).
            delete : np.array
                The index to delete variants whose variance is 0.
        """
        if isinstance(ssq, (list, tuple)):
            ssq = gwasprs.aggregations.SumUp()(*ssq)
            row_count = gwasprs.aggregations.SumUp()(*row_count)
        var = ssq / (row_count - 1)  # This ddof should be noticed when using
        delete = jnp.where(var == 0)[0]
        var = jnp.delete(var, delete)
        return var, delete

    def local_standardize(self, A, var, delete):
        """
        Standardization with the global variance

        Parameters
        ----------
            A : np.array
                Matrix with 0 column means.
            var : np.array
                The global variance vector in shape (n_feature,).
            delete : np.array
                The index to delete variants whose variance is 0.
        Returns
        -------
            A : np.array
                The standardized matrix in shape (n_sample, n_feature).
        """
        A = gwasprs.stats.standardize(A, var, delete)
        return A


class RandomizedSVD(UseCase):
    """
    Randomized SVD to calculate the global eigenvectors
    under the federated scheme.
    This work is inspired by the work of Anne Hartebrodt,
    see https://github.com/AnneHartebrodt/federated-pca-simulation for code,
    arXiv:2205.12109, Algorithm 2 and 5 for algorithms.
    """

    def __init__(self):
        super().__init__()

    @classmethod
    def return_variables(cls, method_name: str):
        if method_name == "local_init":
            return ["A", "V", "n_features"]
        elif method_name == "global_init":
            return ["prev_U", "current_iteration", "converged", "Us"]
        elif method_name == "update_local_U":
            return ["U"]
        elif method_name == "update_global_U":
            return ["U", "S"]
        elif method_name == "check_convergence":
            return ["converged"]
        elif method_name == "update_global_Us":
            return ["prev_U", "Us", "current_iteration"]
        elif method_name == "update_local_V":
            return ["V", "jump_to"]
        elif method_name == "decompose_global_Us":
            return ["U"]
        elif method_name == "compute_local_covariance":
            return ["P", "PPt"]
        elif method_name == "decompose_global_covariance":
            return ["Vp"]
        elif method_name == "recontruct_local_V":
            return ["V"]
        else:
            super().return_variables(method_name)

    def local_init(self, A, k1):
        """
        Prepare parameters for vertical subspace iteration.
        The dimensions need to be private should be placed at 0 axis.

        Parameters
        ----------
            A : np.array
                Standardized matrix in shape (n_sample, n_feature).
            k1 : int
                The number of latent variables.
        Returns
        -------
            A : np.array
                Transposed matrix in shape (n_feature, n_sample).
            V : np.array
                The matrix is randomly generated in shape (n_sample, k1),
                then orthonormalized by QR decomposition.
            n_features : int
                The number of features.
        """
        A = A.T
        V = gwasprs.linalg.randn(A.shape[1], k1)
        V, _ = jsp.linalg.qr(V, mode="economic")
        n_features = A.shape[0]
        return A, V, n_features

    def global_init(self, n_features, k1):
        """
        Initialize the parameters for vertical subspace iteration.

        Parameters
        ----------
            n_features : int
                The number of features.
            k1 : int
                The number of latent variables.
        Returns
        -------
            prev_U : np.array
                The randomly generated matrix in shape (n_feature, k1).
            current_iteration : int
                Current iteration = 1.
            converged : bool
                Convergence flag.
            Us : list of np.array
                Empty list to store eigenvectors from each iteration.
        """
        prev_U = gwasprs.linalg.init_rand_U(n_features, k1)
        current_iteration = 1
        converged = False
        Us = []
        return prev_U, current_iteration, converged, Us

    def update_local_U(self, A, V):
        """
        Update the sharable eigenvectors.

        Parameters
        ----------
            A : np.array
                Standardized matrix in shape (n_feature, n_sample).
            V : np.array
                The private eigenvectors in shape (n_sample, k1).
        Returns
        -------
            U : np.array
                Updated eigenvectors in shape (n_feature, k1).
        """
        U = gwasprs.linalg.update_local_U(A, V)
        return U

    def update_global_U(self, U):
        """
        Update the sharable eigenvectors.

        Parameters
        ----------
            U : np.array
                Eigenvectors in shape (n_feature, k1) from edges.
        Returns
        -------
            U : np.array
                Global orthonormalized eigenvectors in shape (n_feature, k1).
            S : np.array
                The singular values in shape (k1,).
        """
        if isinstance(U, (list, tuple)):
            U = gwasprs.linalg.aggregations.SumUp()(*U)
        U, S = gwasprs.linalg.orthonormalize(U)
        return U, S

    def check_convergence(self, U, prev_U, epsilon):
        """
        Check the convergence of the eigenvectors.

        Parameters
        ----------
            U : np.array
                Current eigenvectors in shape (n_feature, k1).
            prev_U : np.array
                Previous eigenvectors in shape (n_feature, k1).
            epsilon : float
                The tolerance of the convergence.
        Returns
        -------
            converged : bool
                Convergence flag.
        """
        converged, _ = gwasprs.linalg.check_eigenvector_convergence(U, prev_U, epsilon)
        return converged

    def update_global_Us(self, U, Us, current_iteration):
        """
        Store the current eigenvectors.
        Current iteration + 1.

        Parameters
        ----------
            U : np.array
                Current eigenvectors in shape (n_feature, k1).
            Us : list of np.array
                The list to store eigenvectors from each iteration.
            current_iteration : int
                Current iteration.
        Returns
        -------
            prev_U : np.array
                Current as previous eigenvectors in shape (n_feature, k1).
            Us : list of np.array
                Updated list of eigenvectors.
            current_iteration : int
                Current iteration + 1.
        """
        prev_U, Us, current_iteration = gwasprs.linalg.update_Us(
            U, Us, current_iteration
        )
        return prev_U, Us, current_iteration

    def update_local_V(self, A, U, converged, current_iteration, max_iterations):
        """
        Update the private eigenvectors.

        Parameters
        ----------
            A : np.array
                Standardized matrix in shape (n_feature, n_sample).
            U : np.array
                Updated global sharable eigenvectors in shape (n_feature, k1).
            converged : bool
                Convergence flag.
            current_iteration : int
                Used for determining termination.
        Returns
        -------
            V : np.array
                Updated private eigenvectors in shape (n_sample, k1).
            jump_to : str
                If not meeting the termination criteria,
                jump to the next iteration, 'update_local_U'.
                Otherwise, jump to 'decompose_global_Us'.
        """
        V = gwasprs.linalg.update_local_V(A, U)
        if not converged and current_iteration < max_iterations:
            jump_to = "update_local_U"
        else:
            jump_to = "next"
        return V, jump_to

    def decompose_global_Us(self, Us):
        """
        Decompose the stacked eigenvectors from each iteration.

        Parameters
        ----------
            Us : list of np.array
                The list of eigenvectors from each iteration.
        Returns
        -------
            U : np.array
                Decomposed eigenvectors in shape (n_features, k1*I),
                I is the number of iterations.
        """
        U = gwasprs.linalg.decompose_U_stack(Us)
        return U

    def compute_local_covariance(self, A, U):
        """
        Calculate proxy matrix P and the corresponding covariance matrix.

        Parameters
        ----------
            A : np.array
                Standardized matrix in shape (n_feature, n_sample).
            U : np.array
                Decomposed eigenvectors in shape (n_features, k1*I),
                I is the number of iterations.
        Returns
        -------
            P : np.array
                Proxy data matrix in shape (k1*I, n_sample).
            PPt : np.array
                The proxy covariance matrix in shape (k1*I, k1*I).
        """
        P = gwasprs.linalg.create_proxy_matrix(A, U)
        PPt = gwasprs.linalg.covariance_from_proxy_matrix(P)
        return P, PPt

    def decompose_global_covariance(self, PPt, k2):
        """
        Decompose the proxy covariance matrix.

        Parameters
        ----------
            PPt : np.array
                The proxy covariance matrix in shape (k1*I, k1*I).
            k2 : int
                The specified number of latent variables.
        Returns
        -------
            Vp : np.array
                Eigenvectors in shape (k1*I, k2).
        """
        if isinstance(PPt, (list, tuple)):
            PPt = gwasprs.aggregations.SumUp()(*PPt)
        Vp = gwasprs.linalg.svd(PPt)[0][:, :k2]
        return Vp

    def recontruct_local_V(self, P, Vp):
        """
        Recontruct the private eigenvectors.

        Parameters
        ----------
            P : np.array
                Proxy data matrix in shape (k1*I, n_sample).
            Vp : np.array
                Global eigenvectors in shape (k1*I, k2).
        Returns
        -------
            V : np.array
                Final private eigenvectors in shape (n_sample, k2).
        """
        V = gwasprs.linalg.local_V_from_proxy_matrix(P, Vp)
        return V


class GramSchmidt(UseCase):
    """
    Gram-Schmidt orthonormalization under the federated scheme.
    This work is inspired by the work of Anne Hartebrodt,
    see https://github.com/AnneHartebrodt/federated-pca-simulation for code,
    arXiv:2205.12109, Algorithm 3 for algorithm.
    """

    def __init__(self):
        super().__init__()

    @classmethod
    def return_variables(cls, method_name: str):
        if method_name == "local_make_V_as_M":
            return ["M"]
        elif method_name == "local_first_norm":
            return ["partial_norm", "orthogonalized"]
        elif method_name == "global_first_norm":
            return ["global_norms", "eigen_idx"]
        elif method_name == "local_residuals":
            return ["residuals"]
        elif method_name == "global_residuals":
            return ["residuals"]
        elif method_name == "local_nth_norm":
            return ["partial_norm", "orthogonalized"]
        elif method_name == "global_nth_norm":
            return ["global_norms", "eigen_idx", "jump_to"]
        elif method_name == "local_normalization":
            return ["M"]
        elif method_name == "local_make_M_as_V":
            return ["V"]
        else:
            super().return_variables(method_name)

    def local_make_V_as_M(self, V):
        M = V
        return M

    def local_first_norm(self, M):
        """
        Calculate the first norm.

        Parameters
        ----------
            M : np.array
                Matrix in shape (n_sample, k),
                where k is the numnber of latent variables.
        Returns
        -------
            partial_norm : float
                The partial norm of first eigenvector.
            orthogonalized : list of np.array
                The list stores partial eigenvectors.
        """
        partial_norm, orthogonalized = gwasprs.linalg.init_gram_schmidt(M)
        return partial_norm, orthogonalized

    def global_first_norm(self, partial_norm):
        """
        Aggregate the first norms from edges.

        Parameters
        ----------
            partial_norm : float
                The partial norm of first eigenvector.
        Returns
        -------
            global_norms : list of float
                The list stores the first global norm.
            eigen_idx : int
                To indicate which eigenvector being orthonormalized.
        """
        if isinstance(partial_norm, (list, tuple)):
            global_norms = [gwasprs.aggregations.SumUp()(*partial_norm)]
        else:
            global_norms = [partial_norm]
        eigen_idx = 1
        return global_norms, eigen_idx

    def local_residuals(self, M, orthogonalized, eigen_idx, global_norms):
        """
        Calculate the local residuals.

        Parameters
        ----------
            M : np.array
                Matrix in shape (n_sample, k),
                where k is the numnber of latent variables.
            orthogonalized : list of np.array
                The list stores partial eigenvectors.
            eigen_idx : int
                To indicate which eigenvector being orthonormalized.
            global_norms : list of float
                The list stores the global norms.
        Returns
        -------
            residuals : list of float
                i-1 residuals used for orthogonalized ith eigenvector.
        """
        residuals = gwasprs.project.compute_residuals(
            M, orthogonalized, eigen_idx, global_norms
        )
        return residuals

    def global_residuals(self, residuals):
        """
        Aggregate the residuals from edges.

        Parameters
        ----------
            residuals : list of float
                i-1 residuals used for orthogonalized ith eigenvector.
        Returns
        -------
            residuals : np.array
                Global residuals in shape (i-1,).
        """
        if isinstance(residuals, (list, tuple)):
            residuals = gwasprs.aggregations.SumUp()(*residuals)
        return residuals

    def local_nth_norm(self, M, orthogonalized, eigen_idx, residuals):
        """
        Othogonalize eigen_idx+1th eigenvector,
        and calculate its corresponding norm.

        Parameters
        ----------
            M : np.array
                Matrix in shape (n_sample, k),
                where k is the numnber of latent variables.
            orthogonalized : list of np.array
                The list stores partial eigenvectors.
            eigen_idx : int
                To indicate which eigenvector being orthonormalized.
            residuals : np.array
                Global residuals in shape (i-1,),
                which also equals to (eigen_idx,).
        Returns
        -------
            partial_norm : float
                The partial norm of eigen_idx+1th eigenvector.
            orthogonalized : list of np.array
                The list stores partial eigenvectors.
        """
        ortho_v = gwasprs.linalg.orthogonalize(
            M[:, eigen_idx], orthogonalized, residuals
        )
        gwasprs.project.update_ortho_vectors(ortho_v, orthogonalized)
        partial_norm = gwasprs.project.compute_norm(ortho_v)
        return partial_norm, orthogonalized

    def global_nth_norm(self, global_norms, partial_norm, eigen_idx, k2):
        """
        Aggregate the eigen_idx+1th norms from edges.

        Parameters
        ----------
            global_norms : list of float
                The list stores the global norms.
            partial_norm : float
                The partial norm of eigen_idx+1th eigenvector.
            eigen_idx : int
                To indicate which eigenvector being orthonormalized.
            k2 : int
                The number of output latent variables.
        Returns
        -------
            global_norms : list of float
                The updated list stores the global norms.
            eigen_idx : int
                To indicate which eigenvector being orthonormalized.
            jump_to : str
                If approaching the last eigenvector (k2th),
                jump to the normalization step.
                Otherwise, jump to the local_residuals step.
        """
        if isinstance(partial_norm, (list, tuple)):
            norm = gwasprs.aggregations.SumUp()(*partial_norm)
        else:
            norm = partial_norm
        global_norms.append(norm)
        if eigen_idx < k2 - 1:
            jump_to = "local_residuals"
        else:
            jump_to = "next"
        eigen_idx += 1
        return global_norms, eigen_idx, jump_to

    def local_normalization(self, global_norms, orthogonalized):
        """
        Make the lengths of eigenvectors equal to 1.

        Parameters
        ----------
            global_norms : list of float
                The list stores the global norms.
            orthogonalized : list of np.array
                The list stores partial eigenvectors.
        Returns
        -------
            M : np.array
                Normalized eigenvectors in shape (n_sample, k2),
        """
        M = gwasprs.stats.normalize(global_norms, orthogonalized)
        return M

    def local_make_M_as_V(self, M):
        V = M
        return V


class CoxPHRegression(UseCase):
    def __init__(self):
        super().__init__()

    @classmethod
    def return_variables(cls, method_name: str):
        if method_name == "global_create_Xanc":
            return ["Xanc"]
        elif method_name == "local_load_metadata":
            return ["X", "y", "keep_feature_cols"]
        elif method_name == "local_create_proxy_data":
            return ["F", "X_tilde", "Xanc_tilde", "y", "feature_sum"]
        elif method_name == "global_fit_model":
            return ["coef", "coef_var", "baseline_hazard", "feature_mean"]
        elif method_name == "local_recover_survival":
            return ["survival_func"]
        else:
            super().return_variables(method_name)

    def global_create_Xanc(self, n_features, r=100):
        """
        Generate a random anchor matrix Xanc.

        Parameters
        ----------
            n_features : int
                The number of features to be perform Cox-PH regression.
            r : int
                The pseudo-number of samples of Xanc.

        Returns
        -------
            Xanc : np.array
                The anchor matrix with shape (r, n_features).
        """
        return np.random.randn(r, n_features)

    def local_load_metadata(
        self,
        clinical_data_path: str,
        keep_feature_cols: list = None,
        meta_cols: list = None,
    ):
        """
        Load clinical metadata.
        The sample will be dropped if it has any missing values.

        Parameters
        ----------
            clinical_data_path : str
                The path to clinical metadata.
            keep_feature_cols : list of strings
                The features to be perform Cox-PH regression.
            meta_cols : list of str
                The columns recording the sample information.

        Returns
        -------
            X : np.array
                The feature matrix with shape (n_samples, n_features).
            y : np.array
                The concatenated time and event vectors with shape (n_samples, 2).
            keep_feature_cols : list of str
                The features to be perform Cox-PH regression.
            meta: pd.DataFrame
                The sample metadata.
        """
        metadata = pd.read_csv(clinical_data_path)

        if "Unnamed: 0" in metadata.columns:
            metadata.set_index("Unnamed: 0", inplace=True)
            metadata.index.name = None

        print(metadata)

        # Deal with the target columns
        expected_cols = ["time", "event"]
        missed_cols = []
        for col in expected_cols:
            if col not in metadata.columns:
                missed_cols.append(col)
        assert len(missed_cols) == 0, f"Missing columns {missed_cols}."

        meta_cols = [] if meta_cols is None else meta_cols

        # Deal with the feature columns
        if keep_feature_cols is None:
            keep_feature_cols = list(
                filter(lambda x: x not in expected_cols + meta_cols, metadata.columns)
            )
        else:
            unexpected_features = set(keep_feature_cols).difference(metadata.columns)
            if len(unexpected_features) > 0:
                logging.warn(
                    f"The feature columns {unexpected_features} are not in the clinical data. "
                    f"They are removed automatically."
                )
            keep_feature_cols = list(
                filter(lambda x: x in metadata.columns, keep_feature_cols)
            )

        # Deal with the sample metadata columns
        meta = metadata.loc[:, meta_cols]

        # Remove samples with the missing values
        metadata = metadata.loc[:, expected_cols + keep_feature_cols]
        metadata = metadata.dropna()
        if len(metadata) == 0:
            logging.warn("There are no samples left.")

        print(
            f"\033[95m The feature matrix X:\n \033[0m"
            f"{metadata.loc[:, keep_feature_cols]}\n"
            f"\n"
            f"\033[95m The target matrix y:\n \033[0m"
            f"{metadata.loc[:, expected_cols]}\n"
            f"\n"
            f"\033[95m The metadata:\n \033[0m"
            f"{meta}\n"
        )

        X = metadata.loc[:, keep_feature_cols].to_numpy()
        y = metadata.loc[:, expected_cols].to_numpy()
        return X, y, keep_feature_cols, meta

    def local_create_proxy_data(
        self, X, Xanc, y, k=20, bs_prop=0.6, bs_times=20, alpha=0.05, step_size=0.5
    ):
        """
        Create linear projection matrix F,
        projected matrix X_tilde and projected anchor matrix Xanc_tilde.

        Parameters
        ----------
            X : np.array
                The feature matrix with shape (n_samples, n_features).
            Xanc : np.array
                The global anchor matrix with shape (r, n_features).
                where r is the pseudo-number of samples of Xanc.
            y : np.array
                The concatenated time and event vectors with shape (n_samples, 2).
            k : int
                The latent dimension of Fdr. Notice that the ultimate dimension is min(k, len(S)),
                where the S is the number of nonzero singular values.
            bs_prop : float
                The proportion of the samples in a client used for bootstrapping for each time.
            bs_times : int
                The number of times to bootstrap.
            alpha : float
                The level in the confidence intervals.
                It is `alpha` parameter in `lifelines.fitters.coxph_fitter.CoxPHFitter`.
            step_size : float
                Deal with the fitting error, `delta contains nan value(s)`.
                See also: https://lifelines.readthedocs.io/en/latest/Examples.html#problems-with-convergence-in-the-cox-proportional-hazard-model

        Returns
        -------
            F : np.array
                The linear projection matrix to be used for creating the projected matrices,
                X_tilde and Xanc_tilde, its shape is (n_features, m tilde).
            X_tilde : list of np.array
                The projected feature matrix with shape (n_samples, m tilde).
            Xanc_tilde : list of np.array
                The projected global anchor matrix with shape (r, m tilde).
            feature_sum : np.array
                The sums of the features.
        """
        if len(X) == 0:
            return None, [None], [None], np.zeros(X.shape[1])

        projector = survival.DCCoxDataProjector(k, bs_prop, bs_times, alpha, step_size)
        projector.project(X=X, Xanc=Xanc, durations=y[:, 0], events=y[:, 1])

        F = projector.F
        X_tilde = projector.X_tilde
        Xanc_tilde = projector.Xanc_tilde

        feature_sum = np.sum(X, axis=0)

        return F, [X_tilde], [Xanc_tilde], feature_sum

    @staticmethod
    def _global_compute_coef_tilde(coef, coef_var, Gs: survival.BlockMatrix):
        coef_ = list(
            map(
                lambda c: list(map(lambda d: Gs[c, d] @ coef, range(Gs.shape[1]))),
                range(Gs.shape[0]),
            )
        )
        coef_var_ = list(
            map(
                lambda c: list(
                    map(lambda d: Gs[c, d] @ coef_var @ Gs[c, d].T, range(Gs.shape[1]))
                ),
                range(Gs.shape[0]),
            )
        )
        return coef_, coef_var_

    def global_fit_model(
        self, Xs_tilde, Xancs_tilde, ys, sums, alpha=0.05, step_size=0.5
    ):
        """
        Perform the Cox-PH regression on the given Xs_tilde and Xancs_tilde.

        Parameters
        ----------
            Xs_tilde : list of lists of np.array
                The data structure is:
                [
                    [X_tilde],
                    [X_tilde],...
                ]
                The shape for each projected feature matrix is (nc, m tilde).
            Xancs_tilde : list of lists of np.array
                The data structure is:
                [
                    [Xanc_tilde],
                    [Xanc_tilde],...
                ]
                The shape for each projected global anchor matrix is (r, m tilde).
            ys : list of np.array
                The concatenated time and event vectors with shape (nc, 2).
            sums : list of np.array
                The sums of the features.
            alpha : float
                The level in the confidence intervals.
                It is `alpha` parameter in `lifelines.fitters.coxph_fitter.CoxPHFitter`.
            step_size : float
                Deal with the fitting error, `delta contains nan value(s)`.
                See also: https://lifelines.readthedocs.io/en/latest/Examples.html#problems-with-convergence-in-the-cox-proportional-hazard-model

        Returns
        -------
            coef : list of list of np.array
                The beta tilde vector with shape (m tilde,).
            coef_var : list of list of np.array
                The variance-covariance matrix with shape (m tilde, m tilde).
            baseline_hazard : pd.DataFrame
                The baseline hazard values, where the index is the time.
            feature_mean : np.array
                The global means of the features.
        """
        assert isinstance(Xs_tilde[0], list), "Xs_tilde must be a list of lists."
        assert isinstance(Xancs_tilde[0], list), "Xancs_tilde must be a list of lists."
        assert isinstance(ys, list), "y must be a list of 2d arrays."
        assert isinstance(sums, list), "sums must be a list of 1d arrays."
        keep_idx = [i for i, Xc in enumerate(Xs_tilde) if Xc[0] is not None]

        Xs_tilde = survival.BlockMatrix(list(map(lambda idx: Xs_tilde[idx], keep_idx)))
        Xancs_tilde = survival.BlockMatrix(list(map(lambda idx: Xancs_tilde[idx], keep_idx)))
        ys = np.concatenate(list(map(lambda idx: ys[idx], keep_idx)))
        feature_mean = np.sum(sums, axis=0) / ys.shape[0]

        model = survival.DCCoxRegressor(alpha, step_size).fit(
            Xs_tilde=Xs_tilde,
            Xancs_tilde=Xancs_tilde,
            events=ys[:, 1],
            durations=ys[:, 0],
        )

        coef, coef_var = self._global_compute_coef_tilde(
            model.coef, model.coef_var, model.Gs
        )
        baseline_hazard = model.baseline_hazard

        return coef, coef_var, baseline_hazard, feature_mean

    def local_recover_survival(
        self, keep_feature_cols, coef, coef_var, baseline_hazard, mean, F, alpha=0.05
    ):
        """
        Recover the survival function and statistical properties
        from projected coefficients and variance-covariance matrix.

        Parameters
        ----------
            keep_feature_cols : list of strings
                The features to be perform Cox-PH regression.
            coef : np.array
                The beta tilde vector with shape (m tilde,).
            coef_var : np.array
                The variance-covariance matrix with shape (m tilde, m tilde).
            baseline_hazard : pd.DataFrame
                The baseline hazard values, where the index is the time.
            mean : np.array
                The global means of the features.
            F : np.array
                The linear projection matrix to be used for creating the projected matrices,
                X_tilde and Xanc_tilde, its shape is (n_features, m tilde).

        Returns
        -------
            survival_func : SurvivalFunction
                The object recovers the survival function and statistical properties
                from the given coefficients and variance-covariance matrix.
        """
        coef = F @ coef
        coef_var = F @ coef_var @ F.T
        coef = pd.Series(coef, index=keep_feature_cols)
        coef.name = "covariate"
        survival_func = survival.CoxSurvivalFunction(coef, coef_var, baseline_hazard, mean, alpha)
        return survival_func


class KaplanMeier(UseCase):
    def __init__(self):
        super().__init__()

    def return_variables(self, method_name: str):
        if method_name == "local_load_metadata":
            return ["X", "y", "keep_feature_cols"]
        elif method_name == "local_group_by_std":
            return ["grouped_y", "n_std"]
        elif method_name == "global_fit_model":
            return ["fitted", "logrank_stats"]
        else:
            super().return_variables(method_name)

    def local_load_metadata(
        self,
        clinical_data_path: str,
        keep_feature_cols: list = None,
        meta_cols: list = None,
    ):
        """
        Load clinical metadata.
        The sample will be dropped if it has any missing values.

        Parameters
        ----------
            clinical_data_path : str
                The path to clinical metadata.
            keep_feature_cols : list of strings
                The features to be perform Cox-PH regression.
            meta_cols : list of str
                The columns recording the sample information.

        Returns
        -------
            X : np.array
                The feature matrix with shape (n_samples, n_features).
            y : np.array
                The concatenated time and event vectors with shape (n_samples, 2).
            keep_feature_cols : list of str
                The features to be perform Cox-PH regression.
            meta: pd.DataFrame
                The sample metadata.
        """
        metadata = pd.read_csv(clinical_data_path)

        if "Unnamed: 0" in metadata.columns:
            metadata.set_index("Unnamed: 0", inplace=True)
            metadata.index.name = None

        print(metadata)

        # Deal with the target columns
        expected_cols = ["time", "event"]
        missed_cols = []
        for col in expected_cols:
            if col not in metadata.columns:
                missed_cols.append(col)
        assert len(missed_cols) == 0, f"Missing columns {missed_cols}."

        meta_cols = [] if meta_cols is None else meta_cols

        # Deal with the feature columns
        if keep_feature_cols is None:
            keep_feature_cols = list(
                filter(lambda x: x not in expected_cols + meta_cols, metadata.columns)
            )
        else:
            unexpected_features = set(keep_feature_cols).difference(metadata.columns)
            if len(unexpected_features) > 0:
                logging.warn(
                    f"The feature columns {unexpected_features} are not in the clinical data. "
                    f"They are removed automatically."
                )
            keep_feature_cols = list(
                filter(lambda x: x in metadata.columns, keep_feature_cols)
            )

        # Deal with the sample metadata columns
        meta = metadata.loc[:, meta_cols]

        # Remove samples with the missing values
        metadata = metadata.loc[:, expected_cols + keep_feature_cols]
        metadata = metadata.dropna()
        if len(metadata) == 0:
            logging.warn("There are no samples left.")

        print(
            f"\033[95m The feature matrix X:\n \033[0m"
            f"{metadata.loc[:, keep_feature_cols]}\n"
            f"\n"
            f"\033[95m The target matrix y:\n \033[0m"
            f"{metadata.loc[:, expected_cols]}\n"
            f"\n"
            f"\033[95m The metadata:\n \033[0m"
            f"{meta}\n"
        )

        X = metadata.loc[:, keep_feature_cols].to_numpy()
        y = metadata.loc[:, expected_cols].to_numpy()
        return X, y, keep_feature_cols, meta

    ################################################################
    #                                                              #
    #               Use the Standardization Use Case               #
    #                   Get the standardized X                     #
    #                                                              #
    ################################################################

    def local_group_by_std(self, std_X, n_std, y):
        """
        Group the data by the standard deviation.

        Parameters
        ----------
            std_X : np.array
                The standardized feature matrix with shape (n_samples, n_features).
            n_std : int or float or list of int or float
                The weight of standard deviation.
            y : np.array
                The concatenated time and event vectors with shape (n_samples, 2).

        Returns
        -------
            grouped_y : list of list of np.array
                The grouped data with shape (n_features, upper+lower, time+event).
            n_std : list of int or float
                The weight of standard deviation for each feature.
        """
        if isinstance(n_std, (int, float)):
            n_std = [n_std] * std_X.shape[1]

        grouped_y = []
        for i in range(std_X.shape[1]):
            # The std. = 1, so use the n_std directly
            upper_mask = std_X[:, i] >= n_std[i]
            lower_mask = std_X[:, i] <= -n_std[i]
            grouped_y.append([y[upper_mask], y[lower_mask]])

        return grouped_y, n_std

    @staticmethod
    def _fit_kmf(durations, event_observed, alpha=0.05):
        kmf = KaplanMeierFitter(alpha=alpha)
        kmf.fit(durations, event_observed)

        surv_func = kmf.survival_function_.to_numpy().reshape(-1)
        surv_ci = kmf.confidence_interval_survival_function_.to_numpy()
        timeline = kmf.timeline
        median_surv_time = kmf.median_survival_time_

        return {
            "surv_func": surv_func,
            "surv_ci": surv_ci,
            "timeline": timeline,
            "median_surv_time": median_surv_time,
            "n_samples": durations.shape[0],
        }

    def global_fit_model(self, grouped_y, alpha=0.05):
        """
        Fit Kaplan-Meier model for each feature.

        Parameters
        ----------
            grouped_y : list of list of list of np.array
                The grouped data with shape (n_features, upper+lower, time+event).
            alpha : float
                The level in the confidence intervals.

        Returns
        -------
            fitted : list of list of dict
                The fitted Kaplan-Meier model for each feature.
                [[fitted_upper, fitted_lower], [fitted_upper, fitted_lower],...]
            logrank_stats : list of list of float
                The log-rank test statistic and p-value for each feature.
        """
        fitted, logrank_stats = [], []
        for feature_y in zip(*grouped_y):
            upper_y = np.concatenate(
                list(
                    map(
                        lambda x: x[0].reshape(-1, 2) if x[0].ndim == 1 else x[0],
                        feature_y,
                    )
                )
            )
            lower_y = np.concatenate(
                list(
                    map(
                        lambda x: x[1].reshape(-1, 2) if x[1].ndim == 1 else x[1],
                        feature_y,
                    )
                )
            )

            # Fit Kaplan-Meier model
            fitted.append(
                [
                    self._fit_kmf(
                        durations=upper_y[:, 0],
                        event_observed=upper_y[:, 1],
                        alpha=alpha,
                    )
                    if len(upper_y) != 0
                    else None,
                    self._fit_kmf(
                        durations=lower_y[:, 0],
                        event_observed=lower_y[:, 1],
                        alpha=alpha,
                    )
                    if len(lower_y) != 0
                    else None,
                ]
            )

            # Log-rank test
            if len(upper_y) == 0 or len(lower_y) == 0:
                logrank_stats.append([np.nan, np.nan])
            else:
                res = logrank_test(
                    durations_A=upper_y[:, 0],
                    durations_B=lower_y[:, 0],
                    event_observed_A=upper_y[:, 1],
                    event_observed_B=lower_y[:, 1],
                ).summary
                logrank_stats.append([res["test_statistic"][0], res["p"][0]])

        return fitted, logrank_stats


class TabularReader(UseCase):
    def __init__(self):
        super().__init__()

    @classmethod
    def return_variables(cls, method_name: str):
        if method_name == "local_read_csv":
            return ["A"]
        else:
            super().return_variables(method_name)

    def local_read_csv(self, path, meta_cols=None, drop_cols=None, keep_cols=None):
        assert (
            drop_cols is None or keep_cols is None
        ), "only drop_cols or keep_cols, not both at the same time."

        table = pd.read_csv(path, low_memory=False)

        if "Unnamed: 0" in table.columns:
            table.set_index("Unnamed: 0", inplace=True)
            table.index.name = None

        if drop_cols is not None:
            table = table.drop(columns=drop_cols)

        elif keep_cols is not None:
            table = table.loc[:, keep_cols]

        if meta_cols is not None:
            meta = table.loc[:, meta_cols]
            table = table.drop(columns=meta_cols)
        else:
            meta = None

        return meta, table


class Output(UseCase):
    def __init__(self):
        super().__init__()

    @classmethod
    def return_variables(cls, method_name: str):
        if method_name == "regression_results":
            return [""]
        elif method_name == "svd_results":
            return [""]
        elif method_name == "svd_figures":
            return [""]
        elif method_name == "gwas_plot_statistics":
            return [""]
        elif method_name == "cox_regression_results":
            return [""]
        elif method_name == "kaplan_meier_results":
            return [""]
        else:
            super().return_variables(method_name)

    def regression_results(
        self, snp_info, t_stat, pval, beta, n_obs, regression_save_dir
    ):
        snp_info["P"] = pval
        snp_info["T_STAT"] = t_stat
        snp_info["BETA"] = beta
        snp_info["CNT"] = n_obs
        col_list = ["CHR", "ID", "POS", "A1", "A2", "CNT", "BETA", "T_STAT", "P"]
        if "rsID" in snp_info:
            col_list = ["rsID"] + col_list
        snp_info = snp_info.loc[:, col_list]

        if not os.path.exists(regression_save_dir):
            os.makedirs(regression_save_dir)

        glm = os.path.join(regression_save_dir, "gwas.glm")
        if os.path.exists(glm):
            snp_info.to_csv(glm, index=False, mode="a", header=False, sep="\t")
        else:
            snp_info.to_csv(glm, index=False, sep="\t")
        return

    def svd_results(
        self, row_metadata, col_metadata, U, S, V, svd_save_dir, to_pc=False
    ):
        if not os.path.exists(svd_save_dir):
            os.makedirs(svd_save_dir)

        row_index = range(V.shape[0]) if row_metadata is None else row_metadata.index
        col_index = range(U.shape[0]) if col_metadata is None else col_metadata.index

        if to_pc:
            col_PCs = pd.DataFrame(
                U @ np.diag(S),
                index=col_index,
                columns=[f"PC{i+1}" for i in range(U.shape[1])],
            )
            col_PCs = pd.concat([col_metadata, col_PCs], axis=1)

            row_PCs = pd.DataFrame(
                V @ np.diag(S),
                index=row_index,
                columns=[f"PC{i+1}" for i in range(V.shape[1])],
            )
            row_PCs = pd.concat([row_metadata, row_PCs], axis=1)

            col_pc_file = os.path.join(svd_save_dir, "col.pc.csv")
            row_pc_file = os.path.join(svd_save_dir, "row.pc.csv")

            col_PCs.to_csv(col_pc_file, index=None)
            row_PCs.to_csv(row_pc_file, index=None)

        else:
            U = pd.DataFrame(
                U,
                index=col_index,
                columns=[f"Eigenvec{i+1}" for i in range(U.shape[1])],
            )
            U = pd.concat([col_metadata, U], axis=1)

            S = pd.DataFrame(S).T
            S.columns = [f"Sinval{i+1}" for i in range(S.shape[1])]

            V = pd.DataFrame(
                V,
                index=row_index,
                columns=[f"Eigenvec{i+1}" for i in range(V.shape[1])],
            )
            V = pd.concat([row_metadata, V], axis=1)

            u_file = os.path.join(svd_save_dir, "col.eigenvec.csv")
            s_file = os.path.join(svd_save_dir, "s.sinval.csv")
            v_file = os.path.join(svd_save_dir, "row.eigenvec.csv")

            U.to_csv(u_file, index=None)
            S.to_csv(s_file, index=None)
            V.to_csv(v_file, index=None)
        return

    def svd_figures(self, fig_df, label, svd_save_dir):
        if not os.path.exists(svd_save_dir):
            os.makedirs(svd_save_dir)

        label = [None] if label is None else label
        vec_names = [
            col
            for col in fig_df.columns
            if col.startswith("PC") or col.startswith("Eigenvec")
        ]
        
        fig_df.loc[:, vec_names].to_csv(
            os.path.join(
                svd_save_dir,
                f"rafael.{vec_names[0][:-1]}.csv",
            ),
            index=None
        )

        for label_ in label:
            for i in range(len(vec_names) - 1):
                plt.figure(figsize=(7, 6), dpi=200)
                ax = sns.scatterplot(
                    fig_df,
                    x=vec_names[i],
                    y=vec_names[i + 1],
                    hue=label_,
                    s=8,
                    alpha=0.5,
                )

                if label_ is not None:
                    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

                plt.tight_layout()

                plt.savefig(
                    os.path.join(
                        svd_save_dir,
                        f"rafael.{vec_names[i]}.{vec_names[i+1]}.{label_}.png",
                    )
                )
                plt.close()
        return

    def gwas_plot_statistics(self, regression_save_dir):
        glm = gwasplot.read_glm(
            os.path.join(regression_save_dir, "gwas.glm"),
            "CHR",
            "POS",
            "P",
            "\t",
        )
        glm, max_logp = gwasplot.format_glm(glm)
        prefix = os.path.join(regression_save_dir, "gwas")
        
        # Manhattan
        manhattan_glm = gwasplot.prepare_manhattan(glm)
        gwasplot.plot_manhattan(
            *manhattan_glm,
            f"{prefix}.manhattan.png",
            1e-5,
            5e-8,
            max_logp
        )

        # QQ
        qq_glm = gwasplot.prepare_qq(glm)
        gwasplot.plot_qq(qq_glm, f"{prefix}.qq.png")
        return

    def cox_regression_results(self, stats_summary, cox_save_dir):
        if not os.path.exists(cox_save_dir):
            os.makedirs(cox_save_dir)

        stats_summary = stats_summary.sort_values("coef")
        stats_summary.to_csv(os.path.join(cox_save_dir, "rafael.cox.stats.summary.csv"))

        plt.figure(figsize=(8, 6))

        # Scatter plot for mean coefficients
        plt.scatter(
            y=stats_summary.index,
            x=stats_summary["coef"],
            color="steelblue",
            label="Mean",
            zorder=5,
        )

        # Plot upper and lower CIs
        plt.errorbar(
            y=stats_summary.index,
            x=stats_summary["coef"],
            xerr=[
                stats_summary["coef"] - stats_summary["coef lower 95.0% CI"],
                stats_summary["coef upper 95.0% CI"] - stats_summary["coef"],
            ],
            fmt="none",
            ecolor="black",
            elinewidth=1,
            capsize=2,
            label="95% CI",
        )

        # Set y-axis ticks and labels
        plt.yticks(stats_summary.index, stats_summary.index)

        # Set labels and title
        plt.xlabel("log(HR) (95% CI)")
        plt.ylabel("Features")
        plt.title("Cox Proportional Hazard Regression Coefficients")

        # Add gridlines
        plt.grid(axis="x", linestyle="--", alpha=0.7)

        # Add legend and adjust layout
        plt.legend()
        plt.tight_layout()

        # Save figure
        plt.savefig(os.path.join(cox_save_dir, "rafael.cox.coef.png"), dpi=200)

        plt.close()
        return

    def kaplan_meier_results(
        self, keep_feature_cols, fitted, logrank_stats, n_std, km_save_dir
    ):
        if not os.path.exists(km_save_dir):
            os.makedirs(km_save_dir)

        def _plot_surv(kmf, legend, ax, color):
            if kmf is None:
                return

            timeline = kmf["timeline"]
            surv = pd.DataFrame({legend: kmf["surv_func"]}, index=timeline)
            ci = np.array(kmf["surv_ci"])

            ax.plot(
                timeline,
                surv[legend],
                drawstyle="steps-post",
                color=color,
                label=legend,
            )
            ax.fill_between(
                timeline, ci[:, 0], ci[:, 1], alpha=0.3, step="post", color=color
            )

            ax.legend()
            ax.set_xlabel("Time")
            ax.set_ylabel("Survival Probability")

        for i in range(len(keep_feature_cols)):
            feature = keep_feature_cols[i]

            upper_kmf = fitted[i][0]
            lower_kmf = fitted[i][1]

            _, ax = plt.subplots(1, 1, figsize=(8, 6))
            colors = Accent.colors[:2]
            _plot_surv(upper_kmf, f"$\geq +{n_std[i]}\sigma$", ax, colors[1])
            _plot_surv(lower_kmf, f"$\leq -{n_std[i]}\sigma$", ax, colors[0])

            stats, pval = logrank_stats[i]
            surv_median = (
                None if upper_kmf is None else upper_kmf["median_surv_time"],
                None if lower_kmf is None else lower_kmf["median_surv_time"],
            )
            n_samples = (
                0 if upper_kmf is None else upper_kmf["n_samples"],
                0 if lower_kmf is None else lower_kmf["n_samples"],
            )

            ax.text(
                1.05,
                0.5,
                f"Number of observations\n"
                f"  $\geq +{n_std[i]}\sigma$={n_samples[0]}\n"
                f"  $\leq -{n_std[i]}\sigma$={n_samples[1]}\n"
                f"\n"
                f"Median Survival time:\n"
                f"  $\geq +{n_std[i]}\sigma$={surv_median[0]}\n"
                f"  $\leq -{n_std[i]}\sigma$={surv_median[1]}\n"
                f"\n"
                f"Log-rank test:\n"
                f"  statistic={stats:.3f}\n"
                f"  p-value={pval:.3f}",
                fontsize=10,
                verticalalignment="center",
                horizontalalignment="left",
                transform=ax.transAxes,
            )
            ax.set_title(feature)
            ax.set_xlabel("timeline")
            ax.set_ylabel("Survival Probability")
            ax.legend(loc="center left", bbox_to_anchor=(1, 0.95))
            plt.tight_layout()
            plt.savefig(
                os.path.join(km_save_dir, f"rafael.km.{feature}.{n_std[i]}std.png"),
                dpi=200,
            )
            plt.close()
        return


def _read_the_last_nth_line(f, nth=1):
    """Returns the nth before last line of a file (n=1 gives last line)"""
    num_newlines = 0
    try:
        f.seek(-2, os.SEEK_END)
        while num_newlines < nth:
            f.seek(-2, os.SEEK_CUR)
            if f.read(1) == b"\n":
                num_newlines += 1
    except OSError:
        f.seek(0)
    last = f.readline().decode()
    return last


class Dashboard(UseCase):
    def __init__(self):
        super().__init__()

    def tail_log(self, logfile, n=20):
        f = open(logfile, "rb")
        log = []
        for i in range(n):
            log.append(_read_the_last_nth_line(f, i + 1).strip())
        return log


class NaiveLDPruning(UseCase):
    """
    Linkage Disequilibrium Pruning for Bed file
    Here we assumed that snp has been matched and filtered.
        - With same SNP number
        - With sane SNP ID and Allele
        - With Same order
        - maf and missing filtered
    If not, please run qc first.
    """

    def __init__(self):
        super().__init__()

    @classmethod
    def return_variables(cls, method_name: str):
        if method_name == "local_ldprune":
            return ["snp_list"]
        elif method_name == "global_match_snps":
            return ["snp_list"]
        elif method_name == "local_extract_snp":
            return ["out_path"]

    def local_ldprune(
        self,
        bfile_path: str,
        out_path: str,
        win_size: int,
        step: int,
        r2: float,
        extra_arg: str = "",
    ) -> NDArray:
        """
        Run LD prune by plink2

        Parameters
        ----------
            bfile_path : str
                input bfile prefix
            out_path: str
                output prune.in prefix
            win_size: int
                windows size for ld prune
            step: int
                step for ld prune
            r2: int
                r2 threshold for ld prune
            extra_arg: str
                extra argument pass to plink, default is empty string ""
        Returns
        -------
            snp_list : list
                The list of snp ids after ld purne.
        """
        snp_list = gwasprs.ld.prune_ld(
            bfile_path, out_path, win_size, step, r2, extra_arg
        )
        logging.info(f"There are {len(snp_list)} snp left after local ld prune.")
        return snp_list

    def global_match_snps(
        self, snp_lists: List[NDArray], method: Literal["union", "intersect"] = "union"
    ) -> NDArray:
        """
        Aggregate the shared SNPs among all edges.

        Parameters
        ----------
            snp_lists : list of list
                The lists of snp ids after ld prune from edges.
            method : str = "union", "intersect"
                method to aggregate snp lists
        Returns
        -------
            snp_list : list
                The list of snp ids shared among all edges.
        """
        snp_list = gwasprs.ld.match_snp_sets(snp_lists, method=method)
        logging.info(f"There are {len(snp_list)} snp matched among all edges.")
        return snp_list

    def local_extract_snp(
        self, bfile_path: str, out_path: str, snp_list: NDArray
    ) -> str:
        """
        Extract snp according to ld list.

        Parameters
        ----------
            bfile_path : str
                input bfile prefix
            out_path: str
                output bfile and snp_list prefix
            snp_list: NDArray
                The list of snp ids shared among all edges.
        Returns
        -------
            out_path : str
                input bfile and snp_list prefix
        """
        out_path = gwasprs.ld.extract_snps(bfile_path, out_path, snp_list)
        return out_path
