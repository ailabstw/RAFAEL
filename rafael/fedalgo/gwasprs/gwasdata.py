from warnings import warn

import numpy as np
import pandas as pd

from .reader import (
    read_snp_list,
    read_ind_list,
    BedReader,
    FamReader,
    CovReader,
    BimReader,
    PhenotypeReader,
)
from .iterator import SNPIterator, SampleIterator


AUTOSOME_LIST = ()
for i in range(1, 23):
    AUTOSOME_LIST += (i, str(i), f"chr{i}")


def get_mask_idx(df):
    """
    Get the index of missing values in a dataframe.
    """
    mask_miss1 = df.isnull().any(axis=1)
    mask_miss2 = (df == (-9 or -9.0)).any(axis=1)
    return df[mask_miss1 | mask_miss2].index


def impute_cov(cov):
    # nanmean imputation
    col_means = cov.iloc[:, 2:].mean()
    cov.fillna(col_means, inplace=True)
    return cov


def create_unique_snp_id(bim, to_byte=True, to_dict=False):
    """
    Create the unique ID as CHR:POS:A1:A2
    If the A1 and A2 are switched, record the SNP index for adjusting the genptype values (0 > 2, 2 > 0).
    """
    unique_id, sorted_snp_idx = [], []
    bim = bim.reset_index(drop=True)
    for line in bim.iterrows():
        chr = str(line[1]["CHR"])
        pos = str(line[1]["POS"])
        allele = [str(line[1]["A1"])[:23], str(line[1]["A2"])[:23]]
        allele_sorted = sorted(allele)
        unique_id.append(f"{chr}:{pos}:{allele_sorted[0]}:{allele_sorted[1]}")
        if allele != allele_sorted:
            sorted_snp_idx.append(line[0])
    if to_byte:
        unique_id = np.array(unique_id, dtype="S")
    if to_dict:
        unique_id = dict(zip(unique_id, bim.ID))
    return unique_id, sorted_snp_idx


def redirect_genotype(GT, snp_idx):
    GT[:, snp_idx] = 2 - GT[:, snp_idx]
    return GT


def dropped_info(data, subset, cols):
    data_id = list(zip(*data[cols].to_dict("list").values()))
    subset_id = list(zip(*subset[cols].to_dict("list").values()))
    mask = ~np.isin(data_id, subset_id)
    dropped_idx = np.where(mask)[0]
    return data.iloc[dropped_idx, :]


def subset_samples(
    sample_list: (str, list, tuple), data, order=False, list_is_idx=False
):
    """
    Args:
        sample_list (str, list, tuple) : could be a list of sample IDs, a path to sample list file or a list of sample indices.
        data (pd.DataFrame) : the data to be extracted.

    Returns:
        subset_data (pd.DataFrame) : the subset of the data (the index of the subset_data has been reset)
        sample_idx : the indices of sample list in "data", not the indices in "subset_data".
        dropped_data (pd.DataFrame) : the dropped subset.
    """
    # Format sample list
    if isinstance(sample_list, str):
        sample_df = read_ind_list(sample_list)
    elif isinstance(sample_list, (tuple, list)) and not list_is_idx:
        sample_df = pd.DataFrame({"FID": sample_list, "IID": sample_list})
    else:
        sample_df = data.iloc[sample_list, :2].reset_index(drop=True)

    # Follow the order in the ind_list
    if order:
        subset_data = sample_df.merge(data, on=["FID", "IID"])
    else:
        subset_data = data.merge(sample_df, on=["FID", "IID"])

    if len(subset_data) == 0:
        raise IndexError

    # Get the indices of samples in original data for getting the ordered genotype matrix.
    sample_idx = subset_data.merge(data.reset_index())["index"].to_list()

    return subset_data, sample_idx


def subset_snps(snp_list: (str, list, tuple), data, order=False, list_is_idx=False):
    """
    Args:
        snp_list (str, list, tuple) : could be a list of SNP IDs, a path to snp list file or a list of snp indices.
        data (pd.DataFrame) : the data to be extracted.

    Returns:
        subset_data (pd.DataFrame) : the subset of the data (the index of the subset_data has been reset)
        snp_idx : the indices of snp list in "data", not the indices in "subset_data".
        dropped_data (pd.DataFrame) : the dropped subset.
    """
    # Format SNP list
    if isinstance(snp_list, str):
        snp_df = read_snp_list(snp_list)
    elif isinstance(snp_list, (tuple, list)) and not list_is_idx:
        snp_df = pd.DataFrame({"ID": snp_list})
    else:
        snp_df = data.iloc[snp_list, 1].reset_index(drop=True)

    # Follow the order of the snp_list
    if order:
        subset_snps = snp_df.merge(data, on=["ID"])
    else:
        subset_snps = data.merge(snp_df, on=["ID"])

    if len(subset_snps) == 0:
        raise IndexError

    # Get the indices of snps in original data for getting the ordered genotype matrix.
    snp_idx = subset_snps.merge(data.reset_index())["index"].to_list()

    return subset_snps, snp_idx


def index_non_missing_samples(FAM, COV=None):
    """
    Index samples without any missing values in FAM, pheno, or covariates.
    """
    FAM_rm_idx = set(get_mask_idx(FAM))
    if COV is not None:
        COV_rm_idx = get_mask_idx(COV)
        rm_sample_idx = FAM_rm_idx.union(COV_rm_idx)
    else:
        rm_sample_idx = FAM_rm_idx
    keep_ind_idx = set(FAM.index).difference(rm_sample_idx)

    return list(keep_ind_idx)


def create_snp_table(snp_id_list, rs_id_list):
    """
    Create the mapping table that unique IDs can be mapped to the rsIDs.
    """
    snp_id_list, idx = np.unique(snp_id_list, return_index=True)
    recover_order = np.argsort(idx)
    snp_id_list = snp_id_list[recover_order]
    rs_id_list = np.array(rs_id_list)[idx][recover_order]
    snp_id_table = dict(zip(snp_id_list, rs_id_list))

    return list(snp_id_table.keys()), snp_id_table


class GWASData:
    """
    The GWASData performs three main operations, subsect extraction, dropping samples with missing values
    and add unique position ID for each SNP.

    subset()
        This function allows multiple times to extract the subset with the given sample list and SNP list.
        ex. subset(sample_list1)
                    :
            subset(sample_list9)

        Limitations: If the sample list is an index list, and the SNP list is not, \
                     please do it in two steps, the first step can be subset(sample_list, list_is_idx), \
                     the second step can be subset(snp_list). Switching the order is fine.
                     Similar situation is the same as `order`.

        Args:
            sample_list (optional, str, tuple, list, default=None) : Allows the path to the snp list, a list of sample IDs or a list of sample indices. \
                                                                     Note that the np.array dtype is not supported. The default is return the whole samples.

            snp_list (optional, str, tuple, list, default=None)    : Allows the path to the SNP list, a list of SNP IDs or a list of SNP indices. \
                                                                     Note that the np.array dtype is not supported. The default is return the whole SNPs.

            order (boolean, default=False) : Determines the sample and snp order in fam, bim, cov and genotype. \
                                             If True, the order follows the given sample/snp list.

            list_is_idx (boolean, default=False) : If the `sample_list` or the `snp_list` are indices, this parameter should be specified as True.

        Returns:
            Subset of fam (pd.DataFrame)
            Subset of cov (pd.DataFrame)
            Subset of bim (pd.DataFrame)
            Subset of genotype (np.ndarray)
            dropped_fam (pd.DataFrame)
            dropped_cov (pd.DataFrame)
            dropped_bim (pd.DataFrame)


    drop_missing_samples():
        Drop samples whose phenotype or covariates contain missing values ('', NaN, -9, -9.0).

        Returns:
            Subset of fam (pd.DataFrame) : samples without missing values
            Subset of cov (pd.DataFrame) : samples without missing values
            dropped_fam (pd.DataFrame) : samples with missing values
            dropped_cov (pd.DataFrame) : samples with missing values


    add_unique_snp_id():
        Add the unique IDs for each SNP.

        Returns:
            bim : With unique IDs and rsIDs
            genotype : If the A1 and A2 are switched, snp array = 2 - snp array
    """

    def __init__(self, genotype, phenotype, snp, covariate):
        self.__genotype = genotype
        self.__phenotype = phenotype
        self.__snp = snp
        self.__covariate = covariate
        self.ensure_sample_order()

    @classmethod
    def read(cls, bfile_path, cov_path=None, pheno_path=None, pheno_name="PHENO1"):
        GT = BedReader(bfile_path).read()
        bim = BimReader(bfile_path).read()
        fam, cov = format_sample_metadata(bfile_path, cov_path, pheno_path, pheno_name)
        return cls(GT, fam, bim, cov)

    def standard(self):
        self.subset()
        self.drop_missing_samples()
        self.add_unique_snp_id()

    def custom(self, **kwargs):
        self.subset(**kwargs)

        if kwargs.get("impute_cov") is True:
            self.impute_covariates()

        if kwargs.get("drop_missing_samples", True):
            self.drop_missing_samples()

        if kwargs.get("add_unique_snp_id") is True:
            self.add_unique_snp_id()

    def subset(
        self, sample_list=None, snp_list=None, order=False, list_is_idx=False, **kwargs
    ):
        # Sample information
        if sample_list:
            self.__phenotype, sample_idx = subset_samples(
                sample_list, self.__phenotype, order, list_is_idx
            )

            if self.__covariate is not None:
                self.__covariate, _ = subset_samples(
                    sample_list, self.__covariate, order, list_is_idx
                )
        else:
            sample_idx = slice(None)

        # SNP information
        if snp_list:
            self.__snp, snp_idx = subset_snps(snp_list, self.__snp, order, list_is_idx)
        else:
            snp_idx = slice(None)

        # Genotype information
        if (sample_list or snp_list) is not None:  # Don't remove
            self.__genotype = self.__genotype[
                np.s_[sample_idx, snp_idx]
            ]  # This step consts lots of time

    def impute_covariates(self):
        self.__covariate = impute_cov(self.__covariate)

    def drop_missing_samples(self):
        # Re-subset the samples without any missing values
        sample_idx = index_non_missing_samples(self.__phenotype, self.__covariate)
        self.subset(sample_list=sample_idx, list_is_idx=True)

    def add_unique_snp_id(self):
        unique_id, sorted_snp_idx = create_unique_snp_id(
            self.__snp, to_byte=False, to_dict=False
        )
        self.__snp["rsID"] = self.__snp["ID"]
        self.__snp["ID"] = unique_id
        self.__genotype = redirect_genotype(
            self.__genotype, sorted_snp_idx
        )  # This step consts lots of time

    def ensure_sample_order(self):
        if self.__covariate is not None:
            self.__covariate = format_cov(self.__covariate, self.__phenotype)

    @property
    def phenotype(self):
        return self.__phenotype

    @property
    def sample_id(self):
        return list(zip(self.__phenotype.FID, self.__phenotype.IID))

    @property
    def covariate(self):
        return self.__covariate

    @property
    def snp(self):
        return self.__snp

    @property
    def snp_id(self):
        return list(self.__snp.ID)

    @property
    def autosome_snp_id(self):
        return list(self.__snp[self.__snp.CHR.isin(AUTOSOME_LIST)].ID)

    @property
    def rsID(self):
        return list(self.__snp.rsID)

    @property
    def autosome_rsID(self):
        return list(self.__snp[self.__snp.CHR.isin(AUTOSOME_LIST)].rsID)

    @property
    def allele(self):
        return list(zip(self.__snp.A1, self.__snp.A2))

    @property
    def genotype(self):
        return self.__genotype

    @property
    def snp_table(self):
        assert "rsID" in self.__snp.columns
        return create_snp_table(self.snp_id, self.rsID)

    @property
    def autosome_snp_table(self):
        assert "rsID" in self.__snp.columns
        return create_snp_table(self.autosome_snp_id, self.autosome_rsID)

    def __eq__(self, other):
        if self.covariate is None and other.covariate is None:
            return (
                np.array_equal(self.genotype, other.genotype, equal_nan=True)
                and self.phenotype.equals(other.phenotype)
                and self.snp.equals(other.snp)
            )
        else:
            return (
                np.array_equal(self.genotype, other.genotype, equal_nan=True)
                and self.phenotype.equals(other.phenotype)
                and self.snp.equals(other.snp)
                and self.covariate.equals(other.covariate)
            )


def format_cov(cov: pd.DataFrame, fam: pd.DataFrame):
    """
    Read covarities from a given path and map to the corresponding FAM file.
    """
    # the samples in .fam missing covariate values will fill with NaNs
    return fam[["FID", "IID"]].merge(cov, on=["FID", "IID"], how="left", sort=False)


def format_fam(fam: pd.DataFrame, pheno: pd.DataFrame):
    """
    Replace the FAM file with the corresponding pheno file.
    """
    fam.drop(columns="PHENO1", inplace=True)
    # the samples in .fam missing phenotypes values will fill with NaNs
    fam = fam.merge(pheno, on=["FID", "IID"], how="left", sort=False)
    return fam


def format_sample_metadata(
    bfile_path, cov_path=None, pheno_path=None, pheno_name="PHENO1"
):
    """ """
    fam = FamReader(bfile_path).read()
    if pheno_path is not None:
        pheno = PhenotypeReader(pheno_path, pheno_name).read()
        fam = format_fam(fam, pheno)

    if cov_path:
        cov = CovReader(cov_path).read()
        cov = format_cov(cov, fam)
    else:
        cov = None
    return fam, cov


def get_qc_metadata(
    bfile_path, cov_path=None, pheno_path=None, pheno_name="PHENO1", autosome_only=True
):
    # SNPs
    bim = BimReader(bfile_path).read()
    if autosome_only:
        bim = bim[bim.CHR.isin(AUTOSOME_LIST)]
    autosome_snp_id, _ = create_unique_snp_id(bim, to_byte=False, to_dict=False)
    autosome_rsID = list(bim.ID)
    autosome_snp_list, autosome_snp_table = create_snp_table(
        autosome_snp_id, autosome_rsID
    )

    # Samples
    fam, _ = format_sample_metadata(bfile_path, cov_path, pheno_path, pheno_name)
    sample_id = list(zip(fam.FID, fam.IID))
    return autosome_snp_list, sample_id, autosome_snp_table


def read_gwasdata(bfile_path, cov_path=None, pheno_path=None, pheno_name="PHENO1"):
    warn("read_gwasdata is deprecated.", DeprecationWarning, stacklevel=2)
    GT = BedReader(bfile_path).read()
    bim = BimReader(bfile_path).read()
    fam, cov = format_sample_metadata(bfile_path, cov_path, pheno_path, pheno_name)
    return GWASData(GT, fam, bim, cov)


class GWASDataIterator:
    def __init__(
        self, bfile_path, cov_path=None, style="sample-snp", sample_step=1, snp_step=1
    ):
        self.bedreader = BedReader(bfile_path)
        self.bimreader = BimReader(bfile_path)
        self.famreader = FamReader(bfile_path)
        if cov_path is not None:
            self.covreader = CovReader(cov_path)

        q1, r1 = divmod(self.bedreader.n_sample, sample_step)
        q1 = q1 + 1 if r1 != 0 else q1
        q2, r2 = divmod(self.bedreader.n_snp, snp_step)
        q2 = q2 + 1 if r2 != 0 else q2

        if style == "sample":
            self.iterator = SampleIterator(self.bedreader.n_sample, sample_step)
            self._total_steps = q1

        elif style == "snp":
            self.iterator = SNPIterator(self.bedreader.n_snp, snp_step)
            self._total_steps = q2

        elif style == "sample-snp":
            self.iterator = SampleIterator(self.bedreader.n_sample, sample_step).snps(
                self.bedreader.n_snp, snp_step
            )
            self._total_steps = q1 * q2

        elif style == "snp-sample":
            self.iterator = SNPIterator(self.bedreader.n_snp, snp_step).samples(
                self.bedreader.n_sample, sample_step
            )
            self._total_steps = q1 * q2

        else:
            raise Exception(f"{style} style is not supported.")

    def reset(self):
        self.iterator.reset()

    def get_data(self, sample_slc, snp_slc):
        chunk_bed = self.bedreader.read_range((sample_slc, snp_slc))
        chunk_fam = self.famreader.read_range(sample_slc)
        chunk_cov = (
            self.covreader.read_range(sample_slc)
            if hasattr(self, "covreader")
            else None
        )
        chunk_bim = self.bimreader.read_range(snp_slc)
        return GWASData(chunk_bed, chunk_fam, chunk_bim, chunk_cov)

    def __iter__(self):
        return self

    def __next__(self):
        if not self.is_end():
            return self.get_data(*next(self.iterator))
        else:
            raise StopIteration

    def __len__(self):
        # If usigin the `get_data` method manually,
        # the total steps may change
        return self._total_steps

    def is_end(self):
        return self.iterator.is_end()
