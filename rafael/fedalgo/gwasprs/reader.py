import pandas as pd
from bed_reader import open_bed


def read_snp_list(snp_list_path) -> pd.Series:
    """
    Read a snp list from a given path.
    """
    snp_list = pd.read_csv(snp_list_path, sep=r"\s+", header=None)
    snp_list.columns = ["ID"]
    return snp_list


def read_ind_list(ind_list_path) -> pd.DataFrame:
    """
    Read a sample list from a given path.
    """
    ind_list = pd.read_csv(ind_list_path, sep=r"\s+", header=None).iloc[:, :2]
    ind_list.columns = ["FID", "IID"]
    return ind_list


class BedReader:
    def __init__(self, bfile_path: str):
        self.bedloader = open_bed(f"{bfile_path}.bed")

    @property
    def n_snp(self):
        return self.bedloader.sid_count

    @property
    def n_sample(self):
        return self.bedloader.iid_count

    def read(self):
        return self.bedloader.read()

    def read_range(self, range):
        return self.bedloader.read(index=range)


class FamReader:
    def __init__(self, bfile_path):
        self.loader = pd.read_csv(
            f"{bfile_path}.fam", iterator=True, sep=r"\s+", header=None
        )
        self.fam = None

    def read(self):
        if self.fam is None:
            self.fam = self.loader.read()
            self.set_columns()
        return self.fam

    def read_range(self, range):
        self.read()
        return self.fam.iloc[range, :]

    def set_columns(self):
        self.fam.columns = ["FID", "IID", "P", "M", "SEX", "PHENO1"]
        self.fam.FID = self.fam.FID.astype(str)
        self.fam.IID = self.fam.IID.astype(str)


class CovReader:
    def __init__(self, cov_path):
        sep = "," if ".csv" in cov_path else r"\s+"
        self.cov = pd.read_csv(cov_path, sep=sep)

    def read(self):
        self.set_columns()
        return self.cov

    def read_range(self, range):
        self.read()
        return self.cov.iloc[range, :]

    def set_columns(self):
        self.cov.FID = self.cov.FID.astype(str)
        self.cov.IID = self.cov.IID.astype(str)
        self.cov.drop_duplicates(subset=["FID", "IID"], inplace=True)


class BimReader:
    def __init__(self, bfile_path):
        self.loader = pd.read_csv(
            f"{bfile_path}.bim", iterator=True, sep=r"\s+", header=None
        )
        self.bim = None

    def read(self):
        if self.bim is None:
            self.bim = self.loader.read()
            self.set_columns()
        return self.bim

    def read_range(self, range):
        self.read()
        return self.bim.iloc[range, :]

    def set_columns(self):
        self.bim.columns = ["CHR", "ID", "cM", "POS", "A1", "A2"]
        self.bim.A1 = self.bim.A1.astype(str).replace("0", ".")
        self.bim.A2 = self.bim.A2.astype(str).replace("0", ".")
        self.bim.ID = self.bim.ID.astype(str)


class PhenotypeReader:
    def __init__(self, pheno_path, pheno_name) -> None:
        sep = "," if ".csv" in pheno_path else r"\s+"
        self.pheno = pd.read_csv(pheno_path, sep=sep)
        self.pheno = self.pheno[["FID", "IID", pheno_name]]
        self.set_columns()

    def set_columns(self):
        self.pheno.columns = ["FID", "IID", "PHENO1"]
        self.pheno.FID = self.pheno.FID.astype(str)
        self.pheno.IID = self.pheno.IID.astype(str)

    def read(self):
        return self.pheno
