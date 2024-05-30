import sys, os
import unittest
import logging

from rafael.fedalgo.gwasprs.qc import cal_qc_client, filter_snp, cal_het_sd, create_filtered_bed, filter_ind
from rafael.fedalgo.gwasprs.gwasdata import AUTOSOME_LIST
from rafael.fedalgo.gwasprs.reader import BimReader, FamReader

logging.basicConfig(level=logging.DEBUG)

def get_repo_path():
    return os.path.abspath(os.path.join(os.path.abspath(__file__), "../../../.."))


class QcTestCase(unittest.TestCase):

    def setUp(self):
        self.bfile_path = os.path.join(get_repo_path(), 'data/whole/hapmap1_100')
        self.output_path = "/tmp/qc"
        self.het_bin = 1000
        self.het_range = (-0.5, 0.5)
        self.sample_count = 8000

        bim = BimReader(f"{self.bfile_path}").read()
        fam = FamReader(f"{self.bfile_path}").read()
        self.snp_list = bim.loc[bim.CHR.isin(AUTOSOME_LIST)].ID.to_numpy()
        self.sample_count = len(fam.index)
        self.fid_iid_list = list(zip(fam.FID.values, fam.IID.values))


    def test_cal_snp_qc(self):

        # edge
        allele_count, het_hist, het, n_obs = cal_qc_client(
            self.bfile_path, self.output_path, self.snp_list,
            self.het_bin, self.het_range
        )

        # agg
        snp_id = filter_snp(
            allele_count = allele_count,
            snp_id = self.snp_list,
            sample_count =  n_obs,
            save_path = self.output_path,
            geno = 0.1,
            hwe = 5e-7,
            maf = 0.01,
        )

        # agg
        het_std, het_mean = cal_het_sd(het_hist, self.het_range, self.het_bin)

        # edge
        remove_list = filter_ind(het, het_mean, het_std, 5, self.fid_iid_list)

        # edge
        create_filtered_bed(
            bfile_path = self.bfile_path,
            filtered_bfile_path = self.output_path,
            keep_snps = snp_id,
            mind = 0.05 ,
            keep_inds = self.fid_iid_list
        )
