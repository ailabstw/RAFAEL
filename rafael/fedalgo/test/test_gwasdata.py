import unittest
import os
import numpy as np
import pandas as pd
from rafael.fedalgo import gwasprs

def get_repo_path():
    return os.path.abspath(os.path.join(os.path.abspath(__file__), "../../../.."))

bfile_path = os.path.join(get_repo_path(), 'data/whole/hapmap1_100')
cov_path = os.path.join(get_repo_path(), 'data/whole/hapmap1_100.cov')
pheno_path = os.path.join(get_repo_path(), 'data/whole/hapmap1_100.pheno')


class GWASData_Standard_TestCase(unittest.TestCase):

    def setUp(self):
        # General data
        self.bed = gwasprs.reader.BedReader(bfile_path).read()
        self.bim = gwasprs.reader.BimReader(bfile_path).read()
        pheno = gwasprs.reader.PhenotypeReader(pheno_path, 'pheno').read()
        fam = gwasprs.reader.FamReader(bfile_path).read()
        self.fam = gwasprs.gwasdata.format_fam(fam, pheno)
        cov = gwasprs.reader.CovReader(cov_path).read()
        self.cov = gwasprs.gwasdata.format_cov(cov, self.fam)

        # GWASData object
        self.gwasdata = gwasprs.gwasdata.GWASData.read(bfile_path, cov_path, pheno_path, 'pheno')

    def tearDown(self):
        self.bed = None
        self.bim = None
        self.fam = None
        self.cov = None
        self.gwasdata = None

    def test_data_formation(self):
        self.gwasdata.subset()
        pd.testing.assert_frame_equal(self.fam, self.gwasdata.phenotype)
        pd.testing.assert_frame_equal(self.bim, self.gwasdata.snp)
        pd.testing.assert_frame_equal(self.cov, self.gwasdata.covariate)
        np.testing.assert_allclose(self.bed, self.gwasdata.genotype, equal_nan=True)

    def test_drop_missing_samples(self):
        missing_idx = list(set(gwasprs.gwasdata.get_mask_idx(self.fam)).union(gwasprs.gwasdata.get_mask_idx(self.cov)))
        keep_idx = list(set(self.fam.index).difference(missing_idx))
        self.fam = self.fam.iloc[keep_idx,:].reset_index(drop=True)
        self.cov = self.cov.iloc[keep_idx,:].reset_index(drop=True)
        # self.GT = self.bed.read()[keep_idx,:]

        self.gwasdata.subset()
        self.gwasdata.drop_missing_samples()

        pd.testing.assert_frame_equal(self.fam, self.gwasdata.phenotype)
        pd.testing.assert_frame_equal(self.cov, self.gwasdata.covariate)
        # np.testing.assert_allclose(self.GT, self.gwasdata.GT, equal_nan=True)

    def test_standard(self):
        missing_idx = list(set(gwasprs.gwasdata.get_mask_idx(self.fam)).union(gwasprs.gwasdata.get_mask_idx(self.cov)))
        keep_idx = list(set(self.fam.index).difference(missing_idx))
        self.fam = self.fam.iloc[keep_idx,:].reset_index(drop=True)
        self.cov = self.cov.iloc[keep_idx,:].reset_index(drop=True)
        # self.GT = self.bed.read()[keep_idx,:]

        self.gwasdata.standard()

        pd.testing.assert_frame_equal(self.fam, self.gwasdata.phenotype)
        pd.testing.assert_frame_equal(self.cov, self.gwasdata.covariate)
        # np.testing.assert_allclose(self.GT, self.gwasdata.GT, equal_nan=True)


class GWASDataIteratorTestCase(unittest.TestCase):

    def setUp(self):
        self.bedreader = gwasprs.reader.BedReader(bfile_path)
        self.genotype = self.bedreader.read()
        self.n_SNP = self.bedreader.n_snp
        self.n_sample = self.bedreader.n_sample
        self.snp_default_step = 15
        self.sample_default_step = 11

        self.phenotype = gwasprs.reader.FamReader(bfile_path).read()
        self.cov = gwasprs.reader.CovReader(cov_path).read()
        self.snp = gwasprs.reader.BimReader(bfile_path).read()

    def tearDown(self):
        self.genotype = None
        self.phenotype = None
        self.cov = None
        self.snp = None

    def test_iterate_sample(self):
        idx_iter = gwasprs.gwasdata.SampleIterator(self.n_sample, self.sample_default_step)
        dataiter = gwasprs.gwasdata.GWASDataIterator(bfile_path, None,
                                                     style="sample",
                                                     sample_step=self.sample_default_step)

        count = 0
        for idx, result in zip(idx_iter, dataiter):
            genotype = self.genotype[idx]
            snp = self.snp
            phenotype = self.phenotype.iloc[idx[0]]
            ans = gwasprs.gwasdata.GWASData(genotype, phenotype, snp, None)

            np.array_equal(ans.genotype, result.genotype, equal_nan=True)
            pd.testing.assert_frame_equal(ans.snp, result.snp)
            pd.testing.assert_frame_equal(ans.phenotype, result.phenotype)
            self.assertEqual(ans, result)
            count += 1
            
        assert count == len(dataiter)

    def test_iterate_snp(self):
        idx_iter = gwasprs.gwasdata.SNPIterator(self.n_SNP, self.snp_default_step)
        dataiter = gwasprs.gwasdata.GWASDataIterator(bfile_path, None,
                                                     style="snp",
                                                     snp_step=self.snp_default_step)

        count = 0
        for idx, result in zip(idx_iter, dataiter):
            genotype = self.genotype[idx]
            snp = self.snp.iloc[idx[1]]
            phenotype = self.phenotype
            ans = gwasprs.gwasdata.GWASData(genotype, phenotype, snp, None)

            np.array_equal(ans.genotype, result.genotype, equal_nan=True)
            pd.testing.assert_frame_equal(ans.snp, result.snp)
            pd.testing.assert_frame_equal(ans.phenotype, result.phenotype)
            self.assertEqual(ans, result)
            count += 1
            
        assert count == len(dataiter)

    def test_iterate_sample_snp(self):
        idx_iter = gwasprs.gwasdata.SampleIterator(self.n_sample, self.sample_default_step).snps(self.n_SNP, self.snp_default_step)
        dataiter = gwasprs.gwasdata.GWASDataIterator(bfile_path, None,
                                                     style="sample-snp",
                                                     snp_step=self.snp_default_step,
                                                     sample_step=self.sample_default_step)

        count = 0
        for idx, result in zip(idx_iter, dataiter):
            genotype = self.genotype[idx]
            snp = self.snp.iloc[idx[1]]
            phenotype = self.phenotype.iloc[idx[0]]
            ans = gwasprs.gwasdata.GWASData(genotype, phenotype, snp, None)

            self.assertEqual(ans, result)
            count += 1
            
        assert count == len(dataiter)

    def test_iterate_snp_sample(self):
        idx_iter = gwasprs.gwasdata.SNPIterator(self.n_SNP, self.snp_default_step).samples(self.n_sample, self.sample_default_step)
        dataiter = gwasprs.gwasdata.GWASDataIterator(bfile_path, None,
                                                     style="snp-sample",
                                                     snp_step=self.snp_default_step,
                                                     sample_step=self.sample_default_step)

        count = 0
        for idx, result in zip(idx_iter, dataiter):
            genotype = self.genotype[idx]
            snp = self.snp.iloc[idx[1]]
            phenotype = self.phenotype.iloc[idx[0]]
            ans = gwasprs.gwasdata.GWASData(genotype, phenotype, snp, None)

            self.assertEqual(ans, result)
            count += 1
        
        assert count == len(dataiter)

    def test_iterate_sample_with_cov(self):
        idx_iter = gwasprs.gwasdata.SampleIterator(self.n_sample, self.sample_default_step)
        dataiter = gwasprs.gwasdata.GWASDataIterator(bfile_path, cov_path,
                                                     style="sample",
                                                     sample_step=self.sample_default_step)

        count = 0
        for idx, result in zip(idx_iter, dataiter):
            genotype = self.genotype[idx]
            snp = self.snp
            phenotype = self.phenotype.iloc[idx[0]]
            cov = self.cov.iloc[idx[0]]
            ans = gwasprs.gwasdata.GWASData(genotype, phenotype, snp, cov)

            self.assertEqual(ans, result)
            count += 1
        
        assert count == len(dataiter)

    def test_iterate_snp_with_cov(self):
        idx_iter = gwasprs.gwasdata.SNPIterator(self.n_SNP, self.snp_default_step)
        dataiter = gwasprs.gwasdata.GWASDataIterator(bfile_path, cov_path,
                                                     style="snp",
                                                     snp_step=self.snp_default_step)

        count = 0
        for idx, result in zip(idx_iter, dataiter):
            genotype = self.genotype[idx]
            snp = self.snp.iloc[idx[1]]
            phenotype = self.phenotype
            cov = self.cov
            ans = gwasprs.gwasdata.GWASData(genotype, phenotype, snp, cov)

            self.assertEqual(ans, result)
            count += 1
            
        assert count == len(dataiter)
