import unittest
import numpy as np

from rafael.fedalgo import gwasprs


class IteratorTestCase(unittest.TestCase):

    def setUp(self):
        """
        divisible (SNPIterator)
        divisible and not divisible (SNPIterator.samples)
        divisible and divisible
        not divisible (SampleIterator)
        not divisible and divisible (SampleIterator.snps)
        not divisible and not divisible
        """
        self.n_SNP = 60
        self.n_sample = 79
        self.snp_step = 15
        self.sample_step = 20

    def tearDown(self):
        pass

    def test_IndexIterator(self):
        iter = gwasprs.iterator.IndexIterator(self.n_SNP, step=self.snp_step)
        self.assertEqual(slice(0, self.snp_step, None), next(iter))
        self.assertEqual(slice(self.snp_step, 2*self.snp_step, None), next(iter))
        self.assertFalse(iter.is_end())

    def test_SNPIterator(self):
        iter = gwasprs.iterator.SNPIterator(self.n_SNP, step=self.snp_step)
        for i in range(0, self.n_SNP, self.snp_step):
            ans = np.s_[:, i:min(i+self.snp_step, self.n_SNP)]
            result = next(iter)
            self.assertEqual(ans, result)

    def test_SNPIterator_samples(self):
        iter = gwasprs.iterator.SNPIterator(self.n_SNP, step=self.snp_step).samples(self.n_sample, step=self.sample_step)
        for i in range(0, self.n_SNP, self.snp_step):
            for j in range(0, self.n_sample, self.sample_step):
                ans = np.s_[
                    j:min(j+self.sample_step, self.n_sample),
                    i:min(i+self.snp_step, self.n_SNP)
                ]
                result = next(iter)
                self.assertEqual(ans, result)
                
        iter = gwasprs.iterator.SNPIterator(self.n_SNP, step=self.snp_step).samples(self.n_SNP, step=self.snp_step)
        for i in range(0, self.n_SNP, self.snp_step):
            for j in range(0, self.n_SNP, self.snp_step):
                ans = np.s_[
                    j:min(j+self.snp_step, self.n_SNP),
                    i:min(i+self.snp_step, self.n_SNP)
                ]
                result = next(iter)
                self.assertEqual(ans, result)

    def test_SampleIterator(self):
        iter = gwasprs.iterator.SampleIterator(self.n_sample, step=self.sample_step)
        for i in range(0, self.n_sample, self.sample_step):
            ans = np.s_[i:min(i+self.sample_step, self.n_sample), :]
            result = next(iter)
            self.assertEqual(ans, result)

    def test_SampleIterator_snps(self):
        iter = gwasprs.iterator.SampleIterator(self.n_sample, step=self.sample_step).snps(self.n_SNP, step=self.snp_step)
        for i in range(0, self.n_sample, self.sample_step):
            for j in range(0, self.n_SNP, self.snp_step):
                ans = np.s_[
                    i:min(i+self.sample_step, self.n_sample),
                    j:min(j+self.snp_step, self.n_SNP)
                ]
                result = next(iter)
                self.assertEqual(ans, result)

        iter = gwasprs.iterator.SampleIterator(self.n_sample, step=self.sample_step).snps(self.n_sample, step=self.sample_step)
        for i in range(0, self.n_sample, self.sample_step):
            for j in range(0, self.n_sample, self.sample_step):
                ans = np.s_[
                    i:min(i+self.sample_step, self.n_sample),
                    j:min(j+self.sample_step, self.n_sample)
                ]
                result = next(iter)
                self.assertEqual(ans, result)
