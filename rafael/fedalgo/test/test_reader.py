import os
import unittest
from rafael.fedalgo import gwasprs

def get_repo_path():
    return os.path.abspath(os.path.join(os.path.abspath(__file__), "../../../.."))

bfile_path = os.path.join(get_repo_path(), 'data/whole/hapmap1_100')
cov_path = os.path.join(get_repo_path(), 'data/whole/hapmap1_100.cov')
pheno_path = os.path.join(get_repo_path(), 'data/whole/hapmap1_100.pheno')


class BedReaderTestCase(unittest.TestCase):

    def setUp(self):
        self.reader = gwasprs.reader.BedReader(bfile_path)
        self.n_SNP = 100
        self.n_sample = 60

    def tearDown(self):
        self.bed = None

    def test_read(self):
        result = self.reader.read()
        self.assertEqual((self.n_sample, self.n_SNP), result.shape)

    def test_read_single_snp(self):
        result = self.reader.read_range(2)
        self.assertEqual((self.n_sample, 1), result.shape)

    def test_read_range_snps(self):
        result = self.reader.read_range(slice(5, 50, None))
        self.assertEqual((self.n_sample, 45), result.shape)


class FamReaderTestCase(unittest.TestCase):

    def setUp(self):
        self.n_sample = 60
        self.n_feature = 6
        self.reader = gwasprs.reader.FamReader(bfile_path)

    def tearDown(self):
        self.reader = None

    def test_read(self):
        result = self.reader.read()
        self.assertEqual((self.n_sample, self.n_feature), result.shape)

    def test_read_single_sample(self):
        result = self.reader.read_range(2)
        self.assertEqual((self.n_feature, ), result.shape)

    def test_read_range_samples(self):
        result = self.reader.read_range(range(3))
        self.assertEqual((3, self.n_feature), result.shape)


class CovReaderTestCase(unittest.TestCase):

    def setUp(self):
        self.n_sample = 60
        self.n_feature = 4
        self.reader = gwasprs.reader.CovReader(cov_path)

    def tearDown(self):
        self.reader = None

    def test_read(self):
        result = self.reader.read()
        self.assertEqual((self.n_sample, self.n_feature), result.shape)

    def test_read_single_sample(self):
        result = self.reader.read_range(10)
        self.assertEqual((self.n_feature, ), result.shape)

    def test_read_range_samples(self):
        result = self.reader.read_range(range(10))
        self.assertEqual((10, self.n_feature), result.shape)


class BimReaderTestCase(unittest.TestCase):

    def setUp(self):
        self.n_SNP = 100
        self.n_feature = 6
        self.reader = gwasprs.reader.BimReader(bfile_path)

    def tearDown(self):
        self.reader = None

    def test_read(self):
        result = self.reader.read()
        self.assertEqual((self.n_SNP, self.n_feature), result.shape)

    def test_read_simgle_snp(self):
        result = self.reader.read_range(10)
        self.assertEqual((self.n_feature, ), result.shape)

    def test_read_range_snps(self):
        result = self.reader.read_range(range(10))
        self.assertEqual((10, self.n_feature), result.shape)


class PhenotypeReaderTestCase(unittest.TestCase):

    def setUp(self):
        self.n_sample = 60
        self.n_feature = 3
        self.reader = gwasprs.reader.PhenotypeReader(pheno_path, 'pheno')

    def tearDown(self):
        self.reader = None

    def test_read(self):
        result = self.reader.read()
        self.assertEqual((self.n_sample, self.n_feature), result.shape)
