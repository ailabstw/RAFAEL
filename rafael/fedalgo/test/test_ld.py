import os
import unittest

from rafael.fedalgo.gwasprs.ld import prune_ld, match_snp_sets, extract_snps, read_snp_list


def get_repo_path():
    return os.path.abspath(os.path.join(os.path.abspath(__file__), "../../../.."))

class LDPruningTestCase(unittest.TestCase):
    def setUp(self):
        self.bfile_paths = [
            os.path.join(get_repo_path(), 'data/whole/hapmap1_100')
        ]
        self.output_paths = ["/tmp/hapmap1_100"]
        self.ans_snps = os.path.join(get_repo_path(), 'data/whole/hapmap1_100.single.prune.in')
        """
        Ground Truth Command
        plink2 --bfile fed-algo/data/test_bfile/hapmap1_100
               --indep-pairwise 50 5 0.2
               --bad-ld
               --out /tmp/hapmap1_100
        """

    def test_ld(self):
        # step 1 local, add --bad-ld to allow run in < 50 sample
        snp_lists = []
        for bfile_path, output_path in zip(self.bfile_paths, self.output_paths):
            snp_list = prune_ld(bfile_path, output_path, 50, 5, 0.2, "--bad-ld")
            snp_lists.append(snp_list)

        # step 2 global
        snp_list = match_snp_sets(snp_lists, method="union")

        # step 3 local
        for bfile_path, output_path in zip(self.bfile_paths, self.output_paths):
            extract_snps(bfile_path, output_path, snp_list)

        # check ground truth
        ans_snp_list = read_snp_list(self.ans_snps)
        for i in snp_list:
            if i not in ans_snp_list:
                raise KeyError(f"{i} not found in ans_snp_list")

        for i in ans_snp_list:
            if i not in snp_list:
                raise KeyError(f"{i} not found in snp_list")
