import os
import unittest

from rafael.fedalgo.gwasprs.gwasplot import read_glm, format_glm, prepare_manhattan, plot_manhattan, prepare_qq, plot_qq

def get_repo_path():
    return os.path.abspath(os.path.join(os.path.abspath(__file__), "../../../.."))

class GWASPlotTestCase(unittest.TestCase):
    def setUp(self):
        self.glm_path = os.path.join(get_repo_path(), "data/whole/glm/hapmap1_100.PHENO1.glm.linear")
        self.output_prefix = "/tmp/hapmap1_100"

    def test_plot_manhattan(self):
        glm = read_glm(self.glm_path, "#CHROM", "POS", "P", "\t")
        glm, max_log_p = format_glm(glm)
        
        manhattan_glm = prepare_manhattan(glm)
        plot_manhattan(
            *manhattan_glm,
            self.output_prefix,
            max_log_p=max_log_p
        )
        
    def test_plot_qq(self):
        glm = read_glm(self.glm_path, "#CHROM", "POS", "P", "\t")
        glm, _ = format_glm(glm)
        
        qq_glm = prepare_qq(glm)
        plot_qq(qq_glm, self.output_prefix)
