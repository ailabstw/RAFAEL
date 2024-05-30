import unittest
import os

from rafael.utils import get_base_dir
from rafael.pipelines import PipelineController

BASE_DIR = get_base_dir()
        
class GWASPipelineTestCase(unittest.TestCase):
    def setUp(self):
        pass
    
    def test_qc_quant_gwas(self):
        pipeline = PipelineController(os.path.join(BASE_DIR, "configs/pipeline_gwas_qclinear.yml"))
        pipeline()
        
    def test_qc_binary_gwas(self):
        pipeline = PipelineController(os.path.join(BASE_DIR, "configs/pipeline_gwas_qclogistic.yml"))
        pipeline()
        
    def test_qc_pca(self):
        pipeline = PipelineController(os.path.join(BASE_DIR, "configs/pipeline_gwas_qcpca.yml"))
        pipeline()
