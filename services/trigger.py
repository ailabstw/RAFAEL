import requests
import json


class TaskSeries:
    def __init__(self, node_id, server_url):
        self.__node_id = node_id
        self.__server_url = server_url
        self.__tasks = []

    @property
    def server_url(self):
        return self.__server_url
    
    @property
    def node_id(self):
        return self.__node_id

    def add(self, *tasks):
        for task in tasks:
            self.__tasks.append(task(self.node_id, self.server_url))

    def __call__(self, **kwargs):
        """
        expected kwargs format:
            {
                'api1':{
                    'args1':value1,
                    'args2':value2
                },
                'api2':{
                    'args3':value3,
                    'args4':value4
                }
            }
        """
        for task in self.__tasks:
            task(**kwargs)


class Task:
    def __init__(self, node_id, server_url):
        self._server_url = server_url
        self._profile = {"node_id": node_id, "args":{}}

    def post(self, api, **kwargs):
        self._profile['args'].update(kwargs.get(api, {}))
        req = {**self._profile, 'api': api}
        print(
            f'Request:\n',
            f'{json.dumps(req, indent=4)}\n\n',
        )
        response = requests.post(self._server_url, json=req).json()
        print(
            f'Response\n',
            f'{json.dumps(response, indent=4)}\n\n',
            f'-'*20,
            f'\n'
        )
        return response

    def __call__(self, **kwargs):
        for api in self.apis:
            self.post(api, **kwargs)


class CoxPHRegression(Task):
    @property
    def apis(self):
        return ["CoxPHRegression"]
    

class KaplanMeier(Task):
    @property
    def apis(self):
        return ["KaplanMeier"]
    

class PCAfromTabular(Task):
    @property
    def apis(self):
        return ["PCAfromTabular"]
    

class SVDfromTabular(Task):
    @property
    def apis(self):
        return ["SVDfromTabular"]
    

class BasicBfileQC(Task):
    @property
    def apis(self):
        return ["BasicBfileQC"]


class LDPruning(Task):
    @property
    def apis(self):
        return ["LDPruning"]

    
class GenotypePCA(Task):
    @property
    def apis(self):
        return ["GenotypePCA"]

    
class CovariateStandardization(Task):
    @property
    def apis(self):
        return ["CovariateStdz"]


class QuantGWAS(Task):
    @property
    def apis(self):
        return ["QuantGWAS"]


class BinGWAS(Task):
    @property
    def apis(self):
        return ["BinGWAS"]


class FullQuantGWAS(Task):
    @property
    def apis(self):
        return ["FullQuantGWAS"]


class FullBinGWAS(Task):
    @property
    def apis(self):
        return ["FullBinGWAS"]

    
def quant_gwas(node_id, server_address):
    gwasconfig = {
        "config": {
            "clients": [
                "2",
                "3",
                "4"
            ],
            "compensators": [
                "5"
            ],
            "bfile_path": [
                "/volume/gwasfl/SCENARIOS_qc/I/QUANT/split1",
                "/volume/gwasfl/SCENARIOS_qc/I/QUANT/split2",
                "/volume/gwasfl/SCENARIOS_qc/I/QUANT/split3"
            ],
            "cov_path": [
                "/volume/gwasfl/SCENARIOS_qc/I/QUANT/split1.cov",
                "/volume/gwasfl/SCENARIOS_qc/I/QUANT/split2.cov",
                "/volume/gwasfl/SCENARIOS_qc/I/QUANT/split3.cov"
            ],
            "regression_save_dir": [
                "/tmp/",
                "/tmp/",
                "/tmp/"
            ],
            "local_qc_output_path": [
                "/tmp/qc",
                "/tmp/qc",
                "/tmp/qc"
            ],
            "global_qc_output_path": "/tmp/agg",
            "snp_chunk_size": 10000,
            "k2": 10
        }
    }
    
    Job = TaskSeries(node_id, server_address)
    Job.add(FullQuantGWAS)
    Job(**{'FullQuantGWAS':gwasconfig})
    
def binary_gwas(node_id, server_address):
    gwasconfig = {
        "config": {
            "clients": [
                "2",
                "3",
                "4"
            ],
            "compensators": [
                "5"
            ],
            "bfile_path": [
                # "/volume/gwasfl/SCENARIOS_qc/I/BINARY/split1",
                # "/volume/gwasfl/SCENARIOS_qc/I/BINARY/split2",
                # "/volume/gwasfl/SCENARIOS_qc/I/BINARY/split3"
                "/volume/gwasfl/TWB2_240418/DATA/GOUT/I/split0",
                "/volume/gwasfl/TWB2_240418/DATA/GOUT/I/split1",
                "/volume/gwasfl/TWB2_240418/DATA/GOUT/I/split2"
            ],
            "cov_path": [
                # "/volume/gwasfl/SCENARIOS_qc/I/BINARY/split1.cov",
                # "/volume/gwasfl/SCENARIOS_qc/I/BINARY/split2.cov",
                # "/volume/gwasfl/SCENARIOS_qc/I/BINARY/split3.cov"
                "/volume/gwasfl/TWB2_240418/DATA/GOUT/I/split0.cov",
                "/volume/gwasfl/TWB2_240418/DATA/GOUT/I/split1.cov",
                "/volume/gwasfl/TWB2_240418/DATA/GOUT/I/split2.cov"
            ],
            "pheno_path": [
                "/volume/gwasfl/TWB2_240418/DATA/GOUT/I/split0.pheno",
                "/volume/gwasfl/TWB2_240418/DATA/GOUT/I/split1.pheno",
                "/volume/gwasfl/TWB2_240418/DATA/GOUT/I/split2.pheno"
            ],
            "pheno_name": "GOUT",
            "regression_save_dir": [
                "/tmp/",
                "/tmp/",
                "/tmp/"
            ],
            "local_qc_output_path": [
                "/tmp/qc",
                "/tmp/qc",
                "/tmp/qc"
            ],
            "global_qc_output_path": "/tmp/agg",
            "snp_chunk_size": 10000,
            "k2": 10
        }
    }
    
    Job = TaskSeries(node_id, server_address)
    Job.add(FullBinGWAS)
    Job(**{'FullBinGWAS':gwasconfig})
    
    
if __name__ == '__main__':
    node_id = "1"
    server_address = "https://lab5-k8s.corp.ailabs.tw:443/group-genome/rafael-server/tasks"
    # binary_gwas(node_id, server_address)
    quant_gwas(node_id, server_address)
