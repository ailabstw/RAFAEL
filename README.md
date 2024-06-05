![title](./docs/source/_static/logo.png)

**RAFAEL** (**RA**pid **F**ederated **A**nalysis in **EL**astic Framework) is a federated analytic framework provides two high-level features:
- composable federated analytic algorithms in biomedical domain
- accelerate the computation using the [JAX](https://github.com/google/jax) library, which can be further extended to use GPU.

We provide the Dockerfile to make users easily deploy a federated analysis service.

**Index**
- [**The Basic Usage of RAFAEL**](#the-basic-usage-of-rafael)
  - [**1. Clone the RAFAEL Repository**](#1-clone-the-rafael-repository)
  - [**2. Build the RAFAEL Service Docker Image**](#2-build-the-rafael-service-docker-image)
  - [**3. Initialize a RAFAEL service**](#3-initialize-a-rafael-service)
    - [**3.0 Create the Local Docker Network**](#30-create-the-local-docker-network)
    - [**3.1 Run a RAFAEL Server**](#31-run-a-rafael-server)
    - [**3.2 Run a RAFAEL Compensator**](#32-run-a-rafael-compensator)
    - [**3.3 Run RAFAEL Clients**](#33-run-rafael-clients)
  - [**4. Run a Demo**](#4-run-a-demo)
    - [**4.0 The introduction of analysis request**](#40-the-introduction-of-analysis-request)
    - [**Post an analysis request to server**](#post-an-analysis-request-to-server)
    - [**See also**](#see-also)
    - [**Example**](#example)
    - [**4.1 Genome Wide Association Study (GWAS)**](#41-genome-wide-association-study-gwas)
    - [**4.2 Principal Component Analysis (PCA)**](#42-principal-component-analysis-pca)
    - [**4.3 Singular Value Decomposition (SVD)**](#43-singular-value-decomposition-svd)
    - [**4.4 Cox Proportion Hazard Regression**](#44-cox-proportion-hazard-regression)
    - [**4.5 Kaplan-Meier**](#45-kaplan-meier)
  - [**5. Run a Federated Analysis on Your Own Dataset**](#5-run-a-federated-analysis-on-your-own-dataset)
    - [**Example:**](#example-1)
- [**The Advanced Usage of RAFAEL**](#the-advanced-usage-of-rafael)
  - [**Environment Variables to Set Up Service**](#environment-variables-to-set-up-service)
    - [**Service Configuration**](#service-configuration)
    - [**Uvicorn Service**](#uvicorn-service)
  - [**Main Federated Analytic Algorithms**](#main-federated-analytic-algorithms)
    - [**GWAS**](#gwas)
      - [**GWAS - BasicBfileQC (GWASConfig)**](#gwas---basicbfileqc-gwasconfig)
      - [**GWAS - LDPruning (GWASConfig)**](#gwas---ldpruning-gwasconfig)
      - [**GWAS - GenotypePCA (GWASConfig)**](#gwas---genotypepca-gwasconfig)
      - [**GWAS - CovariateStdz (GWASConfig)**](#gwas---covariatestdz-gwasconfig)
      - [**GWAS - QuantGWAS (GWASConfig)**](#gwas---quantgwas-gwasconfig)
      - [**GWAS - FullQuantGWAS (GWASConfig)**](#gwas---fullquantgwas-gwasconfig)
      - [**GWAS - BinGWAS (GWASConfig)**](#gwas---bingwas-gwasconfig)
      - [**GWAS - FullBinGWAS (GWASConfig)**](#gwas---fullbingwas-gwasconfig)
    - [**Linear Algebra**](#linear-algebra)
      - [**LinAlg - RandomizedSVD (SVDConfig)**](#linalg---randomizedsvd-svdconfig)
      - [**LinAlg - PCA (SVDConfig)**](#linalg---pca-svdconfig)
      - [**LinAlg - SVDfromTabular (TabularDataSVDConfig)**](#linalg---svdfromtabular-tabulardatasvdconfig)
      - [**LinAlg - PCAfromTabular (TabularDataSVDConfig)**](#linalg---pcafromtabular-tabulardatasvdconfig)
    - [**Survival**](#survival)
      - [**Survival - CoxPHRegression (CoxPHRegressionConfig)**](#survival---coxphregression-coxphregressionconfig)
      - [**Survival - KaplanMeier (KaplanMeierConfig)**](#survival---kaplanmeier-kaplanmeierconfig)
- [**References**](#references)


# **The Basic Usage of RAFAEL**

## **1. Clone the RAFAEL Repository**
```bash
git clone https://github.com/ailabstw/RAFAEL.git /rafael
cd rafael
```


## **2. Build the RAFAEL Service Docker Image**
```docker
docker build -t rafael .
```


## **3. Initialize a RAFAEL service**
The RAFAEL service automatically generates a service configuration in `services/configs` from the given environment variables and logs in `services/log`. To specify the service log path, use `-e SERVER_LOG_PATH=/your/log/path`, `-e COMPENSATOR_LOG_PATH=/your/log/path`, or `-e CLIENT_LOG_PATH=/your/log/path`. You can also specify the path to save output results in the request.

The uvicorn service will run on the `http://0.0.0.0:${UVICORN_PORT}` in the docker container. The container listen to the uvicorn service by using `docker run  ... -p ${UVICORN_PORT}:${DOCKER_PORT} ...`. See also https://docker-fastapi-projects.readthedocs.io/en/latest/uvicorn.html#troubleshoots.

### **3.0 Create the Local Docker Network**
```docker
docker network create --subnet=172.18.0.0/16 rafael-net
```
See also [docker network](https://docs.docker.com/reference/cli/docker/network/create/#specify-advanced-options)

### **3.1 Run a RAFAEL Server**
```docker
docker run -it --rm --name rafael-server --network rafael-net --ip 172.18.0.2 -p 8000:8000 -v ./services:/rafael/services rafael
```

### **3.2 Run a RAFAEL Compensator**
```docker
docker run -it --rm --name rafael-compensator --network rafael-net --ip 172.18.0.6 -p 8080:8080 -v ./services:/rafael/services -e ROLE=compensator -e PORT=8080 rafael
```

### **3.3 Run RAFAEL Clients**
Start the RAFAEL clients with assigned IDs:

client 1 ID: `f01b3208-11b8-446d-a178-39e18f16f89b`
```docker
docker run -it --rm --name rafael-client1 --network rafael-net --ip 172.18.0.3 -p 8001:8001 -v ./services:/rafael/services -e ROLE=client -e PORT=8001 -e CLIENT_NODE_ID=f01b3208-11b8-446d-a178-39e18f16f89b rafael
```

client 2 ID: `ab5f2e2a-3860-4c56-983d-cd16ea184098`
```docker
docker run -it --rm --name rafael-client2 --network rafael-net --ip 172.18.0.4 -p 8002:8002 -v ./services:/rafael/services -e ROLE=client -e PORT=8002 -e CLIENT_NODE_ID=ab5f2e2a-3860-4c56-983d-cd16ea184098 rafael
```

client 3 ID: `97bc8985-9ce5-481b-a0fb-8c5e6f872158`
```docker
docker run -it --rm --name rafael-client3 --network rafael-net --ip 172.18.0.5 -p 8003:8003 -v ./services:/rafael/services -e ROLE=client -e PORT=8003 -e CLIENT_NODE_ID=97bc8985-9ce5-481b-a0fb-8c5e6f872158 rafael
```

Note: Make sure the server and compensator are ready to be connected by client.


## **4. Run a Demo**
**It's recommended to terminate all services after completing a federated analysis.** The current in-memory data repository is a Python dictionary, which might cause parameter issues when conducting multiple analyses.

### **4.0 The introduction of analysis request**

### **Post an analysis request to server**
---

The base analysis request format in RAFAEL:
```jsonc
{
    "node_id": "${SERVER_NODE_ID}",
    "args": {
        "config": {
            // parameters for the analysis API
        }
    },
    "api": "${ANALYSIS_API}"
}
```
The `config` are the parameters for the analysis API.

For example, the available parameters in `CoxPHRegression` are:

- `clients`: The list of client IDs participating in the analysis.

- `feature_cols`: The feature columns to perform the analysis. Default is to use all features.

- `clinical_data_path`: The paths to the clinical data. The clinical data should contain columns named as **event** and **time** to perform survival analysis.

- `meta_cols`: The sample metadata to be excluded from the survival analysis.

- `save_dir`: The directory path to save the results.

- `r`: The number of samples in the global anchor matrix. Default is 100.

- `k`: The latent dimensions of the SVD. The decomposed matrix is used for creating proxy data matrix. Default is 20.

- `bs_prop`: The proportion of samples to be sampled for each bootstrap. Default is 0.6.

- `bs_times`: The number of bootstrap iterations. Default is 20.

- `alpha`: The statistical significance level. Default is 0.05.

- `step_size`: Deal with the fitting error, `delta contains nan value(s)`. Default is 0.5.

### **See also**
---
[Cox PH Regression analysis spec](https://github.com/ailabstw/RAFAEL/blob/1ed4a9e2a3beb3187cd72d2bfca46e46b8e4a711/rafael/datamodel.py#L312)

[DC-COX: Data collaboration Cox proportional hazards model for privacy-preserving survival analysis on multiple parties](https://www.sciencedirect.com/science/article/pii/S1532046422002696?via%3Dihub)


### **Example**
---
The following script is the example of `POST http://${SERVER_HOST}:${SERVER_PORT}/tasks`: 
```python
import requests

req = {
    "node_id": "7a5f34c4-4415-4b9a-bab7-ebbcdcc23a49",
    "args": {
        "config": {
            "clients": [
                "f01b3208-11b8-446d-a178-39e18f16f89b",
                "ab5f2e2a-3860-4c56-983d-cd16ea184098",
                "97bc8985-9ce5-481b-a0fb-8c5e6f872158"
            ],
            "clinical_data_path":[
                "data/client1/GSE62564-1.csv",
                "data/client2/GSE62564-2.csv",
                "data/client3/GSE62564-3.csv"
            ],
            "save_dir":[
                "/rafael/services/results/f01b3208-11b8-446d-a178-39e18f16f89b",
                "/rafael/services/results/ab5f2e2a-3860-4c56-983d-cd16ea184098",
                "/rafael/services/results/97bc8985-9ce5-481b-a0fb-8c5e6f872158"
            ],
            "meta_cols":[
                "sample-id"
            ]
        }
    },
    "api": "CoxPHRegression"
}

requests.post("http://localhost:8000/tasks", json=req)
```
Refer to `rafael/datamodel.py` for parameter specifications of other analysis APIs.

### **4.1 Genome Wide Association Study (GWAS)**
Dataset: The two clients' demo data hapmap1 is stored in `data/`.

`POST` http://localhost:8000/tasks

Quantitative trait:
```json
{
    "node_id": "7a5f34c4-4415-4b9a-bab7-ebbcdcc23a49",
    "args": {
        "config": {
            "clients": [
                "f01b3208-11b8-446d-a178-39e18f16f89b",
                "ab5f2e2a-3860-4c56-983d-cd16ea184098"
            ],
            "compensators": [
                "8c66f4e8-9d4c-446d-9e6c-cbdf8b285554"
            ],
            "bfile_path": [
                "data/client1/hapmap1_100_1",
                "data/client2/hapmap1_100_2"
            ],
            "cov_path":[
                "data/client1/hapmap1_100_1.cov",
                "data/client2/hapmap1_100_2.cov"
            ],
            "regression_save_dir": [
                "/rafael/services/results/f01b3208-11b8-446d-a178-39e18f16f89b",
                "/rafael/services/results/ab5f2e2a-3860-4c56-983d-cd16ea184098"
            ],
            "local_qc_output_path": [
                "/rafael/services/results/f01b3208-11b8-446d-a178-39e18f16f89b/qc",
                "/rafael/services/results/ab5f2e2a-3860-4c56-983d-cd16ea184098/qc"
            ],
            "global_qc_output_path": "/rafael/services/results/7a5f34c4-4415-4b9a-bab7-ebbcdcc23a49/qc",
            "snp_chunk_size": 10,
            "maf": 0.05,
            "geno": 0.05,
            "mind": 0.05
        }
    },
    "api": "FullQuantGWAS"
}
```

Binary trait:
Assign binary phenotype data with `pheno_path`.
```json
{
    "node_id": "7a5f34c4-4415-4b9a-bab7-ebbcdcc23a49",
    "args": {
        "config": {
            "clients": [
                "f01b3208-11b8-446d-a178-39e18f16f89b",
                "ab5f2e2a-3860-4c56-983d-cd16ea184098"
            ],
            "compensators": [
                "8c66f4e8-9d4c-446d-9e6c-cbdf8b285554"
            ],
            "bfile_path": [
                "data/client1/hapmap1_100_1",
                "data/client2/hapmap1_100_2"
            ],
            "cov_path":[
                "data/client1/hapmap1_100_1.cov",
                "data/client2/hapmap1_100_2.cov"
            ],
            "pheno_path": [
                "data/client1/hapmap1_100_1.pheno",
                "data/client2/hapmap1_100_2.pheno"
            ],
            "regression_save_dir": [
                "/rafael/services/results/f01b3208-11b8-446d-a178-39e18f16f89b",
                "/rafael/services/results/ab5f2e2a-3860-4c56-983d-cd16ea184098"
            ],
            "local_qc_output_path": [
                "/rafael/services/results/f01b3208-11b8-446d-a178-39e18f16f89b/qc",
                "/rafael/services/results/ab5f2e2a-3860-4c56-983d-cd16ea184098/qc"
            ],
            "global_qc_output_path": "/rafael/services/results/7a5f34c4-4415-4b9a-bab7-ebbcdcc23a49/qc",
            "snp_chunk_size": 10,
            "maf": 0.05,
            "geno": 0.05,
            "mind": 0.05
        }
    },
    "api": "FullBinGWAS"
}
```

### **4.2 Principal Component Analysis (PCA)**
Dataset: The three clients' demo data GSE62564 are stored in `data/`.

`POST` http://localhost:8000/tasks
```json
{
    "node_id": "7a5f34c4-4415-4b9a-bab7-ebbcdcc23a49",
    "args": {
        "config": {
            "clients": [
                "f01b3208-11b8-446d-a178-39e18f16f89b",
                "ab5f2e2a-3860-4c56-983d-cd16ea184098",
                "97bc8985-9ce5-481b-a0fb-8c5e6f872158"
            ],
            "file_path":[
                "data/client1/GSE62564-1.csv",
                "data/client2/GSE62564-2.csv",
                "data/client3/GSE62564-3.csv"
            ],
            "svd_save_dir":[
                "/rafael/services/results/f01b3208-11b8-446d-a178-39e18f16f89b/pca",
                "/rafael/services/results/ab5f2e2a-3860-4c56-983d-cd16ea184098/pca",
                "/rafael/services/results/97bc8985-9ce5-481b-a0fb-8c5e6f872158/pca"
            ],
            "to_pc": true,
            "meta_cols":[
                "sample-id",
                "time",
                "event"
            ]
        }
    },
    "api": "PCAfromTabular"
}
```

### **4.3 Singular Value Decomposition (SVD)**
Dataset: The three clients' demo data GSE62564 are stored in `data/`.

`POST` http://localhost:8000/tasks
```json
{
    "node_id": "7a5f34c4-4415-4b9a-bab7-ebbcdcc23a49",
    "args": {
        "config": {
            "clients": [
                "f01b3208-11b8-446d-a178-39e18f16f89b",
                "ab5f2e2a-3860-4c56-983d-cd16ea184098",
                "97bc8985-9ce5-481b-a0fb-8c5e6f872158"
            ],
            "file_path":[
                "data/client1/GSE62564-1.csv",
                "data/client2/GSE62564-2.csv",
                "data/client3/GSE62564-3.csv"
            ],
            "svd_save_dir":[
                "/rafael/services/results/f01b3208-11b8-446d-a178-39e18f16f89b/svd",
                "/rafael/services/results/ab5f2e2a-3860-4c56-983d-cd16ea184098/svd",
                "/rafael/services/results/97bc8985-9ce5-481b-a0fb-8c5e6f872158/svd"
            ],
            "to_pc": false,
            "meta_cols":[
                "sample-id",
                "time",
                "event"
            ]
        }
    },
    "api": "SVDfromTabular"
}
```

### **4.4 Cox Proportion Hazard Regression**
Dataset: The three clients' demo data GSE62564 are stored in `data/`.

`POST` http://localhost:8000/tasks
```json
{
    "node_id": "7a5f34c4-4415-4b9a-bab7-ebbcdcc23a49",
    "args": {
        "config": {
            "clients": [
                "f01b3208-11b8-446d-a178-39e18f16f89b",
                "ab5f2e2a-3860-4c56-983d-cd16ea184098",
                "97bc8985-9ce5-481b-a0fb-8c5e6f872158"
            ],
            "clinical_data_path":[
                "data/client1/GSE62564-1.csv",
                "data/client2/GSE62564-2.csv",
                "data/client3/GSE62564-3.csv"
            ],
            "save_dir":[
                "/rafael/services/results/f01b3208-11b8-446d-a178-39e18f16f89b/cox",
                "/rafael/services/results/ab5f2e2a-3860-4c56-983d-cd16ea184098/cox",
                "/rafael/services/results/97bc8985-9ce5-481b-a0fb-8c5e6f872158/cox"
            ],
            "meta_cols":[
                "sample-id"
            ]
        }
    },
    "api": "CoxPHRegression"
}
```

### **4.5 Kaplan-Meier**
Dataset: The three clients' demo data GSE62564 are stored in `data/`.

`POST` http://localhost:8000/tasks
```json
{
    "node_id": "7a5f34c4-4415-4b9a-bab7-ebbcdcc23a49",
    "args": {
        "config": {
            "clients": [
                "f01b3208-11b8-446d-a178-39e18f16f89b",
                "ab5f2e2a-3860-4c56-983d-cd16ea184098",
                "97bc8985-9ce5-481b-a0fb-8c5e6f872158"
            ],
            "clinical_data_path":[
                "data/client1/GSE62564-1.csv",
                "data/client2/GSE62564-2.csv",
                "data/client3/GSE62564-3.csv"
            ],
            "save_dir":[
                "/rafael/services/results/f01b3208-11b8-446d-a178-39e18f16f89b/km",
                "/rafael/services/results/ab5f2e2a-3860-4c56-983d-cd16ea184098/km",
                "/rafael/services/results/97bc8985-9ce5-481b-a0fb-8c5e6f872158/km"
            ],
            "meta_cols":[
                "sample-id"
            ]
        }
    },
    "api": "KaplanMeier"
}
```

## **5. Run a Federated Analysis on Your Own Dataset**


Similar to the [3.3 Run RAFAEL Clients](#33-run-rafael-clients) with an additional parameter `-v` to mount your own data:

```docker
docker run -it --rm --name ${CLIENT_NAME} --network rafael-net --ip ${CLIENT_HOST} -p ${CLIENT_PORT}:${CLIENT_PORT} -v ./services:/rafael/services -v /path/to/data/directory:/rafael/data -e ROLE=client -e PORT=${CLIENT_PORT} -e CLIENT_NODE_ID=${CLIENT_NODE_ID} rafael
```
In fact, the `/rafael/data` is a recommended path to mount, not necessarily. The input file path of the analysis can be specified in the request.

### **Example:**
Initialize services
```docker
> Initialize a server
> Initialize a compensator

# Client1
docker run -it --rm --name rafael-client1 --network rafael-net --ip 172.18.0.3 -p 8001:8001 -v ./services:/rafael/services -v ~/Desktop/client1_data:/mnt -e ROLE=client -e PORT=8001 -e CLIENT_NODE_ID=f01b3208-11b8-446d-a178-39e18f16f89b rafael

# Client2
docker run -it --rm --name rafael-client2 --network rafael-net --ip 172.18.0.4 -p 8002:8002 -v ./services:/rafael/services -v ~/Desktop/client2_data:/mnt -e ROLE=client -e PORT=8002 -e CLIENT_NODE_ID=ab5f2e2a-3860-4c56-983d-cd16ea184098 rafael
```

`POST` http://localhost:8000/tasks
```json
{
    "node_id": "7a5f34c4-4415-4b9a-bab7-ebbcdcc23a49",
    "args": {
        "config": {
            "clients": [
                "f01b3208-11b8-446d-a178-39e18f16f89b",
                "ab5f2e2a-3860-4c56-983d-cd16ea184098"
            ],
            "compensators": [
                "8c66f4e8-9d4c-446d-9e6c-cbdf8b285554"
            ],
            "bfile_path": [
                "/mnt/hapmap1_100_1",
                "/mnt/hapmap1_100_2"
            ],
            "cov_path":[
                "/mnt/hapmap1_100_1.cov",
                "/mnt/hapmap1_100_2.cov"
            ],
            "regression_save_dir": [
                "/rafael/services/results/f01b3208-11b8-446d-a178-39e18f16f89b",
                "/rafael/services/results/ab5f2e2a-3860-4c56-983d-cd16ea184098"
            ],
            "local_qc_output_path": [
                "/rafael/services/results/f01b3208-11b8-446d-a178-39e18f16f89b/qc",
                "/rafael/services/results/ab5f2e2a-3860-4c56-983d-cd16ea184098/qc"
            ],
            "global_qc_output_path": "/rafael/services/results/7a5f34c4-4415-4b9a-bab7-ebbcdcc23a49/qc",
            "snp_chunk_size": 10,
            "maf": 0.05,
            "geno": 0.05,
            "mind": 0.05
        }
    },
    "api": "FullQuantGWAS"
}
```


# **The Advanced Usage of RAFAEL**
Make sure you have been familiar with how to run a RAFAEL service in the [previous section](#the-basic-usage-of-rafael).  
This section is to provide the hyperparameters for the customized service configurations and more comprehensive analysis.

## **Environment Variables to Set Up Service**
The following environment variables can be assigned like so:
```docker
docker run --rm -it ... -e ROLE=${ROLE} -e PORT=${PORT} ... rafael
```

### **Service Configuration**
These variables control the service role, who to connect, its identity, where to save log, and the network protocol.
- `ROLE`: The service to be created as `ROLE`. Available parameters are `server`, `compensator` and `client`. Default is `server`.

- `SERVER_NODE_ID`: The server node ID. Default is `7a5f34c4-4415-4b9a-bab7-ebbcdcc23a49`.

- `COMPENSATOR_NODE_ID`: The compressor node ID. Default is `8c66f4e8-9d4c-446d-9e6c-cbdf8b285554`.

- `CLIENT_NODE_ID`: The client node ID. Default is randomly generated UUID4.

- `SERVER_LOG_PATH`: The path to save the server log. Default is `/rafael/services/log/server.log`.

- `COMPENSATOR_LOG_PATH`: The path to save the compressor log. Default is `/rafael/services/log/compensator.log`.

- `CLIENT_LOG_PATH`: The path to save the client log. Default is `/rafael/services/log/client-${CLIENT_NODE_ID}.log`

- `PROTOCOL`: The web service protocol to contruct the address as `${PROTOCOL}://${SERVER_HOST}:${SERVER_PORT}` or `${PROTOCOL}://${COMPENSATOR_HOST}:${COMPENSATOR_PORT}`. Default is `ws`.

- `SERVER_HOST`: The sever host to connect to. Default is `172.18.0.2`.

- `SERVER_PORT`: The server port to connect to. Default is `8000`.

- `COMPENSATOR_HOST`: The compensator host to connect to. Default is `172.18.0.6`.

- `COMPENSATOR_PORT`: The compensator port to connect to. Default is `8080`.

### **Uvicorn Service**
These variables are the uvicorn parameters to initialize a service. See also [uvicorn options](https://www.uvicorn.org/#command-line-options).
- `PORT`: The uvicorn service in the container running at `PORT`. Default is `8000`.

- `PING_INTERVAL`: The uvicorn websocket implementation parameter `ws_ping_interval`. Default is set to `600` due to some time-consuming calculations.  

- `PING_TIMEOUT`: The uvicorn websocket implementation parameter `ws_ping_timeout`. Default is set to `300` due to some time-consuming calculations.

- `MAX_MESSAGE_SIZE`: The uvicorn websocket implementation parameter `ws_max_size`. Default is set to `1e20` due to tremendous memory consumption in GWAS.


## **Main Federated Analytic Algorithms**
### **GWAS**
The base parameters for performing federated GWAS are:
- `bfile_path`: The path to bfile with prefix. For example, client1.bed, client1.fam and client1.bim are under `~/data/`, the `bfile_path` should be `~/data/client1`.

- `cov_path`: The path to the covariate file. Default is None.

- `pheno_path`: The path to the phenotype file. Default is None.

- `pheno_name`: The column name of the phenotype in phenotype file.

- `snp_chunk_size`: The number of SNPs to be calculated in a single chunk. Default is 10000.

- `regression_save_dir`: The path to save the results, `gwas.glm`, `gwas.manhattan.png` and `gwas.qq.png`.

The recommended combinations for the federated GWAS (means POST to these APIs step by step):
- `QunatGWAS` for qunatitative trait.

- `BinGWAS` for binary trait.

- `BasicBfileQC` &rarr; `QunatGWAS` for the additional quality control.

- `BasicBfileQC` &rarr; `BinGWAS` for the additional quality control.

- `FullQuantGWAS` for the complete GWAS, including the LD-pruning and PCA.

- `FullBinGWAS` for the complete GWAS, including the LD-pruning and PCA.

#### **GWAS - BasicBfileQC ([GWASConfig](https://github.com/ailabstw/RAFAEL/blob/1ed4a9e2a3beb3187cd72d2bfca46e46b8e4a711/rafael/datamodel.py#L224))**
- `maf`: Filter out all variants with minor allele frequency below the given threshold. Default is 0.05.

- `geno`: Filter out all SNPs with missing call rates exceeding the given value. Default is 0.02.

- `hwe`: Filter out all SNPs having Hardy-Weinberg equilibrium test p-value below the given threshold. Default is 5e-7.

- `mind`: Filters out all samples with missing call rates exceeding the given value. Default is 0.02.

- `local_qc_output_path`: The path to the output file for the QC report in client.

- `global_qc_output_path`: The path to the output file for the QC report in server.

#### **GWAS - LDPruning ([GWASConfig](https://github.com/ailabstw/RAFAEL/blob/1ed4a9e2a3beb3187cd72d2bfca46e46b8e4a711/rafael/datamodel.py#L224))**
- `prune_method`: The method to deal with the remained SNPs from clients after pruning at each client side. Default is "intersect". Available options are "intersect" and "union".

- `win_size`: Window size in variant count. Default is 50.

- `step`: Variant count to shift the window at the end of each step. Default is 5.

- `r2`: Variants whose $r^2$ is greater than given threshold were removed. Default is 0.2.

#### **GWAS - GenotypePCA ([GWASConfig](https://github.com/ailabstw/RAFAEL/blob/1ed4a9e2a3beb3187cd72d2bfca46e46b8e4a711/rafael/datamodel.py#L224))**
- Same in [RandomizedSVD](#linalg---randomizedsvd), but the default of `svd_save_dir` is set to `local_qc_output_path`.

#### **GWAS - CovariateStdz ([GWASConfig](https://github.com/ailabstw/RAFAEL/blob/1ed4a9e2a3beb3187cd72d2bfca46e46b8e4a711/rafael/datamodel.py#L224))**
- No hyperparameters

#### **GWAS - QuantGWAS ([GWASConfig](https://github.com/ailabstw/RAFAEL/blob/1ed4a9e2a3beb3187cd72d2bfca46e46b8e4a711/rafael/datamodel.py#L224))**
- `block_size`: Number of SNPs to run in a block in a process. Defaults is 10000.

- `num_core`: The number of cores to perform parallel computation. Default is 4.

#### **GWAS - FullQuantGWAS ([GWASConfig](https://github.com/ailabstw/RAFAEL/blob/1ed4a9e2a3beb3187cd72d2bfca46e46b8e4a711/rafael/datamodel.py#L224))**
- The union of `BasicBfileQC`, `LDPruning`, `GenotypePCA` and `QuantGWAS`.

#### **GWAS - BinGWAS ([GWASConfig](https://github.com/ailabstw/RAFAEL/blob/1ed4a9e2a3beb3187cd72d2bfca46e46b8e4a711/rafael/datamodel.py#L224))**
- `logistic_max_iters`: The maximum number of iterations in logistic regression.

#### **GWAS - FullBinGWAS ([GWASConfig](https://github.com/ailabstw/RAFAEL/blob/1ed4a9e2a3beb3187cd72d2bfca46e46b8e4a711/rafael/datamodel.py#L224))**
- The union of `BasicBfileQC`, `LDPruning`, `GenotypePCA` and `BinGWAS`.

### **Linear Algebra**
The federated linear algebra in RAFAEL currently supports PCA and SVD, the former can leverage the latter and the APIs supporting the federated standardization to achieve. Hence, the PCA shares the same parameters as the SVD. The tabular data reading APIs are shared as well.

#### **LinAlg - RandomizedSVD ([SVDConfig](https://github.com/ailabstw/RAFAEL/blob/73ae9569aeec9b5bfbea898f30a4bd057dd2a1f1/rafael/datamodel.py#L332))**
This API cannot be directly used. It requires other APIs to prepare the variable `A` and save it to the data repository.
- `k1`: The initial number of latent dimensions. Default is 20.

- `k2`: The output number of latent dimensions. Default is 20.

- `svd_max_iters`: The maximum number of iterations to update the eigenvectors. Default is 20.

- `epsilon`: The tolerance of the convergence. Default is 1e-9.

- `first_n`: The first n latent dimensions to share globally. Default is 4. It is noted that this parameter and its corresponding APIs should be removed when considering a rigorus federatad scenario.

- `to_pc`: The outputs are eigenvecots ($U$ and $V$) or PCs ($U\Sigma$ and $V\Sigma$). Default is False.

- `label`: The data in output figures are colored by what label. Default is None. 

- `svd_save_dir`: The directory to save the eigenvectors.

#### **LinAlg - PCA ([SVDConfig](https://github.com/ailabstw/RAFAEL/blob/73ae9569aeec9b5bfbea898f30a4bd057dd2a1f1/rafael/datamodel.py#L332))**
This API cannot be directly used. It requires other APIs to prepare the variable `A` and save it to the data repository.
- Same in [RandomizedSVD](#linalg---randomizedsvd).

#### **LinAlg - SVDfromTabular ([TabularDataSVDConfig](https://github.com/ailabstw/RAFAEL/blob/73ae9569aeec9b5bfbea898f30a4bd057dd2a1f1/rafael/datamodel.py#L370))**
- Inherited from [RandomizedSVD](#linalg---randomizedsvd).

- `file_path`: The path to the data. 

- `meta_cols`: The metadata column names, which are used for coloring the data in output figures and excluding the unwanted data to participate in SVD/PCA. Default is None, meaning no unwanted data.

- `drop_cols`: The column names to be dropped. Different from `meta_cols`, `drop_cols` are not not used in any downstream task, while the `meta_cols` may be used for labeling. When `drop_cols` is given, the `keep_cols` shouldn't be used. Default is None.

- `keep_cols`: The column names to be kept to perform SVD/PCA. When `keep_cols` is given, the `drop_cols` shouldn't be used. Default is None, meaning to use all feature in the provided data.

#### **LinAlg - PCAfromTabular ([TabularDataSVDConfig](https://github.com/ailabstw/RAFAEL/blob/73ae9569aeec9b5bfbea898f30a4bd057dd2a1f1/rafael/datamodel.py#L370))**
- Same in [SVDfromTabular](#linalg---svdfromtabular).

### **Survival**
The Cox PH Regression is the implementation of [DC-COX](https://www.sciencedirect.com/science/article/pii/S1532046422002696?via%3Dihub), which secures the data in a methematical way. The current Kaplan-Meier survival analysis in RAFAEL only supports continuous data. It leverages the federated standardization APIs to divide samples into two groups and perform Kaplan-Meier survival analysis respectively.

CoxPHRegression and KaplanMeier share the same base parameters:
- `clinical_data_path`: The paths to the clinical data. The clinical data should contain columns named as **event** and **time** to perform survival analysis.

- `feature_cols`: The columns to perform the survival analysis.  Default is to use all features.

- `meta_cols`: The sample metadata to be excluded from the survival analysis. 

- `save_dir`: The directory path to save the results.

#### **Survival - CoxPHRegression ([CoxPHRegressionConfig](https://github.com/ailabstw/RAFAEL/blob/73ae9569aeec9b5bfbea898f30a4bd057dd2a1f1/rafael/datamodel.py#L318C7-L318C28))**
- `r`: The number of samples in the global anchor matrix. Default is 100.

- `k`: The latent dimensions of the SVD. The decomposed matrix is used for creating proxy data matrix. Default is 20.

- `bs_prop`: The proportion of samples to be sampled for each bootstrap. Default is 0.6.

- `bs_times`: The number of bootstrap iterations. Default is 20.

- `alpha`: The statistical significance level. Default is 0.05.

- `step_size`: Deal with the fitting error, `delta contains nan value(s)`. Default is 0.5.

#### **Survival - KaplanMeier ([KaplanMeierConfig](https://github.com/ailabstw/RAFAEL/blob/73ae9569aeec9b5bfbea898f30a4bd057dd2a1f1/rafael/datamodel.py#L327C7-L327C24))**
- `alpha`: The statistical significance level. Default is 0.05.
- `n_std`: Regard `n_std` as $k$. Separate samples into $\geq k*\sigma$ and $\leq-k*\sigma$, but with the standardization, $\sigma=1$, so we can simplify as $\geq k$ and $\leq-k$. Default is 1.


# **References**
- Federated GWAS Regression & Mechanism: [sPLINK: a hybrid federated tool as a robust alternative to meta-analysis in genome-wide association studies](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-021-02562-1)
- Federated SVD: [Federated horizontally partitioned principal component analysis for biomedical applications](https://academic.oup.com/bioinformaticsadvances/article/2/1/vbac026/6574370?login=false)
- Federated Cox PH Regression: [DC-COX: Data collaboration Cox proportional hazards model for privacy-preserving survival analysis on multiple parties](https://www.sciencedirect.com/science/article/pii/S1532046422002696?via%3Dihub)
- PLINK2: [Second-generation PLINK: rising to the challenge of larger and richer datasets](https://academic.oup.com/gigascience/article/4/1/s13742-015-0047-8/2707533?login=false)