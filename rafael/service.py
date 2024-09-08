import os
from typing import Any
import asyncio
import time
import logging
import zipfile
import pickle

from tqdm import tqdm
import numpy as np
import pandas as pd
import jax.numpy as jnp
from fastapi import APIRouter, WebSocket
from fastapi.responses import FileResponse
from aiohttp.web import WebSocketResponse

from rafael.privacy import hyfed
from rafael.fedalgo.gwasprs.gwasdata import GWASDataIterator
from rafael.fedalgo.gwasprs.reader import PhenotypeReader
from .controller import AbstractController
from .data_repository.repository import AbstractRepository, InMemoryRepository
from .datamodel import send_model, recv_model, construct_argument
from . import datamodel, usecases
from . import authentication as auth
from . import network as net
from . import utils


class ServiceController(AbstractController):

    def __init__(self, config_path: str, repo: AbstractRepository = InMemoryRepository()) -> None:
        super().__init__(config_path)
        self.__repo = repo
        self._auth = auth.NodeIdentity(self.config["config"]["node_id"])
        self.router = APIRouter()
        self.router.add_api_route(self.config["config"].get("subpath", "") + "/tasks", self.handle_request, methods=["POST"])
        self.router.add_websocket_route(self.config["config"].get("subpath", "") + "/ws/tasks", self.handle_ws)
        logging.info(f"Service node id: {self._auth.node_id}")
        
        # Use Cases
        self.qc = usecases.BasicBfileQC()
        self.ld = usecases.NaiveLDPruning()
        self.cov = usecases.CovarProcessor()
        self.quant_gwas = usecases.QuantitativeGWAS()
        self.binary_gwas = usecases.BinaryGWAS()
        self.stdz = usecases.Standadization()
        self.svd = usecases.RandomizedSVD()
        self.gso = usecases.GramSchmidt()
        self.surv_cox = usecases.CoxPHRegression()
        self.km = usecases.KaplanMeier()
        self.table_reader = usecases.TabularReader()
        self.output = usecases.Output()

    @property
    def repo(self):
        return self.__repo

    async def eval(self, api, args):
        api = getattr(self, api)
        arg = construct_argument(api, args)
        if arg is None:
            result = await api()
        else:
            result = await api(**arg)
        return result

    async def handle_request(self, request: datamodel.HTTPRequest) -> Any:
        raise NotImplementedError

    async def handle_ws(self, ws: WebSocket) -> Any:
        raise NotImplementedError


class ServerService(ServiceController):

    def __init__(self, config_path) -> None:
        super().__init__(config_path)
        self.router.add_api_route(self.config["config"].get("subpath", "") + "/result/{plan_id}/{task_id}", self.handle_result, methods=["GET"])
        self.__client_ws = net.FastAPIWSConnectionManager()
        self.__compensator_ws = net.FastAPIWSConnectionManager()
        self.timeout = self.config["config"]["timeout"] if "timeout" in self.config["config"].keys() else None

    @property
    def clients(self):
        return list(self.__clients_ws.active_connections.keys())
    
    @property
    def compensators(self):
        return list(self.__compensator_ws.active_connections.keys())

    async def handle_request(self, request: datamodel.HTTPRequest) -> Any:
        self._auth.authenticate(request.node_id)
        return await self.eval(request.api, request.args)
    
    async def handle_result(self, plan_id: str, task_id: str, request: datamodel.FileRequest) -> Any:
        self._auth.authenticate(request.node_id)
        return await self.get_zipped_result(plan_id, task_id, request.clients, request.files)

    async def handle_ws(self, ws: WebSocket) -> Any:
        # Wait for connection
        await ws.accept()

        # Wait for registry
        data = pickle.loads(await ws.receive_bytes())
        registery = datamodel.Registery(**data)

        # authentication
        self.logger.info(f"Registered from {registery.role}:{registery.node_id}")
        if registery.role == "client":
            self.__client_ws.add(registery.node_id, ws)
        elif registery.role == "compensator":
            self.__compensator_ws.add(registery.node_id, ws)
        else:
            raise Exception(f"{registery.role} not found")

        try:
            # This is essential to hold this handle_ws function call forever asynchronously,
            # otherwise existing this function makes websocket lost its connection
            while True:
                await asyncio.Future()  # run forever
        except asyncio.CancelledError:
            pass
    
    def _create_role_task(self, ws: net.FastAPIWSConnectionManager, role: str, api: str, request: dict):
        return asyncio.ensure_future(net.rpc_ws(ws.get(role), api, request, timeout=self.timeout))
    
    def _union_models(self, *models):
        models = list(map(lambda data: recv_model(**data), models))
        
        union_model = {}
        
        args = models[0].model_fields.keys()
        
        for arg in args:
            union_model[arg] = ('List', list(map(lambda x: getattr(x, arg), models)))
        
        union_model = send_model(**union_model)
        
        return union_model
    
    def _aggregate_noise(self, *noises):
        noises = list(map(lambda x: recv_model(**x), noises))

        aggregated_noise = {}

        args = noises[0].model_fields.keys()
        
        for arg in args:
            agg_noise = np.sum(list(map(lambda x: getattr(x, arg), noises)), axis=0)
            aggregated_noise[arg] = ('NPArray', agg_noise)
        
        aggregated_noise = send_model(**aggregated_noise)

        return aggregated_noise
    
    async def request_clients(self, api, config, **kwargs):
        if isinstance(config.clients, str):
            config.clients = [config.clients]

        tasks = []
        for i in range(len(config.clients)):
            client_ = config.clients[i]
            kwargs_ = send_model(**kwargs).model_dump()
            kwargs_.pop('spec')
            request = {'config': config.to_client_config(i).model_dump(), **kwargs_}
            tasks.append(self._create_role_task(self.__client_ws, client_, api, request))

        results = await asyncio.gather(*tasks)

        return self._union_models(*results)
        
    async def request_compensators(self, config, *args, cc_map=None):
        # TODO: Test multiple compensators
        # TODO: Considering where to place cc_map
        if isinstance(config.clients, str):
            config.clients = [config.clients]
        
        if isinstance(config.compensators, str):
            config.compensators = [config.compensators]
        
        if cc_map is None:
            # Group clients to make a single compensator collect noises from assigned clients,
            grouped_clients = np.array_split(config.clients, len(config.compensators))
            cc_map = {config.compensators[i]:list(grouped_clients[i]) for i in range(len(config.compensators))}

        tasks = []
        for i in range(len(config.compensators)):
            compensator_ = config.compensators[i]
            request = {'request':datamodel.CompensatorRequest(clients=cc_map[compensator_], args=[*args]).model_dump()}
            tasks.append(self._create_role_task(self.__compensator_ws, compensator_, 'collect_noise', request))

        results = await asyncio.gather(*tasks)

        return self._aggregate_noise(*results)
    
    async def get_zipped_result(self, plan_id: str, task_id: str, clients: list, files: list) -> Any:
        tasks = []
        for i in range(len(clients)):
            tasks.append(self._create_role_task(self.__client_ws, clients[i], 'compress_result', {'files':files}))
        results = await asyncio.gather(*tasks)
                    
        with zipfile.ZipFile('result.zip', mode='w') as zf:
            
            for i in range(len(clients)):
                data = recv_model(**results[i])
                content = bytes(data.content)
                
                save_tree = f'/result/{plan_id}/{task_id}/{clients[i]}'
                for filename, content in utils.extract_from_zipbytes(content).items():
                    zf.writestr(f'{save_tree}/{filename}', content)
                    
        return FileResponse('result.zip')

    async def gwas_match_snps(self, config: datamodel.GWASConfig):
        results = await self.request_clients('gwas_get_metadata', config)

        t = time.perf_counter_ns()
        autosome_snp_list = self.qc.global_match_snps(results.autosome_snp_list)
        self.logger.info(f"Server - global_match_snps: {time.perf_counter_ns() - t} ns")

        self.repo.add("autosome_snp_list", autosome_snp_list)
        return datamodel.Status(status="OK")

    async def gwas_qc_stats(self, config: datamodel.GWASConfig):
        autosome_snp_list = self.repo.get("autosome_snp_list")
        
        results = await self.request_clients('gwas_qc_stats', config, autosome_snp_list=('list', autosome_snp_list))
        
        t = time.perf_counter_ns()

        autosome_snp_list = self.qc.global_qc_stats(
            results.allele_count, results.n_obs, self.repo.get("autosome_snp_list"),
            config.global_qc_output_path,
            config.geno, config.hwe, config.maf
        )
        self.logger.info(f"Server - global_qc_stats: {time.perf_counter_ns() - t} ns")

        self.repo.add("autosome_snp_list", autosome_snp_list)
        return datamodel.Status(status="OK")

    async def gwas_filter_bfile(self, config: datamodel.GWASConfig):
        autosome_snp_list = self.repo.get("autosome_snp_list", [])
        await self.request_clients('gwas_filter_bfile', config, autosome_snp_list=('list', autosome_snp_list))
        return datamodel.Status(status="OK")
    
    async def gwas_prune_ld(self, config: datamodel.GWASConfig):
        results = await self.request_clients('gwas_local_prune_ld', config)
        
        remained_snps = self.ld.global_match_snps(results.remained_snps, method=config.prune_method)
        
        self.repo.add("ld_remained_snps", remained_snps)
        return datamodel.Status(status="OK")
    
    async def gwas_drop_ld_snps(self, config: datamodel.GWASConfig):
        await self.request_clients('gwas_drop_ld_snps', config, remained_snps=('NPArray', self.repo.pop("ld_remained_snps")))
        
        return datamodel.Status(status="OK")
    
    async def gwas_covariate_stdz_init(self, config: datamodel.GWASConfig):
        results = await self.request_clients('gwas_covariate_stdz_init', config)

        if any(s == "BREAK" for s in results.status):
            return datamodel.Status(status="BREAK")
        else:
            return datamodel.Status(status="OK")
    
    async def gwas_update_covariate(self, config: datamodel.GWASConfig):
        await self.request_clients('gwas_local_update_covariate', config)

        return datamodel.Status(status="OK")

    async def gwas_fit_linear_model(self, config: datamodel.GWASConfig):
        results = await self.request_clients('gwas_calculate_covariance', config)

        if not isinstance(results.XtX[0], list):
            t = time.perf_counter_ns()
            # Get the noise from compensator 
            noise = await self.request_compensators(config, 'XtX', 'Xty')
            
            # Denoise
            XtX = np.sum(results.XtX, axis=0) - noise.XtX
            Xty = np.sum(results.Xty, axis=0) - noise.Xty

            beta = self.quant_gwas.global_fit_model(XtX, Xty)

            self.logger.info(f"Server - global_fit_model: {time.perf_counter_ns() - t} ns")
            self.repo.add("linear_reg_beta", beta)
            return datamodel.Status(status="OK")
        else:
            return datamodel.Status(status="END")

    async def gwas_linear_stats(self, config: datamodel.GWASConfig):
        results = await self.request_clients('gwas_calculate_sse_and_obs', config, beta=('JNPArray', self.repo.get("linear_reg_beta")))

        t = time.perf_counter_ns()

        # Get the noise from compensator 
        noise = await self.request_compensators(config, 'sse')
        
        # Denoise
        sse = np.sum(results.sse, axis=0) - noise.sse
        n_obs = np.sum(results.n_obs, axis=0)
        
        t_stat, pval = self.quant_gwas.global_stats(sse, n_obs)

        self.logger.info(f"Server - t_stats: {time.perf_counter_ns() - t} ns")

        self.repo.add("t_stat", t_stat)
        self.repo.add("pval", pval)
        self.repo.add("n_obs", sum(results.n_obs))
        return datamodel.Status(status="OK")

    async def gwas_write_glm(self, config: datamodel.GWASConfig):
        await self.request_clients(
            "gwas_write_glm",
            config, 
            t_stat=('NPArray', self.repo.get("t_stat")),
            pval=('NPArray', self.repo.get("pval")),
            n_obs=('NPArray', self.repo.get("n_obs")),
        )
        return datamodel.Status(status="OK")

    async def gwas_logistic_init(self, config: datamodel.GWASConfig):
        results = await self.request_clients('gwas_logistic_init', config)

        if results.current_iteration[0] != -1:
            # Get the noise from compensator
            noise = await self.request_compensators(config, 'gradient', 'hessian', 'loglikelihood')

            # Denoise
            gradient = np.sum(results.gradient, axis=0) - noise.gradient
            hessian = np.sum(results.hessian, axis=0) - noise.hessian
            loglikelihood = np.sum(results.loglikelihood, axis=0) - noise.loglikelihood

            beta, prev_beta, prev_loglikelihood, _, _ = self.binary_gwas.global_params(
                jnp.array(gradient), jnp.array(hessian), jnp.array(loglikelihood), 
                results.current_iteration[0], config.logistic_max_iters,
                None, None
            )
            
            # beta, prev_beta, prev_loglikelihood, _, _ = self.binary_gwas.global_params(
            #     results.gradient, results.hessian, results.loglikelihood, 
            #     results.current_iteration[0], config.max_iterations,
            #     None, None
            # )
            
            self.repo.add("logistic_reg_beta", beta)
            self.repo.add("prev_beta", prev_beta)
            self.repo.add("prev_loglikelihood", prev_loglikelihood)
            self.repo.add("n_obs", sum(results.n_obs))
            
            return datamodel.Status(status="OK")
        else:
            return datamodel.Status(status="END")
        
    async def gwas_logistic_update_global_params(self, config: datamodel.GWASConfig):
        while True:
            results = await self.request_clients(
                'gwas_logistic_update_local_params',
                config,
                beta=('JNPArray', self.repo.pop("logistic_reg_beta"))
            )

            # Get the noise from compensator
            noise = await self.request_compensators(config, 'gradient', 'hessian', 'loglikelihood')

            # Denoise
            gradient = np.sum(results.gradient, axis=0) - noise.gradient
            hessian = np.sum(results.hessian, axis=0) - noise.hessian
            loglikelihood = np.sum(results.loglikelihood, axis=0) - noise.loglikelihood
            
            beta, prev_beta, prev_loglikelihood, inv_hessian, jump_to = self.binary_gwas.global_params(
                jnp.array(gradient), jnp.array(hessian), jnp.array(loglikelihood),
                results.current_iteration[0], config.logistic_max_iters,
                self.repo.pop("prev_loglikelihood"), self.repo.pop("prev_beta")
            )
            
            # beta, prev_beta, prev_loglikelihood, inv_hessian, jump_to = self.binary_gwas.global_params(
            #     results.gradient, results.hessian, results.loglikelihood, 
            #     results.current_iteration[0], config.max_iterations,
            #     self.repo.pop("prev_loglikelihood"), self.repo.pop("prev_beta")
            # )
            
            if jump_to == 'local_iter_params':
                self.repo.add("logistic_reg_beta", beta)
                self.repo.add("prev_beta", prev_beta)
                self.repo.add("prev_loglikelihood", prev_loglikelihood)
                continue

            elif jump_to == 'global_stats':
                t_stat, pval, beta = self.binary_gwas.global_stats(beta, inv_hessian)
                self.repo.add("t_stat", t_stat)
                self.repo.add("pval", pval)
                self.repo.add("logistic_reg_beta", beta)
                break

            else:
                raise ValueError(f"Unexpected jump_to: {jump_to}")
        
    async def gwas_write_logistic_glm(self, config: datamodel.GWASConfig):
        await self.request_clients(
            "gwas_write_logistic_glm",
            config, 
            beta=('JNPArray', self.repo.pop("logistic_reg_beta")),
            t_stat=('JNPArray', self.repo.pop("t_stat")),
            pval=('JNPArray', self.repo.pop("pval")),
            n_obs=('JNPArray', self.repo.pop("n_obs")),
        )
        return datamodel.Status(status="OK")
    
    async def gwas_plot_statistics(self, config: datamodel.GWASConfig):
        await self.request_clients("gwas_plot_statistics", config)
        return datamodel.Status(status="OK")
    
    async def BasicBfileQC(self, config: datamodel.GWASConfig):
        try:
            await self.gwas_match_snps(config)
            await self.gwas_qc_stats(config)
            await self.gwas_filter_bfile(config)
            return datamodel.Status(status="OK")

        except Exception as e:
            self.logger.exception(e)
            return datamodel.Status(status=str(e))
    
    async def LDPruning(self, config: datamodel.GWASConfig):
        try:
            await self.gwas_prune_ld(config)
            await self.gwas_drop_ld_snps(config)
            return datamodel.Status(status="OK")

        except Exception as e:
            self.logger.exception(e)
            return datamodel.Status(status=str(e))
    
    async def GenotypePCA(self, config: datamodel.GWASConfig):
        try:
            await self.gwas_genotype_stdz_init(config)
            await self.stdz_global_nanmean(config)
            await self.stdz_global_mean(config)
            await self.stdz_global_var(config)
            await self.stdz_standardize(config)
            await self.RandomizedSVD(config)
            return datamodel.Status(status="OK")

        except Exception as e:
            self.logger.exception(e)
            return datamodel.Status(status=str(e))
    
    async def CovariateStdz(self, config: datamodel.GWASConfig):
        try:
            response = await self.gwas_covariate_stdz_init(config)
            if response.status == "BREAK":  # No covariates found
                return datamodel.Status(status="OK")
            await self.stdz_global_mean(config)
            await self.stdz_global_var(config)
            await self.stdz_standardize(config)
            await self.gwas_update_covariate(config)
            return datamodel.Status(status="OK")

        except Exception as e:
            self.logger.exception(e)
            return datamodel.Status(status=str(e))
    
    async def QuantGWAS(self, config: datamodel.GWASConfig):
        try:
            while True:
                response = await self.gwas_fit_linear_model(config)
                if response.status == "END":  # This represents all snps were processed
                    break
                await self.gwas_linear_stats(config)
                await self.gwas_write_glm(config)
            await self.gwas_plot_statistics(config)
            return datamodel.Status(status="OK")
        except Exception as e:
            self.logger.exception(e)
            return datamodel.Status(status=str(e))

    async def BinGWAS(self, config: datamodel.GWASConfig):
        try:
            while True:
                response = await self.gwas_logistic_init(config)
                if response.status == "END":  # This represents all snps were processed
                    break
                await self.gwas_logistic_update_global_params(config)
                await self.gwas_write_logistic_glm(config)
            await self.gwas_plot_statistics(config)
            return datamodel.Status(status="OK")

        except Exception as e:
            self.logger.exception(e)
            return datamodel.Status(status=str(e))
        
    async def FullQuantGWAS(self, config: datamodel.GWASConfig):
        try:
            await self.BasicBfileQC(config)
            await self.LDPruning(config)
            await self.GenotypePCA(config)
            await self.CovariateStdz(config)
            await self.QuantGWAS(config)
            return datamodel.Status(status="OK")

        except Exception as e:
            self.logger.exception(e)
            return datamodel.Status(status=str(e))
        
    async def FullBinGWAS(self, config: datamodel.GWASConfig):
        try:
            await self.BasicBfileQC(config)
            await self.LDPruning(config)
            await self.GenotypePCA(config)
            await self.CovariateStdz(config)
            await self.BinGWAS(config)
            return datamodel.Status(status="OK")

        except Exception as e:
            self.logger.exception(e)
            return datamodel.Status(status=str(e))

    async def RandomizedSVD(self, config: datamodel.SVDConfig):
        """
        This API utilizes A from the in-memory storage as input.
        Therefore, any preprocessing of A is not handled within this API.
        """
        try:
            # SVD
            await self.svd_global_init(config)
            while True:
                await self.svd_update_global_U(config)
                response = await self.svd_decompose_global_Us(config)
                if response.status == "BREAK":
                    break
            await self.svd_decompose_global_covariance(config)
            await self.svd_reconstruct_V(config)
            
            # Gram-Schmidt 
            await self.svd_to_gso(config)
            await self.gso_global_first_norm(config)
            while True:
                await self.gso_global_residuals(config)
                response = await self.gso_global_nth_norm(config)
                if response.status == "BREAK":
                    break
            await self.gso_normalization(config)
            
            # Final update
            await self.gso_to_svd(config)
            await self.svd_update_global_U(config)
            await self.svd_final_update(config)

            # Write results
            await self.svd_write_results(config)
            await self._svd_get_sharable_params(config)
            await self._svd_share_results(config)

            return datamodel.Status(status="OK")
        
        except Exception as e:
            self.logger.exception(e)
            return datamodel.Status(status=str(e))
        
    async def PCA(self, config: datamodel.SVDConfig):
        """
        This API leverages the RandomizedSVD to perform the PCA.
        Similar to the RandomizedSVD, it utilizes A from the in-memory storage as input.
        """
        try:
            # The standard preprocessing of PCA, centering the feature mean at 0
            await self.stdz_global_mean(config)
            await self.stdz_center_at_zero(config)
            
            # Use the SVD to perform the PCA
            await self.RandomizedSVD(config)

            if not config.to_pc:
                self.logger.warn("Performing PCA, however, the outputs are in the form of SVD.")

            return datamodel.Status(status="OK")
        
        except Exception as e:
            self.logger.exception(e)
            return datamodel.Status(status=str(e))
        
    async def PCAfromTabular(self, config: datamodel.TabularDataSVDConfig):
        """
        The difference from `SVDfromTabular` is the input matrix is centered at 0.
        """
        try:
            await self.read_matrix_from_csv(config)
            await self.PCA(config)
            return datamodel.Status(status="OK")
        
        except Exception as e:
            self.logger.exception(e)
            return datamodel.Status(status=str(e))
        
    async def SVDfromTabular(self, config: datamodel.TabularDataSVDConfig):
        """
        It performs SVD without any value preprocessing.
        """
        try:
            await self.read_matrix_from_csv(config)
            await self.RandomizedSVD(config)
            return datamodel.Status(status="OK")
        
        except Exception as e:
            self.logger.exception(e)
            return datamodel.Status(status=str(e))

    async def CoxPHRegression(self, config: datamodel.CoxPHRegressionConfig):
        try:
            await self.surv_load_clinical_data(config)
            await self.surv_global_cox(config)
            await self.surv_cox_results(config)
            return datamodel.Status(status="OK")
        
        except Exception as e:
            self.logger.exception(e)
            return datamodel.Status(status=str(e))
    
    async def KaplanMeier(self, config: datamodel.KaplanMeierConfig):
        try:
            await self.surv_load_clinical_data(config)
            await self._X_to_A(config)
            
            await self.stdz_global_mean(config)
            await self.stdz_global_var(config)
            await self.stdz_standardize(config)
            
            await self.surv_global_km(config)
            await self.surv_km_results(config)
            return datamodel.Status(status="OK")
        
        except Exception as e:
            self.logger.exception(e)
            return datamodel.Status(status=str(e))

    async def surv_load_clinical_data(self, config: datamodel.CoxPHRegressionConfig):
        results = await self.request_clients('surv_local_load_metadata', config)
        self.repo.add("n_features", results.n_features[0])
        return datamodel.Status(status="OK")
    
    async def surv_global_cox(self, config: datamodel.CoxPHRegressionConfig):
        Xanc = self.surv_cox.global_create_Xanc(self.repo.get("n_features"), config.r)
        results = await self.request_clients('surv_local_create_proxy_data', config, Xanc=('NPArray', Xanc))

        coef, coef_var, baseline_hazard, feature_mean = self.surv_cox.global_fit_model(
            results.X_tilde, results.Xanc_tilde, results.y, results.sums, config.alpha, config.step_size
        )
        
        self.repo.add("coef", coef)
        self.repo.add("coef_var", coef_var)
        self.repo.add("baseline_hazard", baseline_hazard)
        self.repo.add("feature_mean", feature_mean)
        return datamodel.Status(status="OK")
    
    async def surv_cox_results(self, config: datamodel.CoxPHRegressionConfig):
        # This is because the coef and coef_var are different for each client
        for i in range(len(config.clients)):
            await self.request_clients(
                'surv_local_cox_results',
                config.to_client_config(i),
                coef=('NPArray', self.repo.get("coef")[i][0]),
                coef_var=('NPArray', self.repo.get("coef_var")[i][0]),
                baseline_hazard=('PDDataFrame', self.repo.get('baseline_hazard')),
                mean=('NPArray', self.repo.get("mean"))
            )
        
        return datamodel.Status(status="OK")
    
    async def surv_global_km(self, config: datamodel.KaplanMeierConfig):
        results = await self.request_clients('surv_km_group_by_std', config)
        
        fitted_km, stats = self.km.global_fit_model(results.grouped_y, config.alpha)
        
        self.repo.add("fitted_km", fitted_km)
        self.repo.add("logrank_stats", stats)
        
        return datamodel.Status(status="OK")
    
    async def surv_km_results(self, config: datamodel.KaplanMeierConfig):
        await self.request_clients(
           'surv_km_results',
            config,
            fitted_km=('List[List[Optional[Dict[str, NPArray]]]]', self.repo.get("fitted_km")),
            logrank_stats=('List[List[float]]', self.repo.get("logrank_stats"))
        )
        return datamodel.Status(status="OK")
    
    async def gwas_genotype_stdz_init(self, config: datamodel.GWASConfig):
        await self.request_clients('gwas_genotype_stdz_init', config)
        return datamodel.Status(status="OK")
    
    async def stdz_global_nanmean(self, config: datamodel.PlanBaseConfig):
        results = await self.request_clients('stdz_local_col_nansum', config)
        
        mean, _ = self.stdz.global_mean(results.col_sum, results.row_count)
        self.repo.add("mean", mean)
        return datamodel.Status(status="OK")
        
    async def stdz_global_mean(self, config: datamodel.PlanBaseConfig):
        results = await self.request_clients('stdz_local_imputed_mean', config, mean=('JNPArray', self.repo.get("mean")))
        
        mean, _ = self.stdz.global_mean(results.col_sum, results.row_count)
        self.repo.add("mean", mean)
        return datamodel.Status(status="OK")
    
    async def stdz_center_at_zero(self, config: datamodel.PlanBaseConfig):
        await self.request_clients('stdz_center_at_zero', config, mean=('JNPArray', self.repo.get("mean")))
        return datamodel.Status(status="OK")
    
    async def stdz_global_var(self, config: datamodel.PlanBaseConfig):
        results = await self.request_clients('stdz_local_ssq', config, mean=('JNPArray', self.repo.get("mean")))
        
        var, delete = self.stdz.global_var(results.ssq, results.row_count)
        self.repo.add("var", var)
        self.repo.add("delete", delete)
        return datamodel.Status(status="OK")
    
    async def stdz_standardize(self, config: datamodel.PlanBaseConfig):
        await self.request_clients(
            'stdz_local_standardize', 
            config, 
            var=('JNPArray', self.repo.get("var")), 
            delete=('JNPArray', self.repo.get("delete"))
        )
        return datamodel.Status(status="OK")
    
    async def svd_global_init(self, config: datamodel.SVDConfig):
        results = await self.request_clients('svd_local_init', config)
        
        prev_U, current_iteration, converged, Us = self.svd.global_init(results.n_features[0], config.k1)
        
        self.repo.add("prev_U", prev_U)
        self.repo.add("current_iteration", current_iteration)
        self.repo.add("converged", converged)
        self.repo.add("Us", Us)
        
        return datamodel.Status(status='OK')
    
    async def svd_update_global_U(self, config: datamodel.SVDConfig):
        results = await self.request_clients('svd_update_local_U', config)
        
        U, S = self.svd.update_global_U(results.U)
        converged = self.svd.check_convergence(U, self.repo.get("prev_U"), config.epsilon)
        prev_U, Us, current_iteration = self.svd.update_global_Us(U, self.repo.get("Us"), self.repo.get("current_iteration"))

        self.repo.add("U", U)
        self.repo.add("S", S)
        self.repo.add("prev_U", prev_U)
        self.repo.add("current_iteration", current_iteration)
        self.repo.add("converged", converged)
        self.repo.add("Us", Us)
        
        return datamodel.Status(status='OK')
    
    async def svd_decompose_global_Us(self, config: datamodel.SVDConfig):
        results = await self.request_clients(
            'svd_update_local_V',
            config,
            U=('JNPArray', self.repo.get("U")),
            current_iteration=('int', self.repo.get("current_iteration")),
            converged=('bool', self.repo.get("converged")),
        )
        
        if all(results.termination):
            U = self.svd.decompose_global_Us(self.repo.get("Us"))
            self.repo.add("U", U)
            return datamodel.Status(status='BREAK')
            
        else:
            return datamodel.Status(status='CONTINUE')
        
    async def svd_decompose_global_covariance(self, config: datamodel.SVDConfig):
        results = await self.request_clients('svd_compute_local_covariance', config, U=('JNPArray', self.repo.get("U")))
        
        Vp = self.svd.decompose_global_covariance(results.PPt, config.k2)
        
        self.repo.add("Vp", Vp)
        
        return datamodel.Status(status='OK')
    
    async def svd_reconstruct_V(self, config: datamodel.SVDConfig):
        await self.request_clients('svd_recontruct_local_V', config, Vp=('JNPArray', self.repo.get('Vp')))
        
        return datamodel.Status(status='OK')
    
    async def svd_to_gso(self, config: datamodel.PlanBaseConfig):
        await self.request_clients('svd_to_gso', config)
        
        return datamodel.Status(status='OK')
    
    async def gso_global_first_norm(self, config: datamodel.PlanBaseConfig):
        results = await self.request_clients('gso_local_first_norm', config)
        
        global_norms, eigen_idx = self.gso.global_first_norm(results.partial_norm)
        
        self.repo.add("global_norms", global_norms)
        self.repo.add("eigen_idx", eigen_idx)
        
        return datamodel.Status(status='OK')
    
    async def gso_global_residuals(self, config: datamodel.PlanBaseConfig):
        results = await self.request_clients(
            'gso_local_residuals',
            config,
            eigen_idx=('int', self.repo.get("eigen_idx")),
            global_norms=('List[JNPArray]', self.repo.get("global_norms"))
        )

        residuals = self.gso.global_residuals(results.residuals)
        
        self.repo.add("residuals", residuals)
        
        return datamodel.Status(status='OK')
    
    async def gso_global_nth_norm(self, config: datamodel.PlanBaseConfig):
        results = await self.request_clients(
            'gso_local_nth_norm',
            config,
            eigen_idx=('int', self.repo.get("eigen_idx")),
            residuals=('List[JNPArray]', self.repo.get("residuals"))
        )
        
        global_norms, eigen_idx, jump_to = self.gso.global_nth_norm(
            self.repo.get("global_norms"),
            results.partial_norm,
            self.repo.get("eigen_idx"),
            results.k2[0]
        )
        
        self.repo.add("global_norms", global_norms)
        self.repo.add("eigen_idx", eigen_idx)
        
        if jump_to == 'local_residuals':
            return datamodel.Status(status='CONTINUE')
        else:
            return datamodel.Status(status='BREAK')
        
    async def gso_normalization(self, config: datamodel.PlanBaseConfig):
        await self.request_clients(
            'gso_local_normalization',
            config,
            global_norms=('List[JNPArray]', self.repo.get("global_norms"))
        )
        
        return datamodel.Status(status='OK')
    
    async def gso_to_svd(self, config: datamodel.PlanBaseConfig):
        await self.request_clients('gso_to_svd', config)
        
        return datamodel.Status(status='OK')
    
    async def svd_final_update(self, config: datamodel.SVDConfig):
        await self.request_clients(
            'svd_final_update',
            config,
            U=('JNPArray', self.repo.get('U')),
            S=('JNPArray', self.repo.get('S'))
        )
        
        return datamodel.Status(status='OK')
    
    async def svd_write_results(self, config: datamodel.SVDConfig):
        await self.request_clients('svd_write_results', config)
        
        return datamodel.Status(status='OK')

    async def _svd_get_sharable_params(self, config: datamodel.SVDConfig):
        results = await self.request_clients('_svd_get_sharable_params', config)

        vec_name = 'PC' if config.to_pc else 'Eigenvec'
        partial_vec = jnp.concatenate(results.partial_vec)
        partial_vec = pd.DataFrame(partial_vec, columns=[f'{vec_name}{i+1}' for i in range(partial_vec.shape[1])])

        metadata = pd.concat(results.metadata).reset_index(drop=True)
        svd_fig_df = pd.concat([metadata, partial_vec], axis=1)

        self.repo.add("svd_fig_df", svd_fig_df)
        
        return datamodel.Status(status='OK')
    
    async def _svd_share_results(self, config: datamodel.SVDConfig):
        await self.request_clients(
            '_svd_share_results',
            config,
            svd_fig_df=('PDDataFrame', self.repo.get("svd_fig_df"))
        )
        
        return datamodel.Status(status='OK')

    async def read_matrix_from_csv(self, config: datamodel.TabularReaderConfig):
        await self.request_clients('local_read_matrix_from_csv', config)
        
        return datamodel.Status(status='OK')
    
    async def _X_to_A(self, config: datamodel.PlanBaseConfig):
        await self.request_clients('_X_to_A', config)
                
        return datamodel.Status(status='OK')


class ClientService(ServiceController):

    def __init__(self, config_path) -> None:
        super().__init__(config_path)
        if "servers" in self.config.keys():
            self.__servers = self.config["servers"]
            self.__server_ws = net.WSConnectionManager()

        if "compensators" in self.config.keys():
            self.__compensators = self.config["compensators"]
            self.__compensator_ws = net.WSConnectionManager()
        
        self.noise = InMemoryRepository()

    @property
    def node_id(self):
        return self._auth.node_id

    async def run(self):
        for server in self.__servers:
            await self.register(server, self.__server_ws)
        
        for compensator in self.__compensators:
            await self.register(compensator, self.__compensator_ws)

        for server in self.__servers:
            ws = self.__server_ws.get(server["node_id"])
            server_task = asyncio.create_task(self.handle_ws(ws))

        for compensator in self.__compensators:
            ws = self.__compensator_ws.get(compensator["node_id"])
            compensator_task = asyncio.create_task(self.handle_ws(ws))
        
        # Make them hang
        await server_task
        await compensator_task

    async def handle_request(self, request: datamodel.HTTPRequest) -> Any:
        self._auth.authenticate(request.node_id)
        return await self.eval(request.api, request.args)

    async def handle_ws(self, ws: WebSocketResponse) -> Any:
        try:
            while True:
                # Wait for server task
                data = pickle.loads(await ws.receive_bytes())
                request = datamodel.RPCRequest(**data)
                self.logger.info(f"Receive RPC call to {request.api}")
                result = await self.eval(request.api, request.args)
                await ws.send_bytes(pickle.dumps(result.model_dump()))
        finally:
            await self.__server_ws.disconnectall()

    async def register(self, profile, ws_manager: net.WSConnectionManager):
        url = net.construct_url(profile, "/ws/tasks", protocol=profile.get("protocol", "ws"))

        # Corresponds to server ws.accept()
        ws = await ws_manager.connect(profile["node_id"], url)

        # Corresponds to server ws.receive_json()
        registery = {"node_id": str(self.node_id), "role": "client"}

        await ws.send_bytes(pickle.dumps(registery))
        self.logger.info(f"Register as client:{self.node_id}")
        return ws
    
    async def compress_result(self, files: list):
        with zipfile.ZipFile(f'{self.node_id}-result.zip', mode='w') as zf:
            for f in files:
                # Simple file
                if os.path.isfile(f):
                    zf.write(f, arcname=f.split("/")[-1])
                
                # All files under a directory
                elif os.path.isdir(f):
                    for f_ in utils.recur_list_files(f):
                        zf.write(f_, arcname=f_.split("/")[-1])
                
                else:
                    self.logger.error(f"File {f} is not a file or directory")
            
        with open(f'{self.node_id}-result.zip', 'rb') as f:
            content = f.read()
        os.remove(f'{self.node_id}-result.zip')
        
        return send_model(
            filename=('str', f'{self.node_id}-result.zip'),
            content=('list', list(content))
        )
    
    async def send_noise(self, request: datamodel.CompensatorRequest):
        noise = {arg:('NPArray', np.array(self.noise.pop(arg))) for arg in request.args}

        return send_model(**noise)

    async def gwas_get_metadata(self, config: datamodel.GWASConfig):
        t = time.perf_counter_ns()
        autosome_snp_list, sample_list, autosome_snp_table = self.qc.local_get_metadata(
            **config.client_bfile_config()
        )

        self.logger.info(f"Client - get_metadata: {time.perf_counter_ns() - t} ns")

        self.repo.add("autosome_snp_table", autosome_snp_table)
        self.repo.add("sample_list", sample_list)

        return send_model(
            autosome_snp_list=('List[str]', autosome_snp_list)
        )

    async def gwas_qc_stats(self, autosome_snp_list: list, config: datamodel.GWASConfig):
        t = time.perf_counter_ns()
        
        allele_count, n_obs = self.qc.local_qc_stats(
            autosome_snp_list, config.local_qc_output_path,
            self.repo.get("autosome_snp_table"), config.bfile_path,
        )
        self.logger.info(f"Client - local_qc_stats: {time.perf_counter_ns() - t} ns")

        return send_model(
            allele_count=('NPArray', allele_count),
            n_obs=('int', n_obs)
        )

    async def gwas_filter_bfile(self, autosome_snp_list: list, config: datamodel.GWASConfig):
        t = time.perf_counter_ns()

        if len(autosome_snp_list) == 0:
            bfile_path, cov_path = config.bfile_path, config.cov_path
        else:
            bfile_path, cov_path = self.qc.local_filter_bfile(
                autosome_snp_list, config.local_qc_output_path,
                config.cov_path, self.repo.get("sample_list"),
                self.repo.get("autosome_snp_table"),
                config.bfile_path, config.mind,
            )
        self.logger.info(f"Client - filter_bfile: {time.perf_counter_ns() - t} ns")

        self.repo.add("bfile_path", bfile_path)
        self.repo.add("cov_path", cov_path)
        return send_model(status=('str', 'OK'))
    
    async def gwas_local_prune_ld(self, config: datamodel.GWASConfig):
        t = time.perf_counter_ns()
        
        # Bfile
        bfile_path = self.repo.get("bfile_path")
        if bfile_path is None:
            self.logger.info(f"No bfile path stored in memory. Load from configuration.")
            bfile_path = config.bfile_path
        
        remained_snps = self.ld.local_ldprune(
            bfile_path,
            f'{config.local_qc_output_path}.ld',
            win_size=config.win_size,
            step=config.step,
            r2=config.r2,
            extra_arg="--bad-ld"  # This is for the few SNPs situation
        )
        self.logger.info(f"Client - local_prune_ld: {time.perf_counter_ns() - t} ns")

        self.repo.add("bfile_path", bfile_path)
        
        return send_model(remained_snps=('NPArray', remained_snps))
    
    async def gwas_drop_ld_snps(self, remained_snps: np.array, config: datamodel.GWASConfig):
        t = time.perf_counter_ns()

        bfile_path = self.ld.local_extract_snp(
            self.repo.get("bfile_path"),
            f'{config.local_qc_output_path}.ld',
            remained_snps
        )
        self.logger.info(f"Client - drop_ld_snps: {time.perf_counter_ns() - t} ns")
        
        self.logger.info(f'After LD bfile: {bfile_path}')
        
        return send_model(status=('str', 'OK'))
    
    async def gwas_covariate_stdz_init(self, config: datamodel.GWASConfig):
        t = time.perf_counter_ns()
        # Bfile
        bfile_path = self.repo.get("bfile_path")
        if bfile_path is None:
            self.logger.info(f"No bfile path stored in memory. Load from configuration.")
            bfile_path = config.bfile_path

        # Covariates
        cov_path = self.repo.get("cov_path")
        if cov_path is None:
            self.logger.info(f"No covariate path stored in memory. Load from configuration.")
            cov_path = config.cov_path

        self.logger.info(f"Load bed file {bfile_path}.bed for covariate standardization.")
        self.logger.info(f"Load covariate file {cov_path} for covariate standardization.")
        
        covariates, covar_names, sample_ids = self.cov.local_get_covariates(
            bfile_path, cov_path, config.pheno_path, config.pheno_name
        )
        
        self.logger.info(f"Covariate matrix:\n{covariates}\n")
        self.logger.info(f"Covariate names: {covar_names}\n")
        self.logger.info(f"Client - covariate_stdz_init: {time.perf_counter_ns() - t} ns")
        
        if covariates is None:
            return send_model(status=('str', 'BREAK'))

        else:
            self.repo.add("A", covariates)
            self.repo.add("covar_names", covar_names)
            self.repo.add("cov_sample_ids", sample_ids)
            
            return send_model(status=('str', 'OK'))
    
    async def gwas_local_update_covariate(self, config: datamodel.GWASConfig):
        t = time.perf_counter_ns()
        
        # Covariates
        cov_path = self.repo.get("cov_path")
        if cov_path is None:
            self.logger.info(f"No covariate path stored in memory. Load from configuration.")
            cov_path = config.cov_path

        cov_path = self.cov.local_update_covariates(
            cov_path=cov_path,
            covar_matrix=self.repo.get("A"),
            covar_names=self.repo.get("covar_names"),
            cov_sample_ids=self.repo.get("cov_sample_ids"),
            pc_path=os.path.join(f"{config.svd_save_dir}", "row.eigenvec.csv"),
            pc_sample_ids=self.repo.get("sample_info"),
            save_dir=config.local_qc_output_path
        )

        csv_ = pd.read_csv(cov_path, sep='\t')
        self.logger.info(f"\n{csv_}\n")
        self.logger.info(f"Client - local_update_covariate: {time.perf_counter_ns() - t} ns")
        
        if cov_path is not None:
            self.repo.add("cov_path", cov_path)
        
        return send_model(status=('str', 'OK'))

    def _create_gwasdata_iterator(self, config: datamodel.GWASConfig):
        t = time.perf_counter_ns()
        self.logger.info(f"No dataloader stored in memory. Load from configuration.")

        # Bfile
        bfile_path = self.repo.get("bfile_path")
        if bfile_path is None:
            self.logger.info(f"No bfile path stored in memory. Load from configuration.")
            bfile_path = config.bfile_path

        # Covariates
        cov_path = self.repo.get("cov_path")
        if cov_path is None:
            self.logger.info(f"No covariate path stored in memory. Load from configuration.")
            cov_path = config.cov_path

        self.logger.info(f"Load bed file {bfile_path}.bed for regression calculation.")
        self.logger.info(f"Load covariate file {cov_path} for regression calculation.")

        loader =GWASDataIterator(
            bfile_path=bfile_path,
            cov_path=cov_path,
            style='snp',
            snp_step=config.snp_chunk_size
        )
        self.logger.info(f"Client - create dataloader: {time.perf_counter_ns() - t} ns")
        return loader

    async def gwas_calculate_covariance(self, config: datamodel.GWASConfig):
        # GWASDataIterator
        loader = self.repo.get("QuantitativeGWAS")
        pbar = self.repo.get("pbar")
        if loader is None:
            loader = self._create_gwasdata_iterator(config)
            pbar = tqdm(loader, total=len(loader), desc="Processing the chunk of SNPs")

        if not loader.iterator.is_end():
            t = time.perf_counter_ns()
            
            pheno = self.repo.get("pheno")
            if pheno is None and config.pheno_path is not None:
                self.repo.add("pheno", PhenotypeReader(config.pheno_path, config.pheno_name).read())

            dataset = self.quant_gwas.local_load_chunk_gwasdata(next(loader), pheno=self.repo.get("pheno"))
            self.logger.info(f"Client - local_load_gwasdata: {time.perf_counter_ns() - t} ns")
            
            pbar.update(1)

            t = time.perf_counter_ns()
            XtX, Xty = self.quant_gwas.local_calculate_covariances(
                dataset[0],  # genotype
                dataset[1],  # covariates
                dataset[2],  # phenotype
            )
            self.logger.info(f"Client - local_calculate_covariances: {time.perf_counter_ns() - t} ns")

            # Add noise
            XtX_noise = hyfed.randn(*XtX.shape)
            Xty_noise = hyfed.randn(*Xty.shape)
            XtX += XtX_noise
            Xty += Xty_noise

            self.repo.add("y", dataset[2])
            self.repo.add("cov_values", dataset[1])
            self.repo.add("genotype", dataset[0])
            self.repo.add("snp_info", dataset[4])
            self.repo.add("QuantitativeGWAS", loader)
            self.repo.add("pbar", pbar)
            self.noise.add("XtX", XtX_noise)
            self.noise.add("Xty", Xty_noise)

            return send_model(
                XtX=('JNPArray', XtX),
                Xty=('JNPArray', Xty)
            )
        else:

            return send_model(
                XtX=('list', []),
                Xty=('list', [])
            )

    async def gwas_calculate_sse_and_obs(self, beta: jnp.array, config: datamodel.GWASConfig):
        t = time.perf_counter_ns()
        sse, n_obs = self.quant_gwas.local_sse_and_obs(
            beta,
            self.repo.get("y"),
            self.repo.pop("genotype"),
            self.repo.get("cov_values")
        )
        
        sse_noise = hyfed.randn(*sse.shape)
        sse += sse_noise
        
        self.repo.add("beta", beta)
        self.noise.add("sse", sse_noise)
        self.logger.info(f"Client - local_sse_and_obs: {time.perf_counter_ns() - t} ns")

        return send_model(
            sse=('JNPArray', sse),
            n_obs=('JNPArray', n_obs)
        )
    
    async def gwas_write_glm(self, t_stat: np.array, pval: np.array, n_obs: np.array, config: datamodel.GWASConfig):
        beta = self.repo.get("beta")

        # Extract SNP statistics
        beta = beta.view()[:,0]
        t_stat = t_stat.view()[:,0]
        pval = pval.view()[:,0]

        self.output.regression_results(
            self.repo.get("snp_info"), t_stat, pval, beta, n_obs, config.regression_save_dir
        )

        return send_model(status=('str', 'OK'))
    
    async def gwas_logistic_init(self, config: datamodel.GWASConfig):
        # GWASDataIterator
        loader = self.repo.get("BinaryGWAS")
        pbar = self.repo.get("pbar")
        if loader is None:
            loader = self._create_gwasdata_iterator(config)
            pbar = tqdm(loader, total=len(loader), desc="Processing the chunk of SNPs")

        if not loader.iterator.is_end():
            t = time.perf_counter_ns()
            
            pheno = self.repo.get("pheno")
            if pheno is None and config.pheno_path is not None:
                self.repo.add("pheno", PhenotypeReader(config.pheno_path, config.pheno_name).read())
            
            dataset = self.binary_gwas.local_load_chunk_gwasdata(next(loader), pheno=self.repo.get("pheno"))
            self.logger.info(f"Client - local_load_gwasdata: {time.perf_counter_ns() - t} ns")

            pbar.update(1)

            t = time.perf_counter_ns()
            n_obs, gradient, hessian, loglikelihood, current_iteration = self.binary_gwas.local_init_params(
                genotype=dataset[0],
                covariates=dataset[1],
                phenotype=dataset[2],
            )
            self.logger.info(f"Client - local_logistic_initialize: {time.perf_counter_ns() - t} ns")
            
            # Add noise
            gradient_noise = hyfed.randn(*gradient.shape)
            hessian_noise = hyfed.randn(*hessian.shape)
            loglikelihood_noise = hyfed.randn(*loglikelihood.shape)
            
            gradient += gradient_noise
            hessian += hessian_noise
            loglikelihood += loglikelihood_noise
            
            self.repo.add("y", dataset[2])
            self.repo.add("cov_values", dataset[1])
            self.repo.add("genotype", dataset[0])
            self.repo.add("snp_info", dataset[4])
            self.repo.add("BinaryGWAS", loader)
            self.repo.add("pbar", pbar)
            self.repo.add("current_iteration", current_iteration)
            self.noise.add("gradient", gradient_noise)
            self.noise.add("hessian", hessian_noise)
            self.noise.add("loglikelihood", loglikelihood_noise)
            
            print(f"Iteration: {current_iteration}")
            
            return send_model(
                gradient=('JNPArray', gradient),
                hessian=('JNPArray', hessian),
                loglikelihood=('JNPArray', loglikelihood),
                current_iteration=('int', current_iteration),
                n_obs=('JNPArray', n_obs)
            )
            
        else:
            return send_model(
                gradient=('JNPArray', []),
                hessian=('JNPArray', []),
                loglikelihood=('JNPArray', []),
                current_iteration=('int', -1),
                n_obs=('JNPArray', jnp.array([]))
            )
            
    async def gwas_logistic_update_local_params(self, beta: jnp.array, config: datamodel.GWASConfig):
        t = time.perf_counter_ns()
        gradient, hessian, loglikelihood, current_iteration, _ = self.binary_gwas.local_iter_params(
            beta,
            self.repo.get("current_iteration")
        )
        self.logger.info(f"Client - local_update_local_params: {time.perf_counter_ns() - t} ns")
        
        # Add noise
        gradient_noise = hyfed.randn(*gradient.shape)
        hessian_noise = hyfed.randn(*hessian.shape)
        loglikelihood_noise = hyfed.randn(*loglikelihood.shape)
        
        gradient += gradient_noise
        hessian += hessian_noise
        loglikelihood += loglikelihood_noise
        
        self.repo.add("current_iteration", current_iteration)
        self.noise.add("gradient", gradient_noise)
        self.noise.add("hessian", hessian_noise)
        self.noise.add("loglikelihood", loglikelihood_noise)
        
        self.logger.info(f"Iteration: {current_iteration}")

        return send_model(
            gradient=('JNPArray', gradient),
            hessian=('JNPArray', hessian),
            loglikelihood=('JNPArray', loglikelihood),
            current_iteration=('int', current_iteration)
        )
        
    async def gwas_write_logistic_glm(self, beta: jnp.array, t_stat: jnp.array, pval: jnp.array,
                                      n_obs: jnp.array, config: datamodel.GWASConfig):
        self.output.regression_results(
            self.repo.get("snp_info"), t_stat, pval, beta, n_obs, config.regression_save_dir
        )
        
        return send_model(status=('str', 'OK'))
    
    async def gwas_plot_statistics(self, config: datamodel.GWASConfig):
        self.output.gwas_plot_statistics(config.regression_save_dir)
        return send_model(status=('str', 'OK'))
    
    async def surv_local_load_metadata(self, config: datamodel.ClinicalBaseConfig):
        X, y, keep_feature_cols, _ = self.surv_cox.local_load_metadata(config.clinical_data_path, config.feature_cols, config.meta_cols)
        self.repo.add("X", X)
        self.repo.add("y", y)
        self.repo.add("keep_feature_cols", keep_feature_cols)

        return send_model(
            n_features=('int', X.shape[1])
        )
    
    async def surv_local_create_proxy_data(self, Xanc: np.array, config: datamodel.CoxPHRegressionConfig):
        F, X_tilde, Xanc_tilde, feature_sum = self.surv_cox.local_create_proxy_data(
            self.repo.get("X"),
            Xanc,
            self.repo.get("y"),
            config.k,
            config.bs_prop,
            config.bs_times,
            config.alpha,
            config.step_size
        )
        self.repo.add("F", F)

        return send_model(
            X_tilde=('List[NPArray]', X_tilde),
            Xanc_tilde=('List[NPArray]', Xanc_tilde),
            y=('NPArray', self.repo.get("y")),
            sums=('NPArray', feature_sum)
        )
    
    async def surv_local_cox_results(self, coef: np.array, coef_var: np.array, baseline_hazard: pd.DataFrame, mean: np.array, config: datamodel.CoxPHRegressionConfig):
        surv_func = self.surv_cox.local_recover_survival(
            self.repo.get("keep_feature_cols"),
            coef,
            coef_var,
            baseline_hazard,
            mean,
            self.repo.get("F"),
            config.alpha
        )
        self.output.cox_regression_results(surv_func.summary, config.save_dir)
        return send_model(status=('str', 'OK'))

    async def surv_km_group_by_std(self, config: datamodel.KaplanMeierConfig):
        grouped_y, n_std = self.km.local_group_by_std(self.repo.get("A"), config.n_std, self.repo.get("y"))
        
        self.repo.add("n_std", n_std)
        
        return send_model(
            grouped_y=('List[List[NPArray]]', grouped_y),
            n_std=('List[float]', n_std),
        )
        
    async def surv_km_results(self, fitted_km: list, logrank_stats: list, config: datamodel.KaplanMeierConfig):
        self.output.kaplan_meier_results(
            self.repo.get("keep_feature_cols"), fitted_km, logrank_stats, self.repo.get("n_std"), config.save_dir
        )
        
        return send_model(status=('str', 'OK'))

    async def gwas_genotype_stdz_init(self, config: datamodel.GWASConfig):
        # Bfile
        bfile_path = self.repo.get("bfile_path")  # Check for filtered bfile
        if bfile_path is None:
            self.logger.info(f"No bfile path stored in memory. Load from configuration.")
            bfile_path = config.bfile_path
        else:
            bfile_path = f'{bfile_path}.ld'

        # Covariates
        cov_path = self.repo.get("cov_path")  # Check for filtered covariates
        if cov_path is None:
            self.logger.info(f"No covariate path stored in memory. Load from configuration.")
            cov_path = config.cov_path

        self.logger.info(f"Load bed file {bfile_path}.bed for standardization.")
        self.logger.info(f"Load covariate file {cov_path} for standardization.")
        
        genotype, sample_info, snp_info = self.stdz.local_load_gwasdata(
            bfile_path, cov_path, config.pheno_path, config.pheno_name
        )
        self.logger.info(f"The genotype matrix shape: {genotype.shape[0]} samples, {genotype.shape[1]} SNPs.")
        
        # Notice that some sample may be dropped due to the missing covariates, 
        # causing the incomplete genotype matrix to be standardize.
        self.repo.add("A", genotype)
        self.repo.add("sample_info", sample_info)
        self.repo.add("snp_info", snp_info)

        return send_model(status=('str', 'OK'))
    
    async def stdz_local_col_nansum(self, config: datamodel.PlanBaseConfig):
        col_sum, row_count, _ = self.stdz.local_col_nansum(self.repo.get("A"))

        return send_model(
            col_sum=('NPArray', col_sum),
            row_count=('NPArray', row_count)
        )
        
    async def stdz_local_imputed_mean(self, mean: jnp.array, config: datamodel.PlanBaseConfig):
        A, col_sum, row_count, _ = self.stdz.local_imputed_mean(self.repo.pop("A"), mean)
        self.repo.add("A", A)

        return send_model(
            col_sum=('NPArray', col_sum),
            row_count=('NPArray', row_count)
        )
    
    async def stdz_center_at_zero(self, mean: jnp.array, config: datamodel.PlanBaseConfig):
        self.repo.add("A", self.repo.get("A") - mean)
        return send_model(status=('str', 'OK'))
        
    async def stdz_local_ssq(self, mean: jnp.array, config: datamodel.PlanBaseConfig):
        A, ssq, row_count = self.stdz.local_ssq(self.repo.pop("A"), mean)
        self.repo.add("A", A)

        return send_model(
            ssq=('NPArray', ssq),
            row_count=('NPArray', row_count)
        )
        
    async def stdz_local_standardize(self, var: jnp.array, delete: jnp.array, config: datamodel.PlanBaseConfig):
        A = self.stdz.local_standardize(self.repo.pop("A"), var, delete.astype('int32'))
        self.repo.add("A", A)

        return send_model(status=('str', 'OK'))
    
    async def svd_local_init(self, config: datamodel.SVDConfig):
        A, V, n_features = self.svd.local_init(self.repo.pop("A"), config.k1)
        self.repo.add("A", A)
        self.repo.add("V", V)
        
        if self.repo.get("row_metadata") is not None:
            self.repo.add("svd_row_metadata", self.repo.get("row_metadata"))
        
        return send_model(n_features=('int', n_features))
    
    async def svd_update_local_U(self, config: datamodel.SVDConfig):
        U = self.svd.update_local_U(self.repo.get("A"), self.repo.get("V"))
        
        return send_model(U=('JNPArray', U))
    
    async def svd_update_local_V(self, U: jnp.array, converged: bool, current_iteration: int, config: datamodel.SVDConfig):
        V, jump_to = self.svd.update_local_V(
            self.repo.get('A'), 
            U, 
            converged, 
            current_iteration, 
            config.svd_max_iters
        )
        
        self.repo.add("V", V)

        if jump_to == 'update_local_U':
            return send_model(termination=('bool', False))
        else:
            return send_model(termination=('bool', True))
        
    async def svd_compute_local_covariance(self, U: jnp.array, config: datamodel.SVDConfig):
        P, PPt = self.svd.compute_local_covariance(self.repo.get("A"), U)
        
        self.repo.add("P", P)
        self.repo.add("PPt", PPt)
        
        return send_model(PPt=('JNPArray', PPt))
        
    async def svd_recontruct_local_V(self, Vp: jnp.array, config: datamodel.SVDConfig):
        V = self.svd.recontruct_local_V(self.repo.get("P"), Vp)
        
        self.repo.add("V", V)

        return send_model(status=('str', 'OK'))  

    async def svd_to_gso(self, config: datamodel.PlanBaseConfig):
        M = self.gso.local_make_V_as_M(self.repo.get("V"))
        
        self.repo.add("M", M)
        self.repo.add("k", M.shape[1])  # for `global_nth_norm`
        
        return send_model(status=('str', 'OK'))
    
    async def gso_local_first_norm(self, config: datamodel.PlanBaseConfig):
        partial_norm, orthogonalized = self.gso.local_first_norm(self.repo.get("M"))
        
        self.repo.add("orthogonalized", orthogonalized)
        
        return send_model(partial_norm=('JNPArray', partial_norm))
    
    async def gso_local_residuals(self, eigen_idx: int, global_norms: list, config: datamodel.PlanBaseConfig):
        residuals = self.gso.local_residuals(
            self.repo.get("M"),
            self.repo.get("orthogonalized"),
            eigen_idx,
            global_norms
        )
        
        return send_model(residuals=('List[JNPArray]', residuals))
    
    async def gso_local_nth_norm(self, eigen_idx: int, residuals: jnp.array, config: datamodel.PlanBaseConfig):
        partial_norm, orthogonalized = self.gso.local_nth_norm(
            self.repo.get("M"),
            self.repo.get("orthogonalized"),
            eigen_idx,
            residuals
        )
        
        self.repo.add("orthogonalized", orthogonalized)
        
        return send_model(partial_norm=('JNPArray', partial_norm), k2=('int', self.repo.get("k")))
    
    async def gso_local_normalization(self, global_norms: list, config: datamodel.PlanBaseConfig):
        M = self.gso.local_normalization(global_norms, self.repo.get("orthogonalized"))
    
        self.repo.add("M", M)
        
        return send_model(status=('str', 'OK'))
    
    async def gso_to_svd(self, config: datamodel.PlanBaseConfig):
        V = self.gso.local_make_M_as_V(self.repo.get("M"))
    
        self.repo.add("V", V)
        
        return send_model(status=('str', 'OK'))
    
    async def svd_final_update(self, U: jnp.array, S: jnp.array, config: datamodel.SVDConfig):
        self.repo.add("U", U)
        self.repo.add("S", S)

        return send_model(status=('str', 'OK'))
    
    async def svd_write_results(self, config: datamodel.SVDConfig):
        row_metadata = self.repo.get("svd_row_metadata")
        col_metadata = self.repo.get("svd_col_metadata")
        
        self.output.svd_results(
            row_metadata,
            col_metadata,
            self.repo.get("U"),
            self.repo.get("S"),
            self.repo.get("V"),
            config.svd_save_dir,
            to_pc=config.to_pc
        )
        
        return send_model(status=('str', 'OK'))
    
    async def _svd_get_sharable_params(self, config: datamodel.SVDConfig):
        V = self.repo.get("V")[:, :config.first_n]
        
        if config.to_pc:
            S = self.repo.get("S")[:config.first_n]
            V = V @ np.diag(S)
            
        metadata = self.repo.get("svd_row_metadata")
        
        if metadata is not None:
            label = slice(None) if config.label is None else config.label
            metadata = metadata.loc[:, label]
        else:
            metadata = pd.DataFrame({})
        
        return send_model(
            partial_vec=('JNPArray', V),
            metadata=('PDDataFrame', metadata)
        )
        
    async def _svd_share_results(self, svd_fig_df: pd.DataFrame, config: datamodel.SVDConfig):
        self.output.svd_figures(svd_fig_df, config.label, config.svd_save_dir)
        
        return send_model(status=('str', 'OK'))
    
    async def local_read_matrix_from_csv(self, config: datamodel.TabularReaderConfig):
        meta, A = self.table_reader.local_read_csv(config.file_path, config.meta_cols, config.drop_cols, config.keep_cols)
        print(A)
        self.repo.add("A", A.to_numpy())
        self.repo.add("row_metadata", meta)

        return send_model(status=('str', 'OK'))
    
    async def _X_to_A(self, config: datamodel.PlanBaseConfig):
        self.repo.add("A", self.repo.get("X"))
        return send_model(status=('str', 'OK'))
    

class CompensatorService(ServiceController):
    
    def __init__(self, config_path) -> None:
        super().__init__(config_path)
        if "servers" in self.config.keys():
            self.__servers = self.config["servers"]
            self.__server_ws = net.WSConnectionManager()
        self.__client_ws = net.FastAPIWSConnectionManager()
        self.timeout = self.config["config"]["timeout"] if "timeout" in self.config["config"].keys() else None

    @property
    def node_id(self):
        return self._auth.node_id
    
    @property
    def clients(self):
        return list(self.__clients_ws.active_connections.keys())

    async def run(self):
        for server in self.__servers:
            await self.register(server)

        for server in self.__servers:
            ws = self.__server_ws.get(server["node_id"])
            await self.handle_ws(ws)

    async def handle_request(self, request: datamodel.HTTPRequest) -> Any:
        self._auth.authenticate(request.node_id)
        return await self.eval(request.api, request.args)
        
    async def handle_ws(self, ws) -> Any:
        # Handle for client registration
        if isinstance(ws, WebSocket):
            await ws.accept()
            data = pickle.loads(await ws.receive_bytes())
            registery = datamodel.Registery(**data)
            
            self.logger.info(f"Registered from {registery.role}:{registery.node_id}")
            if registery.role == "client":
                self.__client_ws.add(registery.node_id, ws)
            else:
                raise Exception(f"{registery.role} not found")
            
            try:
                # This is essential to hold this handle_ws function call forever asynchronously,
                # otherwise existing this function makes websocket lost its connection
                while True:
                    await asyncio.Future()  # run forever
            except asyncio.CancelledError:
                pass

        # Handle for server task
        else:
            try:
                while True:
                    data = pickle.loads(await ws.receive_bytes())
                    request = datamodel.RPCRequest(**data)
                    self.logger.info(f"Receive RPC call to {request.api}")
                    result = await self.eval(request.api, request.args)
                    await ws.send_bytes(pickle.dumps(result.model_dump()))
            finally:
                await self.__server_ws.disconnectall()

    async def register(self, profile):
        url = net.construct_url(profile, "/ws/tasks", protocol=profile.get("protocol", "ws"))

        # Corresponds to server ws.accept()
        ws = await self.__server_ws.connect(profile["node_id"], url)

        # Corresponds to server ws.receive_json()
        registery = {"node_id": str(self.node_id), "role": "compensator"}

        await ws.send_bytes(pickle.dumps(registery))
        self.logger.info(f"Register as compensator:{self.node_id}")
        return ws

    async def collect_noise(self, request: datamodel.CompensatorRequest):
        wss = list(map(self.__client_ws.get, request.clients))  # get client ws
        tasks = []
        for i in range(len(wss)):
            tasks.append(asyncio.ensure_future(net.rpc_ws(wss[i], "send_noise", {'request':request.model_dump()}, timeout=self.timeout)))
        client_noise = await asyncio.gather(*tasks)
        
        # Aggregate noise
        noise = {}
        for arg in request.args:
            noise[arg] = ('NPArray', np.sum(list(map(lambda x: x[arg], client_noise)), axis=0))

        return send_model(**noise)
