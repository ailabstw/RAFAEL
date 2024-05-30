from rafael.controller import AbstractController
from fedalgo.gwasprs.gwasdata import GWASDataIterator

class DatasetController(AbstractController):
    def __init__(self, config_path: str) -> None:
        super().__init__(config_path)
        self.__usecases = parse_usecase(self.config)
        self.__dataloaders = parse_dataloader(self.config)
        
    @property
    def configs(self):
        return self.__dataloaders
    
    @property
    def usecases(self):
        return self.__usecases
    
    def __getitem__(self, loader_key):
        if isinstance(loader_key, int):
            return self.__dataloaders[loader_key]
        elif isinstance(loader_key, str):
            _, loader_config = self._match_loader_by_name(loader_key)
            return loader_config
        
    def _match_loader_by_name(self, loader_name):
        matched_loaders = list(filter(lambda x: x[0] == loader_name, self.__dataloaders))
        assert len(matched_loaders) == 1, f'Loader name {loader_name} must be unique in config while using name indexing.'
        return matched_loaders[0]
    
    def _get_loader(self, loader_config, **kwargs):
        # bfile_arg can be either "bfile_path" or "filtered_bfile_path".
        # Similar to the covariates.
        bfile_arg = loader_config['bfile_arg']
        cov_arg = loader_config['cov_arg']
        
        # Finad bfile_path and cov_path from kwargs and config.
        kwargs = {**self.config['config'], **kwargs}
        bfile_path = kwargs[bfile_arg]
        cov_path = kwargs.get(cov_arg, None)  # Allow cov_path to be None.
        
        loader = GWASDataIterator(
            bfile_path=bfile_path,
            cov_path=cov_path,
            style=loader_config['style'],
            sample_step=loader_config['sample_chunk_size'],
            snp_step=loader_config['snp_chunk_size'],
        )
        return loader
    
    def get_loader_from_name(self, loader_name, **kwargs):
        _, loader_config = self._match_loader_by_name(loader_name)
        return self._get_loader(loader_config, **kwargs)
        
    def get_loader_from_idx(self, loader_idx, **kwargs):
        _, loader_config = self.__dataloaders[loader_idx]
        return self._get_loader(loader_config, **kwargs)
        
    
def parse_usecase(config):
    usecases = []
    if 'pipeline' in config:
        for uc, _ in config['pipeline']:
            if (uc not in usecases) or (usecases[-1] != uc):
                usecases.append(uc)
    return usecases
    

def parse_dataloader(config):
    """
    Parse dataloader config.
    
    Parameters
    ----------
        config : dict
            Dictionary of usecases and dataloader configs.
    Returns
    -------
        loader_configs : list of tuples
            The tuple stores the dataloader name and its parameters.
    Notes
    -----
        The full spec of the dataloader config is as follows:
        bfile_arg : str, default='bfile_path'
            Sepecify what parameter used to create a dataloader.
            Default value is 'bfile_path' loaded from `config` in .yml.
        cov_arg : str, default='cov_path'
            Sepecify what parameter used to create a dataloader.
            Default value is 'cov_path' loaded from `config` in .yml.
            If it's missing, it will be set to None when creating a dataloader.
        style : str, default='snp'
            The split style of the dataloader. See also `GWASDataIterator`.
        sample_chunk_size : int, default=123456789
            The sample chunk size of the dataloader.
        snp_chunk_size : int, default=123456789
            The SNP chunk size of the dataloader.
        start_idx : int, default=0
            The dataloader is responsible for providing data starting from the `start_idx` + 1th use case.
            Default value is 0, meaning the first use case.
        end_idx : int, default=None
            The dataloader is responsible for providing data ending to the `end_idx`th use case.
            Default value is None, meaning the last use case.
    """
    usecases = parse_usecase(config)
    loader_configs = []
    pseudo_loader = [('whole_loader', {})]
    for loader, params in config.get('dataloader', pseudo_loader):
        loader_config = (
            loader,
            {
                'bfile_arg': params.get('bfile_arg', 'bfile_path'),
                'cov_arg': params.get('cov_arg', 'cov_path'),
                'style': params.get('style', 'snp'),
                'sample_chunk_size': params.get('sample_chunk_size', 123456789),
                'snp_chunk_size': params.get('snp_chunk_size', 123456789),
                'start_idx': params.get('start_idx', 0),
                'end_idx': params.get('end_idx', None),
                'usecases': usecases[params.get('start_idx', 0):params.get('end_idx', None)]
            }
        )
        loader_configs.append(loader_config)
    return loader_configs
        
if __name__ == "__main__":
    obj = DatasetController('/volume/jianhung-fa/rafael/configs/dataloader.yml')
    print(obj.configs)
    print(obj.usecases)
    print('--------------------------------')
    print(obj[0])
    print(obj.get_loader_from_idx(0))
    print(obj[1])
    print(obj.get_loader_from_idx(1, **{'filtered_bfile_path': '/tmp/agg'}))
    print(obj.get_loader_from_name('loader2', **{'filtered_bfile_path': '/tmp/agg'}))
    