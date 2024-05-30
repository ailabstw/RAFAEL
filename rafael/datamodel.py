from typing import List, Dict, Optional, Union, Any, Annotated
import inspect

from pydantic import BaseModel, create_model
from pydantic.functional_validators import AfterValidator
from pydantic.functional_serializers import PlainSerializer
import numpy as np
import pandas as pd
import jax.numpy as jnp
        
        
def construct_argument(func, kwargs):
    # TODO: Refactor this
    # This function implementation is so unstable
    # It relates to the server handling request from dashboard
    # and client handling request from the server
    
    argspec = inspect.getfullargspec(func)
    
    # list of argument names
    api_args = argspec.args
    if 'self' in api_args: api_args.remove('self')

    matched_args = {}
    for arg in api_args:
        try:
            matched_args[arg] = argspec.annotations[arg](**kwargs[arg])
        except:
            matched_args[arg] = argspec.annotations[arg](kwargs[arg])

    return None if len(matched_args) == 0 else matched_args

def send_model(**kwargs):
    """
    Generate a generic response facilitating easy data reception by the server.

    This function is inspired by concepts like 'Dynamic Model Creation', 'Annotated Validators', and 'Custom Serializers'
    in the `pydantic` documentation.

    1. Creating a BaseModel Response:
        This function creates a BaseModel Response by specifying a BaseModel name and mapping argument names to tuples
        containing argument types and values. For example:
        `create_model('foo', a=(int, 1), b=(str, '2'), c=(int, ...))`, where '...' denotes the argument type 'c' as 'int'
        without default value. Refer to 'Dynamic Model Creation' in the pydantic documentation for more details.

    2. Auto-serialize and deserialize numpy arrays:
        To enhance code clarity and readability, this function provides automatic serialization and deserialization
        for numpy arrays (can be extended as well). It defines serializer and validator functions in advance. For instance:
        `NPArray = Annotated[Any, PlainSerializer(lambda x: x.tolist(), return_type=list), AfterValidator(to_np)]`.
        This enables clients to serialize arrays effortlessly and servers to deserialize them without manual intervention.
        Refer to 'Annotated Validators' and 'Custom Serializers' in the pydantic documentation for more details.

    Parameters
    ----------
    kwargs : dict
        A dictionary mapping argument names to tuples of argument types and values.

    Returns
    -------
    AbcResponse: pydantic.BaseModel
        A pydantic.BaseModel with the specified input and an additional argument 'spec' to store all the argument types.
        'spec' is a dictionary mapping argument names to their respective argument types, facilitating reconstruction
        of the pydantic.BaseModel during server reception.
        
    Examples
    --------
        >>> import numpy as np
        >>> from rafael.datamodel import send_model, recv_response
        
        # np.ndarray
        >>> response = send_model(X=('NPArray', np.random.randn(2,3)))
        
        >>> response
        AbcResponse(spec={'X': 'NPArray'}, X=array([[ 0.45919943, -1.03635007,  0.5340125 ],
        [ 0.22415387, -0.25618103,  1.09914376]]))
        
        >>> response.model_dump()
        {'spec': {'X': 'NPArray'},
        'X': [[0.4591994251844278, -1.036350073040202, 0.5340124952215379],
        [0.22415386933819403, -0.2561810280500033, 1.0991437590428985]]}  # The array is auto-serialized to list.
        
        >>> recv_response(**response.model_dump())
        AbcResponse(X=array([[ 0.45919943, -1.03635007,  0.5340125 ],
        [ 0.22415387, -0.25618103,  1.09914376]]))
        
        # jnp.array
        >>> response = send_model(y=('JNPArray', jnp.array([1,2])))
        
        >>> response
        AbcResponse(spec={'y': 'JNPArray'}, y=Array([1, 2], dtype=int32))
        
        >>> response.model_dump()
        {'spec': {'y': 'JNPArray'}, 'y': [1, 2]}  # The array is auto-serialized to list.
        
        >>> recv_response(**response.model_dump())
        AbcResponse(y=Array([1, 2], dtype=int32))
        
        # Integrate with other types
        >>> response = send_model(Xs=('List[NPArray]', [np.random.randn(1,2), np.random.randn(1,2)]))
        
        >>> response
        AbcResponse(spec={'Xs': 'List[NPArray]'}, Xs=[array([[1.39115389, 0.48908983]]), array([[-1.12535307, -1.60642833]])])
        
        >>> response.model_dump()
        {'spec': {'Xs': 'List[NPArray]'},
        'Xs': [[[-1.487762755756832, 0.7095977411970219]],
        [[1.8033636098392538, 0.37932282456185173]]]}
        
        >>> recv_response(**response.model_dump())
        AbcResponse(Xs=[array([[1.39115389, 0.48908983]]), array([[-1.12535307, -1.60642833]])])

    See Also
    --------
    - Dynamic Model Creation: https://docs.pydantic.dev/latest/concepts/models/#dynamic-model-creation
    - Annotated Validators: https://docs.pydantic.dev/latest/concepts/validators/#annotated-validators
    - Custom Serializers: https://docs.pydantic.dev/latest/concepts/serialization/#custom-serializers
    """
    if len(kwargs) == 0:
        return create_model('AbcResponse', spec=(dict, {}))()
    
    # 'arg1', 'arg2', 'arg3'
    argnames = kwargs.keys()
    
    # Extracting argument types and values
    # argtypes: 'list', 'str', 'NPArray'...
    # argvalues: [1,2,3], '1', np.array([1,2])
    argtypes, argvalues = list(zip(*kwargs.values()))
    
    # Creating annotations for arguments. argannos: {'arg1':(list, ...), 'arg2': (NPArray, ...), ...}
    argannos = {arg:(eval(ann), ...) for arg, (ann, _) in kwargs.items()}
    
    # kwargs: {'arg1':[1,2,3], 'arg2':'1', 'arg3':np.array([1,2])}
    kwargs = dict(zip(argnames, argvalues))

    return create_model('AbcResponse', spec=(dict, dict(zip(argnames, argtypes))), **argannos)(**kwargs)

def recv_model(**kwargs):
    """
    Receiving a response and reconstructing the `pydantic.BaseModel`.

    Parameters
    ----------
    kwargs : dict
        A dictionary containing received response data.

    Returns
    -------
    AbcResponse: pydantic.BaseModel
        Reconstructed `pydantic.BaseModel` object with received data.
    """
    spec = kwargs.pop('spec')
    params = {arg: (eval(spec.get(arg)), value) for arg, value in kwargs.items()}
    return create_model('AbcResponse', **params)(**kwargs)

def to_np(x):
    return np.array(x)

def to_jnp(x):
    return jnp.array(x)

def to_pd(x):
    return pd.DataFrame(x)

# Follow the pydantic documentation 'Custom Serializers' and 'Annotated Validators'
NPArray = Annotated[Any, PlainSerializer(lambda x: x.tolist(), return_type=list), AfterValidator(to_np)]
JNPArray = Annotated[Any, PlainSerializer(lambda x: x.tolist(), return_type=list), AfterValidator(to_jnp)]
PDDataFrame = Annotated[Any, PlainSerializer(lambda x: x.to_dict(), return_type=dict), AfterValidator(to_pd)]

# Request/response data model

class Registery(BaseModel):
    node_id: str
    role: str


class RPCRequest(BaseModel):
    api: str
    args: dict


class HTTPRequest(BaseModel):
    node_id: str
    api: str
    args: dict


class FileRequest(BaseModel):
    node_id: str
    clients: List[str]
    files: List[str]


class Status(BaseModel):
    status: str


class PlanBaseConfig(BaseModel):
    clients: Union[List[str], str]
    compensators: Optional[Union[List[str], str]] = None
    cc_map: Optional[dict] = None
    
    def to_client_config(self, idx):
        args = self.__dict__.copy()
        args.update(
            {
                'clients': self._list2str(self.clients, idx),
                'compensators': self._map_compensators(self.clients[idx], self.cc_map, self.compensators)
            }
        )
        return eval(self.__class__.__name__)(**args)
    
    @staticmethod
    def _list2str(arg, idx):
        return arg[idx] if isinstance(arg, list) else arg
    
    @staticmethod
    def _map_compensators(client, cc_map, compensators):
        if cc_map is None:
            return compensators
        else:
            return cc_map[client]


class GWASConfig(PlanBaseConfig):
    # These are for the files
    bfile_path: Union[List[str], str]
    cov_path: Optional[Union[List[str], str]] = None
    pheno_path: Optional[Union[List[str], str]] = None
    pheno_name: Optional[Union[List[str], str]] = 'pheno'
    regression_save_dir: Union[List[str], str]
    local_qc_output_path: Union[List[str], str]
    global_qc_output_path: str
    
    # Logistic sepcific
    logistic_max_iters: int = 16
    
    # File preprocessing
    impute_cov: Optional[bool] = False
    autosome_only: Optional[bool] = True
    
    # These are for the QC
    maf: Optional[float] = 0.05
    geno: Optional[float] = 0.02
    hwe: Optional[float] = 5e-7
    mind: Optional[float] = 0.02
    
    # These are for the PCA (SVDConfig)
    k1: Optional[int] = 20
    k2: Optional[int] = 20
    svd_max_iters: Optional[int] = 20
    epsilon: Optional[float] = 1e-9
    first_n: Optional[int] = 4
    to_pc: Optional[bool] = False
    label: Optional[Union[List[str], str]] = None
    svd_save_dir: Optional[Union[List[str], str]] = None  # default is local_qc_output_path
    
    # System parameters
    block_size: Optional[int] = 10000
    num_core: Optional[int] = 4
    snp_chunk_size: Optional[int] = 10000

    def client_bfile_config(self):
        return {
            'bfile_path': self.bfile_path,
            'cov_path': self.cov_path,
            'pheno_path': self.pheno_path,
            'pheno_name': self.pheno_name,
            'autosome_only': self.autosome_only
        }
    
    def to_client_config(self, idx):
        args = self.__dict__.copy()
        args.update(
            {
                'clients'   : self._list2str(self.clients,    idx),
                'bfile_path': self._list2str(self.bfile_path, idx),
                'cov_path'  : self._list2str(self.cov_path,   idx),
                'pheno_path': self._list2str(self.pheno_path, idx),
                'pheno_name': self._list2str(self.pheno_name, idx),
                'regression_save_dir':  self._list2str(self.regression_save_dir,  idx),
                'local_qc_output_path': self._list2str(self.local_qc_output_path, idx),
                'svd_save_dir': self._list2str(self.local_qc_output_path, idx) if self.svd_save_dir is None \
                    else self._list2str(self.svd_save_dir, idx)
            }
        )
        return GWASConfig(**args)


class CompensatorRequest(BaseModel):
    clients: Union[List[str], str]
    args: Union[List[str], str]
    

class ClinicalBaseConfig(PlanBaseConfig):
    feature_cols: Optional[List[str]] = None
    clinical_data_path: Union[List[str], str]
    meta_cols: Optional[Union[List[str], str]] = None
    save_dir: Union[List[str], str]
    
    def to_client_config(self, idx):
        args = self.__dict__.copy()
        args.update(
            {
                'clients': self._list2str(self.clients, idx),
                'clinical_data_path': self._list2str(self.clinical_data_path, idx),
                'save_dir': self._list2str(self.save_dir, idx)
            }
        )
        return eval(self.__class__.__name__)(**args)


class CoxPHRegressionConfig(ClinicalBaseConfig):
    r: Optional[int] = 100
    k: Optional[int] = 20
    bs_prop: Optional[float] = 0.6
    bs_times: Optional[int] = 20
    alpha: Optional[float] = 0.05
    step_size: Optional[float] = 0.5


class KaplanMeierConfig(ClinicalBaseConfig):
    alpha: Optional[float] = 0.05
    n_std: Optional[Union[List[float], float]] = 1.0


class SVDConfig(PlanBaseConfig):
    k1: Optional[int] = 20
    k2: Optional[int] = 20
    svd_max_iters: Optional[int] = 20
    epsilon: Optional[float] = 1e-9
    first_n: Optional[int] = 4
    to_pc: Optional[bool] = False
    label: Optional[Union[List[str], str]] = None
    svd_save_dir: Union[List[str], str]
    
    def to_client_config(self, idx):
        args = self.__dict__.copy()
        args.update(
            {
                'clients': self._list2str(self.clients, idx),
                'svd_save_dir': self._list2str(self.svd_save_dir, idx)
            }
        )
        return SVDConfig(**args)
    

class TabularReaderConfig(PlanBaseConfig):
    file_path: Union[List[str], str]
    meta_cols: Optional[Union[List[str], str]] = None
    drop_cols: Optional[Union[List[str], str]] = None
    keep_cols: Optional[Union[List[str], str]] = None
    
    def to_client_config(self, idx):
        args = self.__dict__.copy()
        args.update(
            {
                'clients': self._list2str(self.clients, idx),
                'file_path': self._list2str(self.file_path, idx)
            }
        )
        return TabularReaderConfig(**args)


class TabularDataSVDConfig(TabularReaderConfig, SVDConfig):
    def to_client_config(self, idx):
        args = self.__dict__.copy()
        args.update(
            {
                'clients': self._list2str(self.clients, idx),
                'file_path': self._list2str(self.file_path, idx),
                'svd_save_dir': self._list2str(self.svd_save_dir, idx)
            }
        )
        return TabularDataSVDConfig(**args)
