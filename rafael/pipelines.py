from typing import Any, List
import inspect

from rafael import usecases
from rafael.controller import AbstractController


class PipelineController(AbstractController):

    def __init__(self, config_path: str) -> None:
        super().__init__(config_path)
        self.__pipeline = generate_pipeline(self.config)
        self.__output_names = generate_returns(self.config["pipeline"])

    @property
    def pipeline(self):
        return self.__pipeline

    def __call__(self) -> Any:
        outputs = {}
        method_idx = 0
        usecase_idx = 0
        pipeline_list = self.config["pipeline"]

        while method_idx != len(self.pipeline):
            # Eval instance method
            uc_obj, method = self.pipeline[method_idx]
            args = match_arguments(method, **self.config["config"], **outputs)
            new_outputs = eval(
                uc_obj,
                method,
                self.__output_names[method_idx],
                *args
            )
            outputs.update(new_outputs)

            # Prevent jump to the previous same usecase method
            # Index change if usecase changes
            if pipeline_list[method_idx][0] != pipeline_list[usecase_idx][0]:
                usecase_idx = method_idx

            if 'jump_to' in new_outputs.keys() and new_outputs['jump_to'] != 'next':
                current_usecase = pipeline_list[method_idx][0]
                jump_to_method = new_outputs['jump_to']
                method_idx = jump_to(pipeline_list, usecase_idx, current_usecase, jump_to_method)
            else:
                method_idx += 1


def generate_pipeline(config: dict):
    pipeline = []
    if "pipeline" not in config.keys():
        return pipeline
    for (uc, method) in config["pipeline"]:
        uc_inst = getattr(usecases, uc)
        meth_inst = getattr(uc_inst, method)
        pipeline.append((uc_inst, meth_inst))
    return pipeline

def match_arguments(func, **kwargs):
    argspec = inspect.getfullargspec(func)
    argnames = argspec.args[1:] if "self" in argspec.args else argspec.args
    return [kwargs.get(argname, None) for argname in argnames]

def eval(uc_obj, method, output_names, *args):
    output_vals = method(uc_obj, *args)
    if len(output_names) == 1:
        outputs = {output_names[0]: output_vals}
    else:
        outputs = {n: v for (n, v) in zip(output_names, output_vals)}
    return outputs

def generate_returns(pipeline: List[str]):
    returns = []
    for (uc, method) in pipeline:
        uc_inst = getattr(usecases, uc)
        meth_inst = uc_inst.return_variables(method)
        returns.append(meth_inst)
    return returns

def jump_to(pipelines, current_uc_idx, to_uc, to_func):
    to_idx = pipelines[current_uc_idx:].index((to_uc, to_func)) + current_uc_idx
    return to_idx