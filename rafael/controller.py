from types import MappingProxyType

from .logger import setup_logger
from . import configurations


class AbstractController:
    def __init__(self, config_path: str) -> None:
        self.__config = MappingProxyType(configurations.load_yml(config_path))
        self.__logger = setup_logger(log_path=self.__config['config']['log_path'])

    @property
    def config(self):
        return self.__config

    @property
    def logger(self):
        return self.__logger
