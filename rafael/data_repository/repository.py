import abc
import os
import tempfile


class AbstractRepository(abc.ABC):

    @abc.abstractmethod
    def add(self, object):
        raise NotImplementedError

    @abc.abstractmethod
    def get(self, id):
        raise NotImplementedError


class InMemoryRepository(AbstractRepository):

    def __init__(self) -> None:
        super().__init__()
        self.__entities = {}

    def add(self, id: str, object):
        self.__entities[id] = object
        return id

    def get(self, id, value=None):
        return self.__entities.get(id, value)
        
    def pop(self, id, value=None):
        if value is None:
            return self.__entities.pop(id)
        else:
            return self.__entities.pop(id, value)
            
    def __getitem__(self, id):
        return self.__entities[id]

    def list(self):
        return list(self.__entities.keys())


class FileSystemRepository(AbstractRepository):

    def __init__(self, root_path) -> None:
        super().__init__()
        self.__root_path = root_path
        self.__filenames = os.listdir(root_path)

    def add(self, filepath: str):
        filename = filepath.replace(self.__root_path, "")
        self.__filenames.append(filename)

    def get(self, filename: str):
        # regex match?
        return os.path.join(self.__root_path, filename)

    def list(self):
        return self.__filenames


class TemporaryDirectoryRepository(AbstractRepository):

    def __init__(self) -> None:
        super().__init__()
        self.__tempdir = tempfile.TemporaryDirectory()
        self.__root_path = self.__tempdir.name
        self.__filenames = os.listdir(self.__tempdir.name)

    def add(self, filepath: str):
        filename = filepath.replace(self.__root_path, "")
        self.__filenames.append(filename)

    def get(self, filename: str):
        # regex match?
        return os.path.join(self.__root_path, filename)

    def list(self):
        return self.__filenames

    def cleanup(self):
        self.__tempdir.cleanup()


class TabularRepository(AbstractRepository):

    def __init__(self) -> None:
        super().__init__()
        self.__filenames = []

    def add(self, object):
        return super().add(object)

    def get(self, id):
        return super().get(id)

    def list(self):
        return self.__filenames
