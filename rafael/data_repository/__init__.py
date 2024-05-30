from .repository import InMemoryRepository, FileSystemRepository, TabularRepository, TemporaryDirectoryRepository
import os

DEFAULT_FILE_STORAGE = "/Users/yuehhua/workspace/gwasfl/demo_data"

MEMORY_REPO = InMemoryRepository()
# FILE_REPO = FileSystemRepository(DEFAULT_FILE_STORAGE)
FILE_REPO = TemporaryDirectoryRepository()
TABLE_REPO = TabularRepository()


def list():
    global MEMORY_REPO
    global TABLE_REPO
    global FILE_REPO
    return MEMORY_REPO.list() + TABLE_REPO.list() + FILE_REPO.list()


def load(id: str):
    _, extension = os.path.splitext(id)

    if extension == "":
        global MEMORY_REPO
        return MEMORY_REPO.get(id)
    elif extension in [".csv", ".tsv"]:
        global TABLE_REPO
        return None
    else:
        global FILE_REPO
        return FILE_REPO.get(id)


def save(id: str, object):
    _, extension = os.path.splitext(id)

    if extension == "":
        global MEMORY_REPO
        return MEMORY_REPO.add(id, object)
    elif extension in [".csv", ".tsv"]:
        global TABLE_REPO
        return None
    else:
        global FILE_REPO
        return FILE_REPO.add(id, object)
