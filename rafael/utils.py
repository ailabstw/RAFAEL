import os
import io
import zipfile
from functools import reduce
import numpy as np
import pandas as pd

def get_base_dir():
    my_path = os.path.join( os.path.dirname(__file__), "..")
    return os.path.abspath(my_path)


def recur_list_files(dirpath):
    files = []
    for root, _, filenames in os.walk(dirpath):
        for filename in filenames:
            files.append(os.path.join(root, filename))
    return files


def extract_from_zipbytes(content):
    obj = io.BytesIO(content)
    
    with zipfile.ZipFile(obj, "r") as zf:
        extracted = {}
        for filename in zf.namelist():
            with zf.open(filename) as f:
                extracted[filename] = f.read()
    return extracted


class flatten:
    @staticmethod
    def list_of_lists(ls):
        return sum(ls, [])
    
    @staticmethod
    def list_of_dicts(ls):
        return reduce(lambda d1, d2: dict(d1, **d2), ls)
    
    @staticmethod
    def list_of_arrays(ls, axis=0):
        return np.concatenate(ls, axis=axis)
    
    @staticmethod
    def list_of_frames(ls, axis=0):
        return pd.concat(ls, axis=axis)
    
    @staticmethod
    def list_of_blocks(ls):
        block = ls[0]
        for b in ls[1:]:
            block.append(b)
        return block
    