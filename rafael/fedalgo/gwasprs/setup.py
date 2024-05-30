import os
import glob
from ctypes import c_double, c_int32, c_uint32, cdll, POINTER


def setup_plink_hwp():
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plink_hw*so")
    hwp_so_list = glob.glob(base_dir)

    hwp_so = cdll.LoadLibrary(hwp_so_list[0])
    hwp_so.HweP_py.argtypes = [c_int32, c_int32, c_int32, c_uint32]
    hwp_so.HweP_py.restype = c_double
    hwp_so.HweP_vec_py.argtypes = [
        POINTER(c_int32),
        POINTER(c_int32),
        POINTER(c_int32),
        POINTER(c_double),
        c_uint32,
    ]
    return hwp_so


def setup_plink2():
    plink2_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bin/plink2")
    if not os.path.isfile(plink2_path):
        raise RuntimeError(f"executable not found at {plink2_path}")
    return plink2_path
