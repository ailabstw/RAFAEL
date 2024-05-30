import urllib.request
import zipfile
import os
import sys
import platform
import subprocess
import stat

from setuptools import Extension
from setuptools_cpp import ExtensionBuilder

PLINK2_FILENAME = "plink.zip"
PLINK2_DST = "rafael/fedalgo/gwasprs/bin/"
PLINK2_BIN = os.path.join(PLINK2_DST, "plink2")
MODE = stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH

def check_platform():
    os = sys.platform
    if os == "linux":
        isavx2 = subprocess.run(['grep', 'avx2', '/proc/cpuinfo'], stdout=subprocess.PIPE, stderr=subprocess.PIPE).returncode == 0
        isarm = None
    elif os == "darwin":
        isavx2 = subprocess.run(['sysctl', '-a'], stdout=subprocess.PIPE, stderr=subprocess.PIPE).returncode == 0
        isarm = platform.machine() == 'arm64'
    elif os == "win32":
        # TODO: to be tested on Windows
        isavx2 = None
        isarm = None
    is64bit = platform.architecture()[0] == '64bit'
    return os, isavx2, is64bit, isarm

def platform_plink2_url():
    os, isavx2, is64bit, isarm = check_platform()
    # linux
    if os == "linux" and isavx2:
        return "https://s3.amazonaws.com/plink2-assets/alpha5/plink2_linux_avx2_20240526.zip"
    elif os == "linux" and is64bit:
        return "https://s3.amazonaws.com/plink2-assets/alpha5/plink2_linux_x86_64_20240526.zip"
    elif os == "linux":
        return "https://s3.amazonaws.com/plink2-assets/alpha5/plink2_linux_i686_20240526.zip"
    
    # mac
    elif os == "darwin" and isarm:
        return "https://s3.amazonaws.com/plink2-assets/alpha5/plink2_mac_arm64_20240526.zip"
    elif os == "darwin" and isavx2:
        return "https://s3.amazonaws.com/plink2-assets/alpha5/plink2_mac_avx2_20240526.zip"
    elif os == "darwin" and is64bit:
        return "https://s3.amazonaws.com/plink2-assets/alpha5/plink2_mac_20240526.zip"
    
    # windows
    elif os == "win32":
        return "https://s3.amazonaws.com/plink2-assets/alpha5/plink2_win64_20240526.zip"
    else:
        raise ValueError(f"Unsupported platform: {os}")

def download_plink2(plink2_url, plink2_filename, plink2_dst):
    if not os.path.exists(plink2_dst):
        os.makedirs(plink2_dst)
    
    urllib.request.urlretrieve(plink2_url, plink2_filename)
    with zipfile.ZipFile(plink2_filename, 'r') as zip_ref:
        zip_ref.extractall(plink2_dst)
    
    os.remove(plink2_filename)

def build(setup_kwargs):
    if not os.path.exists(PLINK2_BIN):
        download_plink2(platform_plink2_url(), PLINK2_FILENAME, PLINK2_DST)
        os.chmod(PLINK2_BIN, MODE)
    
    setup_kwargs.update(
        {
            "ext_modules": [
                Extension(
                    "rafael.fedalgo.gwasprs.plink_hwp",
                    sources=["rafael/fedalgo/externals/plink_hwp.cc"],
                    library_dirs=["rafael/fedalgo/externals"],
                ),
            ],
            "cmdclass": {"build_ext": ExtensionBuilder},
            "package_data": {
                "rafael.fedalgo.gwasprs": ["bin/plink2"]
            }
        }
    )
