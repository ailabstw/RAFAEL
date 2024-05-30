import os
from typing import Dict, List
import subprocess
import logging
import jax

from .setup import setup_plink2

def jax_dev_count() -> int:
    return jax.device_count()


PLINK2_PATH = setup_plink2()


def bash(command, *args, **kargs):
    PopenObj = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=os.setsid,
        shell=True,
        executable="/bin/bash",
        *args,
        **kargs,
    )
    out, err = PopenObj.communicate()
    out = out.decode("utf8").rstrip("\r\n").split("\n")
    err = err.decode("utf8").rstrip("\r\n").split("\n")
    if PopenObj.returncode != 0:
        logging.error("command failed")
        logging.error(command)
        for i in err:
            logging.error(i)
        raise RuntimeError
    return out, err


class GWASPlotConst:
    MASK_LIV_LIST: List[str] = ["matplotlib", "plotnine"]
    CHR_MAP: Dict[str, str] = {"MT": "26", "X": "23", "Y": "24", "XY": "23", "25": "23"}
    CHR_LIST: List[int] = [i for i in range(1, 27)]

    # column names
    CHR: str = "CHR"
    POS: str = "POS"
    PVALUE: str = "PVALUE"
    LOG_P: str = "LOG_P"
    SNP_COLOR: str = "SNP_COLOR"

    # statistics
    MAX_LOG_P: float = 200.0
    P_LOWER_THRES: float = 1e-5
    P_UPPER_THRES: float = 5e-8
    P_TITLE: str = "-log$_{10}$($\it{p}$value)"

    # QQ
    QQ: str = "QQ"
    UPPER_CI: str = "UPPER_CI"
    LOWER_CI: str = "LOWER_CI"

    # Manhattan
    POS_M: str = "POS_M"  # pos in million
    POS_A: str = "POS_A"  # adjusted pos
    CHR_MARGIN_PAD: int = 40  # for x axis offset
