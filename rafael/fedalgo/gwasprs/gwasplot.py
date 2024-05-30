import logging
from typing import Callable, Optional, Tuple
import numpy as np
import polars as pl
from numpy.typing import NDArray
from scipy import stats
from plotnine import (
    aes,
    element_blank,
    element_line,
    element_text,
    ggplot,
    geom_point,
    geom_hline,
    geom_abline,
    geom_ribbon,
    ggsave,
    guides,
    labs,
    scale_color_manual,
    scale_size_manual,
    scale_x_continuous,
    scale_y_continuous,
    theme_classic,
    theme,
)

from rafael.fedalgo.gwasprs.utils import GWASPlotConst  # don't change to `from .utils` to make python3 gwasplot.py work

def read_glm(
    glm_path: str, chr_col: str, pos_col: str, p_col: str, sep: str
) -> pl.DataFrame:
    """
    Read summary statistics
    Output data frame only have CHR, POS and P columns
    """
    # read and formatting glm
    glm = (
        pl.read_csv(
            glm_path,
            separator=sep,
            dtypes={chr_col: pl.Utf8, pos_col: pl.Int64, p_col: pl.Float64},
            ignore_errors=True,
        )
        .select([chr_col, pos_col, p_col])
        .rename(
            {chr_col: GWASPlotConst.CHR, pos_col: GWASPlotConst.POS, p_col: GWASPlotConst.PVALUE}
        )
        # rename chr in case of special charactor and convert to int
        .with_columns(
            pl.col(GWASPlotConst.CHR)
            .str.replace("chr", "")
            .map_dict(GWASPlotConst.CHR_MAP, default=pl.first())
            .cast(pl.UInt64)
            .alias(GWASPlotConst.CHR)
        )
    )
    raw_snp_count = len(glm)
    logging.info(f"Read {raw_snp_count} snps from glm")
    # filter glm
    glm = glm.filter(
        pl.col(GWASPlotConst.CHR).is_in(GWASPlotConst.CHR_LIST)
        & pl.col(GWASPlotConst.POS).is_null().not_()
        & pl.col(GWASPlotConst.POS).is_nan().not_()
        & (pl.col(GWASPlotConst.POS) > 0)
        & pl.col(GWASPlotConst.PVALUE).is_null().not_()
        & pl.col(GWASPlotConst.PVALUE).is_nan().not_()
        & (pl.col(GWASPlotConst.PVALUE) < 1.0)
        & (pl.col(GWASPlotConst.PVALUE) >= 0.0)
    )
    logging.warning(f"Remove {raw_snp_count - len(glm)} snps")

    return glm


def _get_sig_color(
    glm: pl.DataFrame, p_threds_lower: float, p_threds_upper: float
) -> pl.DataFrame:
    """color odd and even and significant for snp"""
    assert p_threds_upper < p_threds_lower
    glm_adj = glm.with_columns(
        pl.when(pl.col(GWASPlotConst.PVALUE) <= p_threds_upper)
        .then(pl.lit("SIG"))
        .when(pl.col(GWASPlotConst.PVALUE) <= p_threds_lower)
        .then(pl.lit("SUBSIG"))
        .when((pl.col(GWASPlotConst.CHR) % 2) == 0)
        .then(pl.lit("EVEN"))
        .otherwise(pl.lit("ODD"))
        .alias(GWASPlotConst.SNP_COLOR)
    )
    return glm_adj


def format_glm(
    glm: pl.DataFrame,
    p_threds_lower: float = GWASPlotConst.P_LOWER_THRES,
    p_threds_upper: float = GWASPlotConst.P_UPPER_THRES
    ) -> Tuple[pl.DataFrame, float]:
    
    """
    Process the input DataFrame (glm) for both Manhattan and QQ plots.
    Adjusts the P-values to their corresponding -log10 values.
    Sets a maximum threshold for the -log10 values to avoid excessively large values.

    Parameters
    ----------
    glm: pl.DataFrame
        The input DataFrame containing columns "CHR", "POS", "P".
    p_threds_lower: float
        The lower threshold for P-values. Default is 5e-8.
    p_threds_upper: float
        The upper threshold for P-values. Default is 1e-5.

    Returns
    -------
    glm: pl.DataFrame
        The processed DataFrame
    max_log_p: float
        The maximum -log10 value.
    """

    # get max pvalue
    max_log_p = round(
        glm.filter(pl.col(GWASPlotConst.PVALUE) > 0.0)
        .select(pl.col(GWASPlotConst.PVALUE).min().log10() * -1.0)[GWASPlotConst.PVALUE]
        .to_numpy()[0],
        ndigits=0,
    )
    if max_log_p > GWASPlotConst.MAX_LOG_P:
        max_log_p = GWASPlotConst.MAX_LOG_P
    logging.info(
        f"max log_p set to {max_log_p}, snp with pvalue==0 will be set to max_log_p for plot."
    )
    # get log 10 pvalue
    glm = glm.with_columns(
        (pl.col(GWASPlotConst.PVALUE).log10() * -1.0).alias(GWASPlotConst.LOG_P)
    ).with_columns(
        pl.when(
            (pl.col(GWASPlotConst.PVALUE) == 0.0)
            | (pl.col(GWASPlotConst.LOG_P) > max_log_p)
            | pl.col(GWASPlotConst.LOG_P).is_infinite()
        )
        .then(pl.lit(max_log_p))
        .otherwise(pl.col(GWASPlotConst.LOG_P))
        .alias(GWASPlotConst.LOG_P)
    )
    glm = _get_sig_color(glm, p_threds_lower, p_threds_upper)
    return glm, max_log_p


def format_label(format_string: str) -> Callable:
    """create axis lable for ggplot"""

    def _customlabel(x):
        x = [format_string.format(i) for i in x]
        return x

    return _customlabel


def custom_breaks(
    n: int, round_num: Optional[int] = None, max_value: Optional[float] = None
) -> Callable:
    """create axis breaks for ggplot"""

    def _custombreaks(x):
        x = [i for i in x if np.isfinite(i) and not np.isnan(i)]

        if isinstance(max_value, float):
            if x[-1] > max_value:
                x[-1] = max_value

        r = max(x) - min(x)
        breaks = np.linspace(0, max(x) - (0.01) * r, n)
        if round_num is not None:
            breaks = [round(i, round_num) for i in breaks]
        return breaks

    return _custombreaks


def _ppoints(n: int) -> NDArray:
    """
    numpy analogue or `R`'s `ppoints` function
    see details at http://stat.ethz.ch/R-manual/R-patched/library/stats/html/ppoints.html
    """
    a = 3.0 / 8.0 if n <= 10 else 1.0 / 2
    return (np.arange(n) + 1 - a) / (float(n) + 1 - 2 * a)


def _get_ci(glm_qq: pl.DataFrame):
    float_ci = 0.95
    param_list0 = list(range(1, len(glm_qq) + 1))
    param_list1 = param_list0.copy()
    param_list1.reverse()
    ci_upper = stats.beta.ppf(q=(1 + float_ci) / 2, a=param_list0, b=param_list1)
    ci_lower = stats.beta.ppf(q=(1 - float_ci) / 2, a=param_list0, b=param_list1)
    glm_qq = glm_qq.with_columns(
        pl.Series(name=GWASPlotConst.UPPER_CI, values=-1.0 * np.log10(ci_upper)),
        pl.Series(name=GWASPlotConst.LOWER_CI, values=-1.0 * np.log10(ci_lower)),
    )
    return glm_qq


def prepare_qq(glm: pl.DataFrame) -> pl.DataFrame:
    """
    Prepare data for qq plot
    
    Parameters
    ----------
    glm: pl.DataFrame
        The data frame contains columns, "CHR", "POS", "P", "LOG_P" and "SNP_COLOR".

    Returns
    -------
    glm_qq: pl.DataFrame
        The formatted data frame contains columns, "SNP_COLOR", "LOG_P", "QQ" ,"CI_LOWER" and "CI_UPPER".
    """
    # add quantile against pvalue for qqplot
    qq = _ppoints(len(glm))
    glm = glm.sort(GWASPlotConst.PVALUE, descending=False).with_columns(
        pl.Series(name=GWASPlotConst.QQ, values=qq)
    )
    glm_qq = glm.with_columns((pl.col(GWASPlotConst.QQ).log10() * -1.0).alias(GWASPlotConst.QQ))
    # get ci
    glm_qq = _get_ci(glm_qq)
    glm_qq = glm_qq.select(
        [
            GWASPlotConst.SNP_COLOR,
            GWASPlotConst.LOG_P,
            GWASPlotConst.QQ,
            GWASPlotConst.UPPER_CI,
            GWASPlotConst.LOWER_CI,
        ]
    )
    return glm_qq


def plot_qq(glm_qq: pl.DataFrame, output: str):
    p = (
        ggplot(glm_qq)
        + geom_ribbon(
            aes(x=GWASPlotConst.QQ, ymin=GWASPlotConst.LOWER_CI, ymax=GWASPlotConst.UPPER_CI),
            fill="#d5d7e3",
            alpha=0.5,
        )
        + geom_abline(
            intercept=0,
            slope=1,
            linetype=(3, (5, 5)),
            color="red",
            size=0.95,
            alpha=0.7,
        )
        + geom_point(
            aes(x=GWASPlotConst.QQ, y=GWASPlotConst.LOG_P, color=GWASPlotConst.SNP_COLOR),
            alpha=0.99,
        )
        + theme_classic()
        + theme(
            axis_title=element_text(size=15, color="black", alpha=0.87),
            axis_text=element_text(size=14, colour="#7e84a3"),
            axis_line=element_blank(),
            axis_ticks=element_blank(),
            panel_grid_minor_y=element_blank(),
            panel_grid_minor_x=element_blank(),
            panel_grid_major_x=element_blank(),
            panel_grid_major_y=element_line(colour="#f1f1f5", size=2),
        )
        + labs(x=f"Expected: {GWASPlotConst.P_TITLE}", y=f"Observed: {GWASPlotConst.P_TITLE}")
        + guides(fill=False, color=False, size=False)
        + scale_color_manual(
            values=["gray", "gray", "#f2a838", "#f2a838"],
            breaks=["ODD", "EVEN", "SUBSIG", "SIG"],
        )
    )

    ggsave(p, output, height=7, width=9.5, dpi=300)
    logging.info("Successfully plot qqplot!")
    
    
def _adjust_pos(glm_adj: pl.DataFrame) -> pl.DataFrame:
    """adjust pos for plot manhattan"""
    glm_adj = glm_adj.with_columns(
        (pl.col(GWASPlotConst.POS) / 1e6).alias(GWASPlotConst.POS_M)
    ).sort(by=GWASPlotConst.CHR)

    # calculate chr len and cummulated chr len
    glm_agg_chr = (
        glm_adj.group_by(GWASPlotConst.CHR)
        .agg(
            (
                pl.col(GWASPlotConst.POS_M).max()
                - pl.col(GWASPlotConst.POS_M).min()
                + GWASPlotConst.CHR_MARGIN_PAD
            ).alias("CHR_LEN")
        )
        .with_columns(pl.col("CHR_LEN").cumsum().alias("CUM_CHR_LEN"))
    )
    cum_chr_lens = glm_agg_chr["CUM_CHR_LEN"].to_numpy()
    cum_chr_lens = [0.0] + cum_chr_lens.tolist()

    # use the cummulated chr len as the offset to adjust pos
    da_list = []
    for offset, (_chrom, da) in zip(cum_chr_lens, glm_adj.group_by(GWASPlotConst.CHR)):
        da = da.with_columns(
            (
                pl.col(GWASPlotConst.POS_M) - pl.col(GWASPlotConst.POS_M).min() + pl.lit(offset)
            ).alias(GWASPlotConst.POS_A)
        )
        da_list.append(da)
    glm_adj: pl.DataFrame = pl.concat(da_list)

    return glm_adj


def prepare_manhattan(glm: pl.DataFrame) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Prepare data for manhattan plot

    Parameters
    ----------
    glm: pl.DataFrame
        The data frame contains columns, "CHR", "POS", "P", "LOG_P" and "SNP_COLOR".

    Returns
    -------
    glm_adj: pl.DataFrame
        The formatted data frame contains columns, "CHR", "POS_A", "SNP_COLOR", "LOG_P".
    glm_mean_chr: pl.DataFrame
        The data frame records the anchor axis labels.
    """
    glm_adj = _adjust_pos(glm)
    glm_adj = glm_adj.select(
        [GWASPlotConst.CHR, GWASPlotConst.POS_A, GWASPlotConst.SNP_COLOR, GWASPlotConst.LOG_P]
    )
    # get mean for anchor axis label
    glm_mean_chr = glm_adj.group_by(GWASPlotConst.CHR).agg(
        (pl.col(GWASPlotConst.POS_A).max() - pl.col(GWASPlotConst.POS_A).min()) / 2
        + pl.col(GWASPlotConst.POS_A).min()
    )

    return glm_adj, glm_mean_chr


def plot_manhattan(
    glm_adj: pl.DataFrame,
    glm_mean_chr: pl.DataFrame,
    output: str,
    p_lower_thres: float = GWASPlotConst.P_LOWER_THRES,
    p_upper_thres: float = GWASPlotConst.P_UPPER_THRES,
    max_log_p: float = GWASPlotConst.MAX_LOG_P,
):
    p = (
        ggplot(glm_adj)
        + geom_point(
            aes(
                x=GWASPlotConst.POS_A,
                y=GWASPlotConst.LOG_P,
                color=GWASPlotConst.SNP_COLOR,
                size=GWASPlotConst.SNP_COLOR,
            ),
            alpha=0.7,
        )
        + geom_hline(
            yintercept=-1 * np.log10(p_upper_thres),
            linetype=(3, (5, 5)),
            color="#7f84a0",
            size=0.95,
        )
        + geom_hline(
            yintercept=-1 * np.log10(p_lower_thres),
            linetype=(3, (5, 5)),
            color="#7f84a0",
            size=0.95,
            alpha=0.8,
        )
        + theme_classic()
        + theme(
            axis_title=element_text(size=15, color="black", alpha=0.87),
            axis_title_y=element_text(margin={"r": 12}),
            axis_title_x=element_text(margin={"t": 15}),
            axis_text=element_text(size=14, colour="#7e84a3"),
            axis_text_y=element_text(margin={"r": 9}),
            axis_text_x=element_text(margin={"t": 15}),
            axis_line=element_blank(),
            axis_ticks=element_blank(),
            panel_grid_minor_y=element_blank(),
            panel_grid_major_y=element_line(colour="#f1f1f5", size=1),
        )
        + labs(x="chromosome", y=GWASPlotConst.P_TITLE)
        + guides(fill=False, color=False, size=False)
        + scale_y_continuous(
            expand=(0, 0, 0.3, 0),
            breaks=custom_breaks(n=5, round_num=2, max_value=max_log_p * 1.02),
            labels=format_label(format_string="{:.01f}"),
        )
        + scale_x_continuous(
            expand=(0.02, 0, 0.02, 0),
            breaks=glm_mean_chr[GWASPlotConst.POS_A].to_numpy(),
            labels=glm_mean_chr[GWASPlotConst.CHR].to_numpy(),
        )
        + scale_color_manual(
            values=["#5f96ff", "#62e0b8", "#f2a838", "#f99600"],
            breaks=["ODD", "EVEN", "SUBSIG", "SIG"],
        )
        + scale_size_manual(
            values=[0.6, 0.6, 1.2, 1.8], breaks=["ODD", "EVEN", "SUBSIG", "SIG"]
        )
    )

    ggsave(p, output, height=7, width=18, dpi=300)
    logging.info("Successfully plot manhattan!")
    
    
def parse_args():
    import argparse

    parser = argparse.ArgumentParser()

    # required
    parser.add_argument(
        "-i", "--glm", type=str, required=True, help="gwas summay statistics"
    )
    parser.add_argument(
        "-o", "--out_preifx", type=str, required=True, help="output prefix"
    )

    # column name
    parser.add_argument(
        "-C", "--chrom", type=str, default=GWASPlotConst.CHR, help="chrom col name"
    )
    parser.add_argument(
        "-P", "--pos", type=str, default=GWASPlotConst.POS, help="pos col name"
    )
    parser.add_argument(
        "-S", "--stat", type=str, default=GWASPlotConst.PVALUE, help="pvalue col name"
    )
    parser.add_argument(
        "-s", "--sep", type=str, default="\t", help="separator for glm, 1 char"
    )

    # threshold
    parser.add_argument(
        "-u", "--upper_thres", type=float, default=5e-8, help="upper threshold"
    )
    parser.add_argument(
        "-l", "--lower_thres", type=float, default=1e-5, help="lower threshold"
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    logging.getLogger()
    args = parse_args()

    glm = read_glm(
        args.glm,
        args.chrom,
        args.pos,
        args.stat,
        args.sep
    )
    glm, max_logp = format_glm(glm)

    # Manhattan
    manhattan_glm = prepare_manhattan(glm)
    plot_manhattan(
        *manhattan_glm,
        f"{args.out_preifx}.manhattan.png",
        args.lower_thres,
        args.upper_thres,
        max_logp
    )

    # QQ
    qq_glm = prepare_qq(glm)
    plot_qq(qq_glm, f"{args.out_preifx}.qq.png",)
