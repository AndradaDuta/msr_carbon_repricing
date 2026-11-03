"""Step 8: Figures 1 and 2 of the thesis, drawn from the firm-year carbon frame.

Input:   data/processed/carbon_firm_year.parquet
Outputs: output/figures/fig_carbon_raw_vs_log.pdf, fig_carbon_within_year_z.pdf
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.common import CARBON, load_config, output_dir, processed_path

INK, RAW_COLOR, LOG_COLOR = "#1a1a1a", "#5b6b7a", "#2f4a63"
YEAR_SHADES = ["#9db4c7", "#4f7191", "#22384c"]
STYLE = {"font.family": "serif", "font.size": 10, "axes.linewidth": 0.8,
         "xtick.direction": "out", "ytick.direction": "out", "xtick.major.width": 0.8,
         "ytick.major.width": 0.8, "savefig.dpi": 300, "figure.dpi": 150}


def _clean(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=3.5)


def _kde(ax, data, grid, color, label):
    density = stats.gaussian_kde(data)(grid)
    ax.plot(grid, density, color=color, lw=1.6, label=label)
    ax.fill_between(grid, density, color=color, alpha=0.10)


def figure_raw_vs_log(carbon, path):
    raw = carbon["carbon_intensity"].to_numpy()
    logged = np.log(raw)
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.4, 3.5))

    cap = np.quantile(raw, 0.99)  # clip the top 1% for a readable raw panel
    left.hist(raw[raw <= cap], bins=45, color=RAW_COLOR, edgecolor="white", linewidth=0.3)
    left.set_xlabel("Scope-1 emissions / revenue")
    left.set_ylabel("Firm-years")
    left.set_title("Raw intensity", fontsize=10.5, pad=6)
    left.text(0.96, 0.92, f"skew = {stats.skew(raw):.1f}", transform=left.transAxes,
              ha="right", va="top", fontsize=8.5, color="#555")
    left.text(0.96, 0.83, "top 1% clipped\nfor display", transform=left.transAxes,
              ha="right", va="top", fontsize=7.5, color="#888")
    _clean(left)

    right.hist(logged, bins=45, density=True, color=LOG_COLOR, edgecolor="white",
               linewidth=0.3, alpha=0.85)
    grid = np.linspace(logged.min(), logged.max(), 300)
    right.plot(grid, stats.norm.pdf(grid, logged.mean(), logged.std()), color=INK,
               lw=1.3, ls="--", label="Normal fit")
    right.set_xlabel("log(Scope-1 / revenue)")
    right.set_ylabel("Density")
    right.set_title("Log intensity", fontsize=10.5, pad=6)
    right.text(0.96, 0.92, f"skew = {stats.skew(logged):.2f}", transform=right.transAxes,
               ha="right", va="top", fontsize=8.5, color="#555")
    right.legend(frameon=False, fontsize=8.3, loc="upper left")
    _clean(right)

    fig.suptitle("Distribution of carbon intensity", fontsize=10.5, y=1.02)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def figure_within_year(carbon, path):
    years = np.sort(carbon["year"].unique())
    sample_years = [int(years[0]), int(years[len(years) // 2]), int(years[-1])]
    shade = dict(zip(sample_years, YEAR_SHADES))
    logged = np.log(carbon["carbon_intensity"])
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.6, 3.5))

    grid = np.linspace(logged.quantile(0.005), logged.quantile(0.995), 400)
    for year in sample_years:
        _kde(left, logged[carbon["year"] == year].to_numpy(), grid, shade[year], str(year))
    left.set_xlabel("log(Scope-1 / revenue)")
    left.set_ylabel("Density")
    left.set_title("A. Log intensity, by year", fontsize=10.5, pad=6, loc="left")
    left.legend(frameon=False, fontsize=8.5, title="Year", title_fontsize=8.5)
    _clean(left)

    grid = np.linspace(-4, 4, 400)
    for year in sample_years:
        _kde(right, carbon.loc[carbon["year"] == year, CARBON].to_numpy(), grid, shade[year], str(year))
    right.plot(grid, stats.norm.pdf(grid), color=INK, lw=1.2, ls="--", label="N(0,1)")
    right.set_xlabel("Within-year z-score of log intensity")
    right.set_ylabel("Density")
    right.set_title("B. After within-year standardisation", fontsize=10.5, pad=6, loc="left")
    right.legend(frameon=False, fontsize=8.5)
    _clean(right)

    fig.suptitle("Within-year cross-sectional standardisation of carbon intensity",
                 fontsize=10.5, y=1.02)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def run(cfg):
    mpl.rcParams.update(STYLE)
    carbon = pd.read_parquet(processed_path(cfg, "carbon"))
    figures = output_dir(cfg, "figures")
    figure_raw_vs_log(carbon, figures / "fig_carbon_raw_vs_log.pdf")
    figure_within_year(carbon, figures / "fig_carbon_within_year_z.pdf")
    print(f"[8] wrote Figures 1, 2 -> {figures}")


if __name__ == "__main__":
    run(load_config())
