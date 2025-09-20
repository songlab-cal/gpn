import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import roc_auc_score, average_precision_score

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import AutoMinorLocator
from scipy.stats import fisher_exact


HGVS = ['chrom', 'pos', 'ref', 'alt']

def stratified_bootstrap_se(y_true, y_score,
                            n_bootstraps=1000, seed=42,
                            return_mean=False):
    """
    Stratified bootstrap CI for a performance metric:
      - y_true:      array of 0/1 labels
      - y_score:     array of scores (higher ⇒ positive)
      - n_bootstraps: # of bootstrap resamples
    Returns (mean, lower, upper) of the bootstrap distribution.
    """
    rng = np.random.RandomState(seed)
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    n_pos, n_neg = len(pos_idx), len(neg_idx)

    boot_scores_auroc = []
    boot_scores_auprc = []
    for _ in range(n_bootstraps):
        # sample positives *with* replacement
        samp_pos = rng.choice(pos_idx,  size=n_pos, replace=True)
        # sample negatives *with* replacement
        samp_neg = rng.choice(neg_idx,  size=n_neg, replace=True)
        # combine
        idx = np.concatenate([samp_pos, samp_neg])
        boot_scores_auroc.append(roc_auc_score(y_true[idx], y_score[idx]))
        boot_scores_auprc.append(average_precision_score(y_true[idx], y_score[idx]))

    boot_scores_auroc = np.array(boot_scores_auroc)
    boot_scores_auprc = np.array(boot_scores_auprc)
    
    if return_mean:
        return (
            np.std(boot_scores_auroc, ddof=1), np.std(boot_scores_auprc, ddof=1), 
            np.mean(boot_scores_auroc), np.mean(boot_scores_auprc),
        )
    else:
        return np.std(boot_scores_auroc, ddof=1), np.std(boot_scores_auprc, ddof=1)

def barplot(df, metric, palette, figsize=(2,2.25), pos_prop=None, title=None, save_path=None):
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"] #
    plt.rcParams["font.weight"] = 'regular'
    plt.rcParams["ytick.color"] = "black" #'#474747'
    plt.rcParams["xtick.color"] = "black" #'#474747'

    plt.rcParams["mathtext.fontset"] = "dejavuserif"
    plt.rcParams["mathtext.default"] = "regular"
    plt.rcParams["mathtext.sf"] = "dejavusans"

    plt.figure(figsize=figsize)
    sns.barplot(
        data=df,
        y="Model",
        x=metric,
        palette=palette,
        #edgecolor='#474747', #'#777777', 
        #linewidth=0.5,
    )
    sns.despine();
    plt.title(title, fontsize=11, fontweight='light', color='black');
    baseline = 0.5 if metric == "AUROC" else pos_prop
    ax = plt.gca()
    #ax.axhline(baseline, ls='--', color="grey")
    #plt.xticks(rotation=45, ha="right")
    #limit = min(baseline, results_clinvar[metric].min()) - 0.01
    limit = baseline
    
    # Check if metric values are tuples/lists and extract first element if so
    metric_values = df[metric].apply(lambda x: x[0] if isinstance(x, (tuple, list)) else x)
    se_values = df[f"{metric}_se"].apply(lambda x: x[0] if isinstance(x, (tuple, list)) else x)
    
    max_metric = float(metric_values.max())
    max_se = float(se_values.max())
    
    plt.gca().set_xlim(left=limit, right=max_metric + max_se)
    
    # Add minor ticks with reduced frequency
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))  # Increase divisor to make minor ticks sparser
    
    # Make major ticks longer and thicker than minor ticks
    ax.tick_params(axis='x', which='major', length=6)
    ax.tick_params(axis='x', which='minor', length=4, width=0.5)
    
    ax.errorbar(
            x=metric_values,
            y=df["Model"],
            xerr=se_values,
            fmt='none',      # Do not add markers (those are already in the pointplot)
            color='#474747', #"#777777",
            linewidth=1.5,
    )
    
    # for p in ax.patches:
    #     ax.text(max(p.get_width(), baseline),  # X position, here at the end of the bar
    #             p.get_y() + p.get_height()/2,  # Y position, in the middle of the bar
    #             f'{p.get_width():.3f}',  # Text to be displayed, formatted to 3 decimal places
    #             va='center'  # Vertical alignment
    #             )
    plt.ylabel("");
    if save_path is not None:
        plt.savefig('./' + save_path, bbox_inches="tight")

def get_pos_prop(subtitle):
    n_pos = int(subtitle.split('=')[1].split(' ')[0])
    n_neg = int(subtitle.split('=')[1].split(' ')[-1])
    return n_pos/(n_pos+n_neg), n_pos, n_neg

def get_odds_ratio(df, threshold_ns):
    rows = []
    negative_set = df.filter(~pl.col("label")).sort("score")
    for n in threshold_ns:
        threshold = negative_set[n]["score"]
        group_counts = (
            df.group_by(["label", pl.col("score") <= threshold]).len()
            .sort(["label", "score"])["len"].to_numpy().reshape((2,2))
        )
        odds_ratio, p_value = fisher_exact(group_counts, alternative='greater')
        rows.append([n, odds_ratio, p_value])
    # in updated polars, might need to add orient="row"
    return pl.DataFrame(rows, schema=["n", "Odds ratio", "p_value"], orient="row")

def format_number(num):
    """
    Converts a number into a more readable format, using K for thousands, M for millions, etc.
    Args:
    - num: The number to format.
    
    Returns:
    - A formatted string representing the number.
    """
    if num >= 1e9:
        return f'{num/1e9:.1f}B'
    elif num >= 1e6:
        return f'{num/1e6:.1f}M'
    elif num >= 1e3:
        return f'{num/1e3:.1f}K'
    else:
        return str(num)

def barplot_with_numbers(
    df, metric, title, palette, groupby="Consequence",
    width=2, height=2, nrows=1, ncols=1,
    save_path=None, wspace=None, hspace=None,
    x=None, y=None, 
):

    plt.rcParams["mathtext.fontset"] = "dejavuserif"
    plt.rcParams["mathtext.default"] = "regular"
    plt.rcParams["mathtext.sf"] = "dejavusans"

    if groupby not in df.columns: df[groupby] = "all"
    groups = df[groupby].unique()
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, sharex=False, sharey=False,
        figsize=(width*ncols, height*nrows), squeeze=False,
        gridspec_kw={'wspace': wspace, 'hspace': hspace},
    )

    for group, ax in zip(groups, axes.flat):
        df_g = df[df[groupby]==group].sort_values(metric, ascending=False)
        n_pos, n_neg = df_g.n_pos.iloc[0], df_g.n_neg.iloc[0]

        if metric == "AUROC":
            baseline = 0.5
        elif metric == "AUPRC":
            baseline = n_pos / (n_pos + n_neg)
        elif metric == "Odds ratio":
            baseline = 1

        g = sns.barplot(
            data=df_g,
            y="Model",
            x=metric,
            palette=palette,
            ax=ax,
        )
        sns.despine()
        sample_size = f"n={format_number(n_pos)} vs. {format_number(n_neg)}"
        subtitle = f"{group}\n{sample_size}" if len(groups) > 1 else sample_size
        g.set_title(subtitle, fontsize=10, fontweight="light")
        g.set(xlim=baseline, ylabel="")

        for bar, model in zip(g.patches, df_g.Model):
            if metric == "Odds ratio":
                text = f'{bar.get_width():.1f}'
                if df_g[df_g.Model==model].p_value.iloc[0] >= 0.05:
                    text = text + " (NS)"
            else:
                text = f'{bar.get_width():.3f}'
            
            g.text(
                max(bar.get_width(), baseline)+0.01,  # X position, here at the end of the bar
                bar.get_y() + bar.get_height()/2,  # Y position, in the middle of the bar
                text,  # Text to be displayed, formatted to 3 decimal places
                va='center',  # Vertical alignment
                fontweight='light'
            )

        # Add minor ticks with reduced frequency
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))  # Increase divisor to make minor ticks sparser
        
        # Make major ticks longer and thicker than minor ticks
        ax.tick_params(axis='x', which='major', length=6)
        ax.tick_params(axis='x', which='minor', length=4, width=0.5)

        #if metric == "Odds ratio":
        #    for index, row in df_g.iterrows():
        #        if row['p_value'] >= 0.05:
        #            g.text(y=index, x=row['Odds ratio'], s='(ns)', ha='right')
        
    plt.suptitle(title, x=x, y=y, fontsize=11, fontweight="light")
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")


def barplot_vertical_aligned(
    df, metric, title, palette, groupby="Consequence",
    width=4, height=2, hspace=0.1, right_margin=0.8, suptitle_y=0.96,
    save_path=None, model_order=None, group_order=None,
):
    
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"] #
    plt.rcParams["font.weight"] = 'regular'
    plt.rcParams["ytick.color"] = '#474747'
    plt.rcParams["xtick.color"] = '#474747'
    plt.rcParams["mathtext.fontset"] = "dejavuserif"
    plt.rcParams["mathtext.default"] = "regular"
    plt.rcParams["mathtext.sf"] = "dejavusans"
    
    # 1) ensure group column
    if groupby not in df.columns:
        df[groupby] = "all"
    if group_order is None:
        groups = list(df[groupby].unique())
    else:
        groups = group_order
    n_panels = len(groups)

    # 2) fixed model order
    if model_order is None:
        model_order = list(dict.fromkeys(df['Model']))

    # 3) constant baseline for OR/AUROC
    if metric == "AUROC":
        const_baseline = 0.5
    elif metric == "Odds ratio":
        const_baseline = 1.0
    else:
        const_baseline = None

    # 4) make figure & axes
    fig, axes = plt.subplots(
        n_panels, 1,
        sharex=False,
        figsize=(width, height * n_panels),
        gridspec_kw={'hspace': hspace}
    )
    fig.subplots_adjust(right=right_margin)

    for ax, group in zip(axes, groups):
        # a) subset, drop NA & duplicates, reorder
        df_g = (
            df[df[groupby] == group]
            .dropna(subset=[metric])
            .drop_duplicates(subset='Model', keep='first')
            .set_index('Model')
            .reindex(model_order)
            .dropna(subset=[metric])
            .reset_index()
        )

        n_pos, n_neg = df_g.n_pos.iloc[0], df_g.n_neg.iloc[0]
        baseline = (
            (n_pos / (n_pos + n_neg)) if metric == "AUPRC"
            else const_baseline
        )

        # b) draw vertical bars
        sns.barplot(
            data=df_g,
            x="Model", y=metric,
            order=model_order,
            palette=palette,
            ax=ax
        )
        sns.despine(ax=ax, bottom=True)

        # # c) baseline line
        # if baseline is not None:
        #     ax.axhline(baseline, ls='--', color='gray', lw=1)

        # d) adaptive y‑limits per panel
        # y_min = min(df_g[metric].min(), baseline) if baseline is not None else df_g[metric].min()
        # y_max = df_g[metric].max()
        # margin = (y_max - y_min) * 0.1 if y_max != y_min else y_max * 0.1
        # ax.set_ylim(y_min - margin, y_max + margin)

        max_val = df_g[metric].max()
        upper = round(max_val)# / 5) * 5
        ax.set_ylim(1, max_val)
        ax.set_yticks([1, upper])
        # e) annotate bars
        # for bar, model in zip(ax.patches, df_g.Model):
        #     val = bar.get_height()
        #     if metric == "Odds ratio" and df_g.loc[df_g.Model == model, "p_value"].iloc[0] >= 0.05:
        #         txt = f"{val:.1f} (NS)"
        #     else:
        #         fmt = ".1f" if metric == "Odds ratio" else ".3f"
        #         txt = f"{val:{fmt}}"
        #     ax.text(
        #         bar.get_x() + bar.get_width()/2,
        #         max(val, baseline if baseline is not None else 0),
        #         txt,
        #         ha='center', va='bottom', fontsize=8
        #     )

        # f) vertical subtitle on right
        sample_size = f"n={format_number(n_pos)} vs. {format_number(n_neg)}"
        subtitle = f"{group}\n{sample_size}" if n_panels > 1 else sample_size
        ax.text(
            1.05, 0.5,
            subtitle,
            transform=ax.transAxes,
            ha='left', va='center',
            #rotation=90,
            rotation_mode='anchor',
            fontsize=12,
            fontweight='light',
            clip_on=False
        )

        # g) x‑tick labels: only bottom panel, tilted
        # remove all bottom‐ticks
        #ax.tick_params(axis='x', which='both', bottom=False)
        
        if ax is axes[-1]:
            ax.tick_params(axis='x', rotation=45, labelsize=12)
            for lbl in ax.get_xticklabels():
                lbl.set_ha('right')
        else:
            ax.tick_params(labelbottom=False, bottom=False)

        # remove individual y‑labels
        ax.set_ylabel("")
        ax.set_xlabel("")

        # Add minor ticks with reduced frequency
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))  # Increase divisor to make minor ticks sparser
        
        # Make major ticks longer and thicker than minor ticks
        ax.tick_params(axis='y', which='major', length=6)
        ax.tick_params(axis='y', which='minor', length=4, width=0.5)

    # 5) shared labels & title
    fig.supylabel('Odds ratio', fontsize=13, **{"x":-0.3})
    fig.suptitle(title, y=suptitle_y, fontsize=15, fontweight='light')

    # super‐tight layout
    fig.tight_layout(h_pad=0.1, rect=[0, 0, right_margin, suptitle_y])

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    plt.show()

def barplot_vertical(
    df, metric, title, palette, groupby="Consequence", 
    width=2, height=2, nrows=1, ncols=1,
    save_path=None, wspace=None, hspace=None,
    x=None, y=None,
):
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["font.weight"] = 'regular'
    plt.rcParams["ytick.color"] = "black"
    plt.rcParams["xtick.color"] = "black"

    plt.rcParams["mathtext.fontset"] = "dejavuserif"
    plt.rcParams["mathtext.default"] = "regular"
    plt.rcParams["mathtext.sf"] = "dejavusans"

    if groupby not in df.columns: df[groupby] = "all"
    groups = df[groupby].unique()
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, sharex=False, sharey=False,
        figsize=(width*ncols, height*nrows), squeeze=False,
        gridspec_kw={'wspace': wspace, 'hspace': hspace},
    )

    for group, ax in zip(groups, axes.flat):
        df_g = df[df[groupby]==group].sort_values(metric, ascending=False)
        n_pos, n_neg = df_g.n_pos.iloc[0], df_g.n_neg.iloc[0]

        if metric == "AUROC":
            baseline = 0.5
        elif metric == "AUPRC":
            baseline = n_pos / (n_pos + n_neg)
        elif metric == "Odds ratio":
            baseline = 1

        g = sns.barplot(
            data=df_g,
            x="Model",
            y=metric,
            palette=palette,
            ax=ax,
        )
        sns.despine()
        sample_size = f"n={format_number(n_pos)} vs. {format_number(n_neg)}"
        subtitle = f"{group}\n{sample_size}" if len(groups) > 1 else sample_size
        g.set_title(subtitle, fontsize=10, fontweight="light", pad=15)
        g.set(ylim=baseline, xlabel="")

        # Add padding between y-axis and first bar
        ax.margins(x=0.1)

        # Rotate x-axis labels for better readability
        plt.setp(g.get_xticklabels(), rotation=45, ha='right')

        for bar, model in zip(g.patches, df_g.Model):
            if metric == "Odds ratio":
                text = f'{bar.get_height():.1f}'
                if df_g[df_g.Model==model].p_value.iloc[0] >= 0.05:
                    text = text + " (NS)"
            else:
                text = f'{bar.get_height():.3f}'
            
            g.text(
                bar.get_x() + bar.get_width()/2,  # X position, center of bar
                max(bar.get_height(), baseline)+0.01,  # Y position, above the bar
                text,  # Text to be displayed
                ha='center',  # Horizontal alignment
                va='bottom',  # Vertical alignment
                fontweight='light',
                rotation=0
            )
        
        # Add minor ticks with reduced frequency
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        
        # Make major ticks longer and thicker than minor ticks
        ax.tick_params(axis='y', which='major', length=6)
        ax.tick_params(axis='y', which='minor', length=4, width=0.5)

    plt.suptitle(title, x=x, y=y, fontsize=11, fontweight="light")
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")