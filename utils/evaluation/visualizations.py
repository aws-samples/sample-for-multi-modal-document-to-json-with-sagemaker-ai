"""Visualization module for evaluation results."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List
from textwrap import fill
from matplotlib.figure import Figure
from .config import VisualizationConfig

# Distance category mapper used for the stacked bar chart visualization
DIST_CUT_MAPPER = {
    -1: "missing groundtruth",
    0: "exact match",
    1: "1",
    2: "2",
    3: "3",
    4: "4",
    5: "5",
    6: "6+",
    7: "predicted 'None'",
    8: "key missing",
    9: "invalid JSON",
}

# Custom colors matching the original notebook visualization
DIST_CUT_COLORS = [
    "#bababa", "#66c2a5", "#abdda4", "#e6f598", "#ffffbf",
    "#fee08b", "#fdae61", "#f46d43", "#abd9e9", "#74add1", "#bebada",
]


def _prepare_dist_cut(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare dist_cut column from raw dist values for the stacked bar chart."""
    df = df.copy()
    df["dist_cut"] = df["dist"].clip(upper=6)
    df["dist_cut"] = df["dist_cut"].replace(-2, 7)
    df["dist_cut"] = df["dist_cut"].replace(-3, 8)
    df["dist_cut"] = df["dist_cut"].replace(-4, 9)
    df["dist_cut"] = df["dist_cut"].replace(DIST_CUT_MAPPER)
    df["dist_cut"] = pd.Categorical(
        df["dist_cut"], categories=DIST_CUT_MAPPER.values(), ordered=True
    )
    return df


def plot_edit_distance_heatmap(
    df: pd.DataFrame,
    ordered_values: Optional[List[str]] = None,
    width: float = 22,
    height: float = 18,
    geom_vlines: Optional[List[float]] = None,
    config: Optional[VisualizationConfig] = None,
):
    """Stacked bar chart of character edit distance distributions per entity, faceted by model.

    This replicates the plotnine-based visualization from the original evaluation notebook.

    Args:
        df: DataFrame with columns ``entity``, ``dist``, and ``pretty_name``.
            ``dist`` contains raw integer edit distances (0 = exact match,
            negative values encode special cases such as missing ground truth).
        ordered_values: Entity names in the desired display order (bottom to top).
        width: Figure width in inches.
        height: Figure height in inches.
        geom_vlines: X-intercept positions for dashed vertical lines that
            separate entity categories.
        config: Optional ``VisualizationConfig`` (currently unused but kept for
            API consistency).

    Returns:
        A plotnine ``ggplot`` object that can be displayed with ``.show()`` or
        saved with ``.save()``.
    """
    from plotnine import (
        ggplot, aes, geom_bar, facet_wrap, labs, coord_flip,
        scale_fill_manual, theme_minimal, theme, element_rect,
        geom_vline, scale_x_discrete,
    )

    if geom_vlines is None:
        geom_vlines = []

    # If DataFrame is in wide format (entity columns with dist values), melt it
    if "dist" not in df.columns and ordered_values is not None:
        id_vars = [c for c in df.columns if c not in ordered_values]
        df = pd.melt(df, id_vars=id_vars, value_vars=ordered_values, var_name="entity", value_name="dist")
        if "name" in df.columns and "pretty_name" not in df.columns:
            df["pretty_name"] = df["name"]

    # Prepare the dist_cut categorical column
    df_plot = _prepare_dist_cut(df)

    # Sort and create ordered pretty_name for faceting
    df_plot = df_plot.sort_values(["pretty_name", "entity"])
    df_plot["pretty_name_ordered"] = pd.Categorical(
        df_plot["pretty_name"],
        categories=list(dict.fromkeys(df_plot["pretty_name"])),
        ordered=True,
    )

    def wrap_labels(label):
        return fill(label, width=int(width) + 4)

    plot = (
        ggplot(df_plot, aes(x="entity", fill="factor(dist_cut)"))
        + geom_bar(position="stack", color="black")
        + facet_wrap("~pretty_name_ordered", scales="free", labeller=lambda x: wrap_labels(x))
        + labs(x="Entity", y="Count", fill="Char. edit distance")
        + coord_flip()
        + scale_fill_manual(values=DIST_CUT_COLORS, labels=DIST_CUT_MAPPER)
        + theme_minimal()
        + theme(
            figure_size=(width, height),
            panel_background=element_rect(fill="white", color=None),
            plot_background=element_rect(fill="white", color=None),
        )
        + geom_vline(xintercept=geom_vlines, linetype="dashed", color="#4d4d4d", size=2.2)
    )

    if ordered_values is not None:
        plot = plot + scale_x_discrete(limits=ordered_values)

    return plot


def plot_exact_match_comparison(
    df: pd.DataFrame,
    ordered_values: Optional[List[str]] = None,
    geom_vlines: Optional[List[float]] = None,
    width: float = 12,
    height: float = 8,
    config: Optional[VisualizationConfig] = None,
):
    """Horizontal dodged bar chart comparing exact match per entity across models.

    Returns a plotnine ``ggplot`` object.
    """
    from plotnine import (
        ggplot, aes, geom_bar, geom_hline, geom_vline, labs, coord_flip,
        scale_fill_brewer, scale_y_continuous, scale_x_discrete,
        theme_minimal, theme, element_rect, element_text,
    )

    if geom_vlines is None:
        geom_vlines = []

    plot = (
        ggplot(df, aes(x="entity", y="exact_match", group="model", fill="model"))
        + geom_bar(stat="identity", position="dodge", colour="gray")
        + labs(title="Exact match (higher better)", x="Entity", y="exact_match", fill="model_name")
        + coord_flip()
        + scale_fill_brewer(type="qual", palette="Set3")
        + theme_minimal()
        + theme(
            figure_size=(width, height),
            panel_background=element_rect(fill="white", color=None),
            plot_background=element_rect(fill="white", color=None),
            axis_text_x=element_text(angle=45, hjust=1),
        )
        + scale_y_continuous(
            breaks=[x / 100.0 for x in range(0, 101, 5)],
            labels=[f"{i}%" for i in range(0, 101, 5)],
        )
        + geom_hline(yintercept=[x / 100.0 for x in range(0, 101, 5)], color="darkgray", size=0.5, alpha=0.5)
        + geom_vline(xintercept=geom_vlines, linetype="dashed", color="#4d4d4d", size=2.2)
    )

    if ordered_values is not None:
        plot = plot + scale_x_discrete(limits=ordered_values)

    return plot


def plot_metric_distribution(
    df: pd.DataFrame,
    metric: str,
    ordered_values: Optional[List[str]] = None,
    geom_vlines: Optional[List[float]] = None,
    width: float = 12,
    height: float = 8,
    config: Optional[VisualizationConfig] = None,
):
    """Horizontal dodged bar chart for a given metric per entity across models.

    Returns a plotnine ``ggplot`` object.
    """
    from plotnine import (
        ggplot, aes, geom_bar, geom_vline, labs, coord_flip,
        scale_fill_brewer, scale_x_discrete,
        theme_minimal, theme, element_rect, element_text,
    )

    if geom_vlines is None:
        geom_vlines = []

    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not found in DataFrame")

    plot = (
        ggplot(df, aes(x="entity", y=metric, group="model", fill="model"))
        + geom_bar(stat="identity", position="dodge", color="gray")
        + labs(title=f"{metric} - Character edits for entities (lower better)", fill="model_name")
        + coord_flip()
        + scale_fill_brewer(type="qual", palette="Set3")
        + theme_minimal()
        + theme(
            figure_size=(width, height),
            panel_background=element_rect(fill="white", color=None),
            plot_background=element_rect(fill="white", color=None),
            axis_text_x=element_text(angle=45, hjust=1),
        )
        + geom_vline(xintercept=geom_vlines, linetype="dashed", color="#4d4d4d", size=2.2)
    )

    if ordered_values is not None:
        plot = plot + scale_x_discrete(limits=ordered_values)

    return plot


def plot_category_performance(
    df: pd.DataFrame,
    categories: dict,
    metric: str = "exact_match",
    width: float = 12,
    height: float = 8,
    config: Optional[VisualizationConfig] = None,
):
    """Bar chart of a metric grouped by feature category, with a bar per model.

    Args:
        df: DataFrame with columns ``entity``, ``model``, and the chosen *metric*.
        categories: Mapping of entity name → category string.
        metric: Column name to plot (e.g. ``exact_match``, ``cer_score``).

    Returns:
        A plotnine ``ggplot`` object.
    """
    from plotnine import (
        ggplot, aes, geom_bar, labs, coord_flip,
        scale_fill_brewer,
        theme_minimal, theme, element_rect, element_text,
    )

    df = df.copy()
    df["category"] = df["entity"].map(lambda e: str(categories.get(e, "unknown")))

    grouped = df.groupby(["category", "model"])[metric].mean().reset_index()

    metric_labels = {
        "exact_match": "exact_match (higher is better)",
        "cer_score": "cer_score (lower is better)",
    }
    y_label = metric_labels.get(metric, metric)

    plot = (
        ggplot(grouped, aes(x="category", y=metric, fill="model"))
        + geom_bar(stat="identity", position="dodge", colour="gray")
        + labs(title=f"Performance by Feature Category ({metric})", x="Feature Category", y=y_label, fill="model_name")
        + coord_flip()
        + scale_fill_brewer(type="qual", palette="Set3")
        + theme_minimal()
        + theme(
            figure_size=(width, height),
            panel_background=element_rect(fill="white", color=None),
            plot_background=element_rect(fill="white", color=None),
            axis_text_x=element_text(angle=45, hjust=1),
        )
    )

    return plot


def plot_model_ranking(
    comparison_df: pd.DataFrame,
    config: Optional[VisualizationConfig] = None
) -> Figure:
    """Overall model ranking visualization."""
    if config is None:
        config = VisualizationConfig()
    
    fig = Figure(figsize=config.figsize)
    ax = fig.add_subplot(111)
    
    sns.heatmap(
        comparison_df,
        annot=True,
        fmt='.3f',
        cmap='RdYlGn',
        center=0.5,
        ax=ax,
        cbar_kws={'label': 'Score'}
    )
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Metric')
    ax.set_title('Model Performance Comparison')
    
    fig.tight_layout()
    return fig


def plot_null_percentage_by_entity(stats_df: pd.DataFrame):
    """Horizontal bar chart of null percentage per entity using plotnine.

    Args:
        stats_df: DataFrame with ``entity`` and ``null_percentage`` columns,
            e.g. built from :func:`categorization.analyze_feature_distribution`.

    Returns:
        A plotnine ``ggplot`` object.
    """
    from plotnine import ggplot, aes, geom_bar, labs, coord_flip, theme_bw

    plot_data = stats_df.sort_values("null_percentage", ascending=True).copy()
    plot_data["entity"] = pd.Categorical(
        plot_data["entity"], categories=plot_data["entity"].tolist(), ordered=True
    )
    return (
        ggplot(plot_data, aes(x="entity", y="null_percentage"))
        + geom_bar(stat="identity", fill="steelblue")
        + coord_flip()
        + labs(title="Null Percentage by Entity", x="Entity", y="Null Percentage")
        + theme_bw()
    )


def _remove_outliers(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Remove outliers using IQR method."""
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    return df[(df[col] >= q1 - 1.5 * iqr) & (df[col] <= q3 + 1.5 * iqr)]


def plot_length_boxplots(
    df: pd.DataFrame,
    entities: Optional[List[str]] = None,
    log_scale: bool = True,
) -> Figure:
    """Horizontal boxplots of entity string lengths with statistics table.

    Replicates ``plot_length_boxplots_with_limits_horizontal`` from the
    original ``utils.entities`` module but works with the raw DataFrame
    format (``labels`` dict column).

    Args:
        df: DataFrame with a ``labels`` dict column.
        entities: Entity names to include. If *None*, uses all keys found.
        log_scale: Use log scale for the length axis.

    Returns:
        A matplotlib ``Figure``.
    """
    # Extract lengths from labels dict
    rows = []
    for _, row in df.iterrows():
        if not isinstance(row["labels"], dict):
            continue
        for entity, val in row["labels"].items():
            if entities and entity not in entities:
                continue
            is_null = val is None or str(val) in ("None", "null", "", "nan", "NaN")
            rows.append({"entity": entity, "length": None if is_null else len(str(val))})

    df_len = pd.DataFrame(rows)

    # Stats on original data
    stats_orig = df_len.groupby("entity")["length"].agg(
        total_count="count", null_count=lambda x: x.isnull().sum()
    )

    # Remove nulls, then outliers per entity
    df_clean = df_len.dropna(subset=["length"]).copy()
    df_no_outliers = pd.concat(
        [_remove_outliers(g, "length") for _, g in df_clean.groupby("entity")],
        ignore_index=True,
    )

    # Stats on clean data
    stats_clean = df_no_outliers.groupby("entity")["length"].agg(
        clean_count="count",
        mean=lambda x: f"{x.mean():.1f}",
        median=lambda x: f"{x.median():.1f}",
        std=lambda x: f"{x.std():.1f}",
        min=lambda x: f"{x.min():.1f}",
        max=lambda x: f"{x.max():.1f}",
    )

    stats = stats_orig.join(stats_clean)
    stats["outliers_removed"] = stats["total_count"] - stats["null_count"] - stats["clean_count"].astype(float)
    stats = stats[["total_count", "null_count", "outliers_removed", "clean_count", "mean", "median", "std", "min", "max"]]

    # Plot
    fig = Figure(figsize=(15, 15))
    gs = fig.add_gridspec(2, 1)

    ax1 = fig.add_subplot(gs[0])
    sns.boxplot(data=df_no_outliers, y="entity", x="length", color="skyblue", ax=ax1)
    sns.stripplot(data=df_no_outliers, y="entity", x="length", color="navy", alpha=0.2, size=4, jitter=0.2, ax=ax1)
    if log_scale:
        ax1.set_xscale("log")
    ax1.set_title("Response Length Distribution by Entity (Outliers Removed)", fontsize=12, pad=20)
    ax1.set_ylabel("Entity", fontsize=10)
    ax1.set_xlabel("Length (log scale)" if log_scale else "Length", fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[1])
    table = ax2.table(
        cellText=stats.values, rowLabels=stats.index, colLabels=stats.columns,
        cellLoc="center", loc="center", bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    ax2.axis("off")

    fig.tight_layout()
    return fig


def plot_null_statistics(
    null_df: pd.DataFrame,
    metric: str = "null_in_predictions_pct",
    ordered_values: Optional[List[str]] = None,
    width: float = 12,
    height: float = 8,
    config: Optional[VisualizationConfig] = None,
):
    """Horizontal dodged bar chart of null/missing statistics per entity across models.

    Args:
        null_df: DataFrame from ``compute_null_statistics`` with a ``model`` column.
            Concatenate results from multiple models before passing.
        metric: Column to plot. Common choices:
            ``null_in_predictions_pct``, ``missing_key_pct``, ``null_in_labels_pct``.
        ordered_values: Entity names in desired display order.

    Returns:
        A plotnine ``ggplot`` object.
    """
    from plotnine import (
        ggplot, aes, geom_bar, labs, coord_flip,
        scale_fill_brewer, scale_x_discrete,
        theme_minimal, theme, element_rect, element_text,
    )

    metric_labels = {
        "null_in_predictions_pct": "Null in Predictions (%)",
        "missing_key_pct": "Missing Keys in Predictions (%)",
        "null_in_labels_pct": "Null in Labels (%)",
    }
    y_label = metric_labels.get(metric, metric)

    fill_col = "model" if "model" in null_df.columns else None

    # For prediction nulls, add ground truth null rate as a reference bar
    if metric == "null_in_predictions_pct" and fill_col and "null_in_labels_pct" in null_df.columns:
        gt_rows = null_df.drop_duplicates("entity")[["entity", "null_in_labels_pct"]].copy()
        gt_rows["model"] = "Ground Truth (null in labels)"
        gt_rows = gt_rows.rename(columns={"null_in_labels_pct": metric})
        null_df = pd.concat([null_df[["entity", metric, "model"]], gt_rows], ignore_index=True)

    if fill_col:
        plot = (
            ggplot(null_df, aes(x="entity", y=metric, fill="model"))
            + geom_bar(stat="identity", position="dodge", colour="gray")
            + labs(title=y_label, x="Entity", y=y_label, fill="model_name")
        )
    else:
        plot = (
            ggplot(null_df, aes(x="entity", y=metric))
            + geom_bar(stat="identity", colour="gray")
            + labs(title=y_label, x="Entity", y=y_label)
        )

    plot = (
        plot
        + coord_flip()
        + scale_fill_brewer(type="qual", palette="Set3")
        + theme_minimal()
        + theme(
            figure_size=(width, height),
            panel_background=element_rect(fill="white", color=None),
            plot_background=element_rect(fill="white", color=None),
            axis_text_x=element_text(angle=45, hjust=1),
        )
    )

    if ordered_values is not None:
        plot = plot + scale_x_discrete(limits=ordered_values)

    return plot
