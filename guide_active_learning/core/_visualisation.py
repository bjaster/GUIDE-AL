from typing import cast, List, Optional, Tuple, TYPE_CHECKING

import graphviz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import seaborn as sns

from guide_active_learning.core._math import cm2inch
from guide_active_learning.core._misc import linfeature_to_str
from guide_active_learning.core.io import save_pdf, save_plot_figure
from guide_active_learning.misc import (
    get_default_result_path,
    make_folder_path,
    make_output_filename,
)

if TYPE_CHECKING:
    from guide_active_learning.GUIDE.guide_tree import TreeNode
__all__ = [
    "plot_df",
    "plot_benchmark_mean_std",
    "tree_to_dot",
    "plot_tree_graph",
]


def get_color(number: int) -> str:
    tableau_colors = [
        "red",
        "blue",
        "green",
        "purple",
        "orange",
        "olive",
        "cyan",
        "gray",
        "brown",
        "pink",
    ]
    return tableau_colors[number % len(tableau_colors)]


def plot_df(data: pd.DataFrame, save_name: Optional[str]) -> None:
    plt.subplots(dpi=100, figsize=(cm2inch(13), cm2inch(9)))
    sns.heatmap(
        data.select_dtypes(float).corr(method="pearson").round(2),
        vmin=-1,
        vmax=1,
        annot=True,
    )

    if save_name is not None:
        save_pdf(filename=save_name)


def plot_benchmark_mean_std(
    data: List[Tuple[str, List[np.ndarray], List[np.ndarray]]],
    dataset_str: str,
    num_datapoints: str,
    save_figure: bool = False,
    dpi: int = 300,
) -> None:
    fig, ax = plt.subplots(dpi=dpi)
    ax2 = ax.twinx()

    for i, (label, mean_scores, std_scores) in enumerate(data):
        ax.plot(mean_scores, label=label, color=get_color(i))
        ax2.plot(
            std_scores,
            label=label,
            color=get_color(i),
            linestyle="dashed",
            alpha=0.2,
        )

    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0, pos.width * 0.9, pos.height * 0.5])
    ax.legend(loc="center", bbox_to_anchor=(0.5, 1.6))

    ax.set_title(f"{dataset_str}-Dataset Number Datapoints: {num_datapoints}")
    ax.set_xlabel("Iteration [-]")
    ax.set_ylabel("Classification Accuracy")
    ax2.set_ylabel("Std(Classification Accuracy)")
    fig.show()

    filename = make_output_filename(
        "benchmark_plot",
        f"initial_datapoints_{num_datapoints}",
        extension=".png",
    )
    if save_figure:
        save_plot_figure("benchmark", filename=filename, fig=fig)


def feature_to_str(tree: "TreeNode", parent_tree: Optional["TreeNode"] = None) -> str:
    if type(tree.threshold) not in [str, tuple]:
        thres = round(cast(float, tree.threshold), 2)
    else:
        thres = cast(float, tree.threshold)

    if type(tree.feature) != list:
        tmp_text = f"{tree.feature}<={thres}"
    else:
        tmp_text = (
            linfeature_to_str(cast(List[np.ndarray], tree.feature)) + f"<={thres}"
        )

    if parent_tree is None:
        return tree.name + ' [label="' + tmp_text + '"] ;\n'
    else:
        dot_text = tree.name + ' [label="' + tmp_text + '"] ;\n'
        dot_text += parent_tree.name + " -> " + tree.name + " [labelangle=-45];\n"
        return dot_text


def label_tree_str(
    tree: "TreeNode", parent_tree: "TreeNode", side: str = "right"
) -> str:
    tmp_text = (
        tree.name + ' [label="' + f'{cast(pd.Series, tree.value).index[0]}"] ; \n'
    )
    if side == "right":
        tmp_text += parent_tree.name + " -> " + tree.name + " [labelangle=45] ;\n"
    else:
        tmp_text += parent_tree.name + " -> " + tree.name + " [labelangle=-45] ;\n"
    return tmp_text


def tree_to_dot(
    tree: Optional["TreeNode"], dot_text: Optional[str] = None, initial: bool = False
) -> str:
    """Helping function for plotting of tree"""
    dot_text = "" if dot_text is None else dot_text
    if tree is None:
        return dot_text

    if tree.feature is not None:
        if initial is True:
            dot_text += feature_to_str(tree)

        if tree.left_tree is not None:
            if tree.left_tree.feature is not None:
                dot_text += feature_to_str(tree.left_tree, parent_tree=tree)
            else:
                dot_text += label_tree_str(tree.left_tree, parent_tree=tree)

        if tree.right_tree is not None:
            if tree.right_tree.feature is not None:
                dot_text += feature_to_str(tree.right_tree, parent_tree=tree)
            else:
                dot_text += label_tree_str(tree.right_tree, parent_tree=tree)

    dot_text = tree_to_dot(tree.left_tree, dot_text=dot_text)  # recursion left
    dot_text = tree_to_dot(tree.right_tree, dot_text=dot_text)  # recursion right

    return dot_text


def plot_tree_graph(*folders: str, text: str, name: str) -> str:
    graphviz_text = (
        'digraph Tree {\nnode [shape=box, fontname="helvetica"] '
        ';\nedge [fontname="helvetica"] ;\n' + text + "}"
    )
    graph = graphviz.Source(graphviz_text, format="png")
    save_path = make_folder_path(get_default_result_path(), *folders, filename=name)
    graph.render(save_path, view=True)
    return graphviz_text
