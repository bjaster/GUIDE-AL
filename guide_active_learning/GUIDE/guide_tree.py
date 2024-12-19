from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from guide_active_learning.core import (
    calculate_column_beta,
    calculate_column_gamma,
    compute_information_gain,
    compute_lda,
    plot_tree_graph,
    tree_to_dot,
)
from guide_active_learning.GUIDE.guide_analysis import sf_main_effect
from guide_active_learning.GUIDE.guide_misc import sf_interaction, split_dataframe
from guide_active_learning.GUIDE.split_points import (
    sp_interaction_features,
    sp_main_feature,
)
from guide_active_learning.GUIDE.splitting import linear_split, univariate_split
from scipy.stats import chi2

pd.options.mode.chained_assignment = None
# Setzen des Pfades zu den Graphviz-Exe-Dateien
os.environ["PATH"] += os.pathsep + "C:/Program Files/Graphviz/bin"
# Suppress NumPy RuntimeWarning
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")

__all__ = [
    "TreeNode",
    "DecisionTreeClassifierGUIDE",
    "create_tree_plot",
]


@dataclass
class TreeNode:
    name: str
    dataset: pd.DataFrame
    depth: int
    feature: Optional[Union[List[str], str, np.ndarray]] = None
    threshold: Optional[Union[str, List[str], float]] = None
    value: Optional[pd.Series] = None
    left_tree: Optional[TreeNode] = None
    right_tree: Optional[TreeNode] = None
    adapted: bool = False
    runs: int = 0
    pool: Optional[pd.DataFrame] = pd.DataFrame()
    unl_pool: Optional[pd.DataFrame] = pd.DataFrame()
    metrics: Optional[pd.DataFrame] = pd.DataFrame()

    def __post_init__(self) -> None:
        self.gini = self.compute_gini_node(self.dataset["target"])

    @staticmethod
    def compute_gini_node(target_feature: pd.Series) -> float:
        return 1 - sum((target_feature.value_counts() / len(target_feature)) ** 2)


@dataclass
class DecisionTreeClassifierGUIDE:
    train_df: Optional[pd.DataFrame] = None
    target: str = "target"
    target_expression: Optional[pd.DataFrame] = None
    max_depth: int = 5
    adapted: bool = False
    split_type: str = "mixed"  # mixed,linear,univariat
    use_linear_split: bool = True
    min_samples_split: int = 2
    min_info_gain: float = 0.05
    root: Optional[TreeNode] = None
    value: int = 0
    num_split_quantity: Optional[int] = None

    def __post_init__(self) -> None:
        if self.split_type == "univariat":
            self.use_linear_split = False

    def fit(self, df: pd.DataFrame, target: str) -> None:
        self.train_df = df
        self.target = target
        self.target_expression = df[self.target].unique()
        self.root = self._build_tree(df, depth=0, name="0")

    def _label_TreeNode(self, df: pd.DataFrame, name: str, depth: int) -> TreeNode:
        return TreeNode(
            name=name,
            value=df[self.target].value_counts(normalize=True),
            dataset=df,
            depth=depth,
            adapted=self.adapted,
        )

    def _build_tree(self, df: pd.DataFrame, *, depth: int, name: str) -> TreeNode:
        if (
            len(df[self.target].value_counts()) == 1
            or (len(df.drop(columns=self.target).columns) == 1)
            or (depth > self.max_depth)
            or (len(df) <= self.min_samples_split)
        ):
            return self._label_TreeNode(df, name, depth)
        else:
            self.value += 1
            split_feature = self._select_split_feature(df_tmp=df, target=self.target)
            if split_feature is None:
                return self._label_TreeNode(df, name, depth)

            split_feature, split_point = self._select_split_point(
                df_tmp=df, feature=split_feature, target=self.target
            )
            split_point = split_point[-1]
            df1, df2 = split_dataframe(
                df=df,
                split_point=split_point,
                split_feature=split_feature,
                target=self.target,
            )

            # recursion
            if (
                compute_information_gain(df_par=df, df_c1=df1, df_c2=df2)
                < self.min_info_gain
            ):
                return self._label_TreeNode(df, name, depth)

            else:
                node_left = self._build_tree(df1, depth=depth + 1, name=name + str(1))
                node_right = self._build_tree(df2, depth=depth + 1, name=name + str(2))
                return TreeNode(
                    name=name,
                    feature=split_feature,
                    threshold=split_point,
                    left_tree=node_left,
                    right_tree=node_right,
                    dataset=df,
                    depth=depth,
                    adapted=self.adapted,
                )

    def _select_split_feature(
        self, df_tmp: pd.DataFrame, target: str
    ) -> Optional[Union[str, List[str], np.ndarray]]:
        non_constant_predictors = list(df_tmp.columns[df_tmp.nunique() > 1])
        if target not in non_constant_predictors:
            non_constant_predictors.append(target)

        df = df_tmp[non_constant_predictors]
        if len(non_constant_predictors) == 1 and "target" in non_constant_predictors:
            return None

        len_columns = len(df.drop(columns=target).columns)
        len_number_columns = len(
            df.drop(columns=target).select_dtypes(include="number").columns
        )

        alpha = 0.05 / len_columns

        main_effect = sf_main_effect(df=df, target=target)
        if max(main_effect.values()) > chi2.isf(alpha, 1) or len_columns == 1:
            return max(main_effect, key=cast(Callable[[str], int], main_effect.get))
        else:
            interaction = sf_interaction(df, target)
            if self.use_linear_split:
                return linear_split(
                    df=df,
                    target=target,
                    interaction=interaction,
                    main_effect=main_effect,
                    len_number_columns=len_number_columns,
                    beta=calculate_column_beta(len_columns),
                    gamma=calculate_column_gamma(len_number_columns),
                )

            else:
                return univariate_split(
                    df=df,
                    target=target,
                    interaction=interaction,
                    main_effect=main_effect,
                    beta=calculate_column_beta(len_columns),
                )

    def _select_split_point(
        self,
        df_tmp: pd.DataFrame,
        feature: Union[str, List[str], np.ndarray],
        target: str,
    ) -> Tuple[
        Union[str, List[str], np.ndarray],
        Tuple[Union[str, List[str]], Union[float, str]],
    ]:
        df = df_tmp[:]
        if isinstance(feature, str):
            split_point_ = sp_main_feature(df, feature)

        else:  # isinstance(feature, list):
            if not isinstance(feature[0], np.ndarray):
                sf, _ = sp_interaction_features(df, cast(List[str], feature), target)
                assert sf is not None, "sf should not be None here"
                feature = sf
                split_point_ = sp_main_feature(df, feature)

            else:
                lda_df = compute_lda(
                    df=df,
                    feature=cast(np.ndarray, feature),
                    target=self.target,
                )
                split_point_ = sp_main_feature(df=lda_df, feature="LDA")

        return feature, split_point_

    def _predict_sample(
        self, node: TreeNode, sample: pd.Series, probabilities: bool = False
    ) -> Union[str, Dict[str, Any]]:
        if node.value is not None:
            if probabilities:
                if self.target_expression is None:
                    raise ValueError("Adjust trees before the prediction.")
                diff = np.setdiff1d(self.target_expression, np.array(node.value.index))
                return_dict = dict(zip(diff, [0] * len(diff)))
                return_dict.update(dict(node.value))
                return return_dict
            else:
                return cast(str, node.value.index[0])
        else:
            if isinstance(node.threshold, float) and not isinstance(node.feature, list):
                # univariate split
                if sample[node.feature] <= node.threshold:
                    return self._predict_sample(
                        cast(TreeNode, node.left_tree),
                        sample,
                        probabilities=probabilities,
                    )
                else:
                    return self._predict_sample(
                        cast(TreeNode, node.right_tree),
                        sample,
                        probabilities=probabilities,
                    )

            elif isinstance(node.feature, list):
                lda_sample = compute_lda(
                    df=sample, feature=node.feature, target=self.target
                )

                if lda_sample <= node.threshold:
                    return self._predict_sample(
                        cast(TreeNode, node.left_tree),
                        sample,
                        probabilities=probabilities,
                    )
                else:
                    return self._predict_sample(
                        cast(TreeNode, node.right_tree),
                        sample,
                        probabilities=probabilities,
                    )

            else:  # categorical splitting
                if node.threshold == sample[node.feature]:
                    return self._predict_sample(
                        cast(TreeNode, node.left_tree),
                        sample,
                        probabilities=probabilities,
                    )
                else:
                    return self._predict_sample(
                        cast(TreeNode, node.right_tree),
                        sample,
                        probabilities=probabilities,
                    )

    def predict(self, df: pd.DataFrame) -> List[str]:
        predictions = []
        for sample in range(len(df)):
            prediction = self._predict_sample(
                cast(TreeNode, self.root), df.iloc[sample, :]
            )
            predictions.append(cast(str, prediction))
        return predictions

    def predict_probabilities(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        probabilities = []
        for sample in range(len(df)):
            prediction = self._predict_sample(
                cast(TreeNode, self.root), df.iloc[sample, :], probabilities=True
            )
            probabilities.append(cast(Dict[str, Any], prediction))
        return probabilities

    def _print_tree(self, node: TreeNode, indent: Optional[str] = None) -> None:
        indent = "" if indent is None else indent

        if node.value is not None:
            print(indent + "Leaf:", node.value)
        else:
            print(
                indent + "Feature:",
                node.feature,
                "Threshold:",
                node.threshold,
            )
            if node.left_tree is not None:
                print(indent + "True Branch:")
                self._print_tree(node.left_tree, indent + "  ")
            if node.right_tree is not None:
                print(indent + "False Branch:")
                self._print_tree(node.right_tree, indent + "  ")

    def score(self, df_test: pd.DataFrame) -> float:
        res = self.predict(df_test)
        rights = 0
        df_test.reset_index(inplace=True, drop=True)
        for r in range(len(res)):
            if res[r] == df_test.loc[r, "target"]:
                rights += 1
        return rights / len(df_test)

    def print_tree(self) -> None:
        if self.root is not None:
            self._print_tree(self.root)

    def plot_tree(self, *folders: str, name: str) -> str:
        dot_text = tree_to_dot(cast(TreeNode, self.root), initial=True)
        return plot_tree_graph(*folders, text=dot_text, name=name)

    def return_all_nodes(self, only_split: bool = False) -> List[TreeNode]:
        nodes = self._return_all_subnodes(
            cast(TreeNode, self.root), only_split, nodelist=[]
        )
        return nodes

    def _return_all_subnodes(
        self, node: TreeNode, only_split: bool, nodelist: List[TreeNode]
    ) -> List[TreeNode]:
        if only_split:
            if self._is_split_node(node):
                nodelist.extend([node])
        else:
            nodelist.extend([node])

        if node.left_tree is not None:
            nodelist = self._return_all_subnodes(node.left_tree, only_split, nodelist)

        if node.right_tree is not None:
            nodelist = self._return_all_subnodes(node.right_tree, only_split, nodelist)

        return nodelist

    @staticmethod
    def _is_split_node(node: TreeNode) -> bool:
        return True if node.feature is not None else False

    def return_all_leaves(self) -> List[TreeNode]:
        leaves = self._call_all_leaves(cast(TreeNode, self.root), leaveslist=[])
        return leaves

    def _call_all_leaves(
        self, node: TreeNode, leaveslist: List[TreeNode]
    ) -> List[TreeNode]:
        if node.value is not None and node.feature is None:
            leaveslist.extend([node])

        else:
            leaveslist = self._call_all_leaves(
                cast(TreeNode, node.left_tree), leaveslist
            )
            leaveslist = self._call_all_leaves(
                cast(TreeNode, node.right_tree), leaveslist
            )

        return leaveslist

    def compute_rel_volume_node(self, node: TreeNode, num_snowballs: int) -> float:
        numerator = node.runs / num_snowballs
        denominator = len(node.dataset) / len(cast(pd.DataFrame, self.train_df))
        if denominator == 0:
            print("Warning: Denominator is zero. No training sample in tree node!")
            denominator = 1 / len(cast(pd.DataFrame, self.train_df))
        return numerator / denominator

    def ison_prediction_path(self, x: pd.Series, check_node: TreeNode) -> bool:
        result = self._run_prediction_path(cast(TreeNode, self.root), x, check_node)
        return result

    def _run_prediction_path(
        self, node: TreeNode, sample: pd.Series, check_node: TreeNode
    ) -> bool:
        if node == check_node:
            return True

        elif node.value is None:
            if isinstance(node.threshold, float) and not isinstance(node.feature, list):
                # univariate split
                if sample[node.feature] <= node.threshold:
                    return self._run_prediction_path(
                        cast(TreeNode, node.left_tree), sample, check_node
                    )
                else:
                    return self._run_prediction_path(
                        cast(TreeNode, node.right_tree), sample, check_node
                    )

            elif isinstance(node.feature, list):
                lda_sample = compute_lda(
                    df=sample, feature=node.feature, target=self.target
                )

                if lda_sample <= node.threshold:
                    return self._run_prediction_path(
                        cast(TreeNode, node.left_tree), sample, check_node
                    )
                else:
                    return self._run_prediction_path(
                        cast(TreeNode, node.right_tree), sample, check_node
                    )

            else:  # categorical splitting
                if node.threshold == sample[node.feature]:
                    return self._run_prediction_path(
                        cast(TreeNode, node.right_tree), sample, check_node
                    )
                else:
                    return self._run_prediction_path(
                        cast(TreeNode, node.left_tree), sample, check_node
                    )

        return False

    def run_through_tree(self, df: pd.DataFrame) -> None:
        for i in range(len(df)):
            self._mark_nodes(cast(TreeNode, self.root), df.iloc[[i]])

    def _mark_nodes(self, node: TreeNode, sample: pd.DataFrame) -> None:
        node.runs += 1
        node.pool = pd.concat([node.pool, sample], ignore_index=True)
        '''
        if node.pool.empty:
            node.pool = sample.to_frame().T
        else:
            node.pool.loc[len(node.pool)] = sample
        '''


        if node.value is None:
            if isinstance(node.threshold, float) and not isinstance(node.feature, list):
                if sample.iloc[0][node.feature]<= node.threshold:
                    self._mark_nodes(cast(TreeNode, node.left_tree), sample)
                else:
                    self._mark_nodes(cast(TreeNode, node.right_tree), sample)

            elif isinstance(node.feature, list):
                lda_sample = compute_lda(
                    df=sample, feature=node.feature, target=self.target
                )

                if lda_sample.iloc[0]['LDA'] <= node.threshold:
                    self._mark_nodes(cast(TreeNode, node.left_tree), sample)
                else:
                    self._mark_nodes(cast(TreeNode, node.right_tree), sample)

            else:  # categorical splitting
                if node.threshold == sample.iloc[0][node.feature]:
                    self._mark_nodes(cast(TreeNode, node.right_tree), sample)
                else:
                    self._mark_nodes(cast(TreeNode, node.left_tree), sample)


def create_tree_plot(df: pd.DataFrame, max_depth: int = 10, min_info_gain: float = 0.075) -> str:
    dtc = DecisionTreeClassifierGUIDE(max_depth=max_depth, min_info_gain=min_info_gain)
    dtc.fit(df, target="target")
    dot_text = tree_to_dot(dtc.root, initial=True)
    graphviz_text = (
            'digraph Tree {\nnode [shape=box, fontname="helvetica"] '
            ';\nedge [fontname="helvetica"] ;\n' + dot_text + "}"
    )
    return graphviz_text

