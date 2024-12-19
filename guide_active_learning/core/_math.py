import math
from typing import Any, cast, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from scipy.spatial.distance import minkowski


__all__ = [
    "cm2inch",
    "calculate_column_beta",
    "calculate_column_gamma",
    "calculate_std",
    "calculate_mean_std",
    "compute_wilsonhilferty",
    "comp_mean_vectors",
    "scatter_within",
    "scatter_between",
    "compute_gini_np",
    "compute_gini_pd",
    "compute_information_gain",
    "compute_lda",
    "compute_minkowski_distance",
    "compute_categorical_distances",
    "compute_minimize",
    "create_intervals",
    "calculate_contribution_table",
    "linear_lda_transform",
]


def cm2inch(value: Union[int, float]) -> float:
    return value * 0.3937


def calculate_column_beta(len_columns: int) -> float:
    return 0.05 / (len_columns * (len_columns - 1)) if len_columns > 1 else 0


def calculate_column_gamma(len_number_columns: int) -> float:
    return (
        0.05 / (len_number_columns * (len_number_columns - 1))
        if len_number_columns > 1
        else 0
    )


def calculate_std(data: np.ndarray) -> np.ndarray:
    return cast(np.ndarray, (data - np.min(data)) / (np.max(data) - np.min(data)))


def calculate_mean_std(
    result_array: np.ndarray,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    # mean_scores = [np.mean(result_array[i], axis=0) for i in range(len(result_array))]
    mean_scores = np.mean(result_array, axis=0)
    # std_scores = [np.std(result_array[i], axis=0) for i in range(len(result_array))]
    std_scores = np.std(result_array, axis=0)
    return mean_scores, std_scores


def compute_wilsonhilferty(chi2_result: Dict[int, Any]) -> float:
    chi2 = chi2_result[0]
    deg = chi2_result[2]
    if deg != 0 and deg > 1:
        return cast(
            float,
            ((7 / 9) + math.sqrt(deg) * ((chi2 / deg) ** (1 / 3) - 1 + (2 / (9 * deg))))
            ** 3,
        )

    return cast(float, chi2) if deg != 0 and deg <= 1 else (7 / 9) ** 3


def comp_mean_vectors(X: np.ndarray, y: np.ndarray) -> List[np.ndarray]:
    # helping functions for LDA computation
    class_labels = np.unique(y)
    mean_vectors = []
    for cl in class_labels:
        mean_vectors.append(np.mean(X[y == cl], axis=0))
    return mean_vectors


def scatter_within(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    class_labels = np.unique(y)
    n_features = X.shape[1]
    mean_vectors = comp_mean_vectors(X, y)
    S_W = np.zeros((n_features, n_features))
    for cl, mv in zip(class_labels, mean_vectors):
        class_sc_mat = np.zeros((n_features, n_features))
        for row in X[y == cl]:
            row, mv = row.reshape(n_features, 1), mv.reshape(n_features, 1)
            class_sc_mat += (row - mv).dot((row - mv).T)
        S_W += class_sc_mat
    return S_W


def scatter_between(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    overall_mean = np.mean(X, axis=0)
    n_features = X.shape[1]
    mean_vectors = comp_mean_vectors(X, y)
    S_B = np.zeros((n_features, n_features))
    for i, mean_vec in enumerate(mean_vectors):
        n = X[y == i + 1, :].shape[0]
        mean_vec = mean_vec.reshape(n_features, 1)
        overall_mean = overall_mean.reshape(n_features, 1)
        S_B += n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)
    return S_B


def compute_gini_np(df: pd.DataFrame) -> float:
    if len(df) == 0:
        return 1
    else:
        unique, counts = np.unique(df[:, -1], return_counts=True)
        return cast(float, 1 - np.sum((counts / len(df)) ** 2))


def compute_gini_pd(df: pd.DataFrame) -> float:
    target = "target"
    return cast(float, 1 - sum((df[target].value_counts() / len(df)) ** 2))


def compute_information_gain(
    df_par: pd.DataFrame, df_c1: pd.DataFrame, df_c2: pd.DataFrame
) -> float:
    gini_parent = compute_gini_pd(df_par)
    gini_c1 = compute_gini_pd(df_c1)
    gini_c2 = compute_gini_pd(df_c2)
    return (
        gini_parent
        - (len(df_c1) / len(df_par)) * gini_c1
        - (len(df_c2) / len(df_par)) * gini_c2
    )


def compute_lda(
    df: Union[pd.Series, pd.DataFrame],
    feature: Union[List[str], np.ndarray],
    target: str,
) -> pd.DataFrame:
    tmp = [str(i) for i in list(feature[1])]
    tmp.append(target)
    df = df[:]
    df = df[tmp]

    if type(df) == pd.DataFrame:  # multiple datapoints
        # check order of features
        if list(df.columns[:-1]) == list(feature[1]):
            lda_arr = df.drop(columns=target).to_numpy().dot(feature[0])

        else:
            lda_arr = df.drop(columns=target).to_numpy().dot(np.flip(feature[0]))

    elif type(df) == pd.Series:  # single datapoint
        # check order of features
        if np.all(df.index[:-1] == list(feature[1])):
            lda_arr = df.drop(index=target).to_numpy().dot(feature[0])

        else:
            lda_arr = df.drop(index=target).to_numpy().dot(np.flip(feature[0]))
        return lda_arr

    else:
        raise TypeError("compute_lda only with pandas series or dataframe")

    lda_df = pd.DataFrame(
        np.hstack((lda_arr.reshape(-1, 1), df[target].values.reshape(-1, 1))),
        columns=["LDA", target],
    )
    lda_df["LDA"] = pd.to_numeric(lda_df["LDA"])

    return lda_df


def compute_minkowski_distance(
    array_initial: np.ndarray,
    array_test: np.ndarray,
    distances: np.ndarray,
    norm: int = 2,
) -> np.ndarray:
    for i in range(len(array_initial)):
        for j in range(len(array_test)):
            distances[i, j] = minkowski(array_test[j, :], array_initial[i, :], p=norm)

    return distances


def compute_categorical_distances(
    array1: np.ndarray,
    array2: np.ndarray,
    df_initial: pd.DataFrame,
    array_test: np.ndarray,
    numerical_distances: np.ndarray,
) -> float:
    cat_distances = np.ones((len(array1), len(array2)))
    for j in range(len(array2)):
        bool_arr = np.sum(np.isin(array1, array2[j, :]), axis=1)
        intersec = np.array(
            [len(np.union1d(array1[i, :], array2[j, :])) for i in range(len(array1))]
        )

        cat_distances[:, j] = 1 - bool_arr / intersec

    no_features = df_initial.shape[-1] - 1
    # 01-Scaling
    normed_distances = (numerical_distances - np.min(numerical_distances)) / (
        np.max(numerical_distances) - np.min(numerical_distances)
    )
    # TODO: right normalization ?!
    return cast(
        float,
        (array2.shape[-1] / no_features) * cat_distances
        + (array_test.shape[-1] / no_features) * normed_distances,
    )


def compute_minimize(array: np.ndarray, minimize: str = "min") -> np.ndarray:
    if minimize == "mean":
        return cast(np.ndarray, np.mean(array, axis=0))
    elif minimize == "min":
        return cast(np.ndarray, np.min(array, axis=0))
    else:
        return array


def create_intervals(
    mean: float, std: float, num_intervals: int = 3, divider: int = 3
) -> List[float]:
    intervals = [-math.inf, math.inf]

    if num_intervals % 2 == 0 or num_intervals == 2:
        intervals.append(mean)

    if num_intervals != 2:
        for i in range(1, math.ceil(num_intervals / 2)):
            intervals.append(mean - i * std * (math.sqrt(3) / divider))
            intervals.append(mean + i * std * (math.sqrt(3) / divider))

    return sorted(intervals)


def calculate_contribution_table(
    df: pd.DataFrame,
    target: str,
    column: str,
    use_interval: bool = False,
) -> pd.DataFrame:
    tmp = df.loc[:, [column, target]]

    if use_interval:
        mean = tmp.drop(columns=target).mean()
        std = tmp.drop(columns=target).std()
        if len(tmp) >= 20 * len(df[target].value_counts()):
            intervals = create_intervals(
                mean.values[0], std.values[0], num_intervals=4, divider=2
            )

        else:
            intervals = create_intervals(
                mean.values[0], std.values[0], num_intervals=3, divider=2
            )

        tmp["interval"] = pd.cut(
            tmp[column], bins=intervals, labels=False, duplicates="drop"
        )
        cont_table = pd.crosstab(tmp[target], tmp["interval"])
    else:
        cont_table = pd.crosstab(tmp[target], tmp[column])

    # delete empty rows and columns
    cont_table = cont_table.loc[(cont_table != 0).any(axis=1)]
    cont_table = cont_table.loc[:, (cont_table != 0).any(axis=0)]
    return cont_table


def linear_lda_transform(
    *, matrix_in: np.ndarray, matrix_out: np.ndarray, arr: np.ndarray, df: pd.DataFrame
) -> Tuple[np.ndarray, pd.DataFrame]:
    # starting with LDA from:
    # https://sebastianraschka.com/Articles/2014_python_lda.html#preparing-the-sample-data-set
    # other possibility sklearn: deviations between methods
    eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(matrix_in).dot(matrix_out))
    # project data on new axis
    max_eig_vec = eig_vecs[:,np.argmax(eig_vals)]  # changed
    if isinstance(max_eig_vec[0], complex) or isinstance(max_eig_vec[1], complex):
        max_eig_vec = np.array([i.real for i in max_eig_vec])

    arr_lda = arr[:, :-1].dot(max_eig_vec)
    tar = arr[:, -1].reshape(-1, 1)
    arr_lda = np.hstack((arr_lda.reshape(-1, 1), tar))

    df_lda = pd.DataFrame(data=arr_lda, columns=["LDA_feature", df.columns[-1]])
    return max_eig_vec, df_lda
