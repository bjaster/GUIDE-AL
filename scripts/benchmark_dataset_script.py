import time
from typing import Any, Dict, Optional

from guide_active_learning.ActiveLearning import investigation_loop
from guide_active_learning.analysis import evaluate_results
from guide_active_learning.misc import with_settings


@with_settings
def main_loop(dataset_str: str, settings: Optional[Dict[str, Any]] = None) -> None:
    if settings is None:
        raise ValueError("Settings should not be None.")
    # use the timestamp as the unique identifier for this benchmark
    benchmark_str = settings["benchmark_name"]
    print(dataset_str)
    for active_learning_method in settings["active_learning_methods"]:
        if active_learning_method in [
            "qbc_exex",
            "qbc_exex_2",
            "qbc_exex_2cat",
            "uncertainty_guide",
            "uncertainty_rf",
            "qbc_exex_rf",
        ]:
            calculate_ensemble = True
        else:
            # It is not necessary to calculate an ensemble in this instance.
            calculate_ensemble = False

        if active_learning_method in [
            "qbc_exex",
            "qbc_exex_2",
            "qbc_exex_2cat",
            "qbc_exex_rf"
        ]:
            alpha_weights = settings["alpha_weights"]
        else:
            # It is unnecessary to perform a loop over the alpha weights.
            alpha_weights = None

        investigation_loop(
            dataset_str=dataset_str,
            benchmark_str=benchmark_str,
            active_learning_method=active_learning_method,
            initial_datapoints=settings["initial_datapoints"],
            alpha_weights=alpha_weights,
            split_type=settings["split_type"],
            max_depth=settings["max_depth"],
            use_linear_split=settings["use_linear_split"],
            min_info_gain=settings["min_info_gain"],
            ensemble_size=settings["ensemble_size"],
            num_benchmark=settings["num_benchmark"],
            active_learning_steps=settings["active_learning_steps"],
            parallel_computation=settings["parallel_computation"],
            calculate_ensemble=calculate_ensemble,
            pool_synth=settings["pool_synth"],
        )


@with_settings
def perform_benchmark(settings: Optional[Dict[str, Any]] = None) -> None:
    if settings is None:
        raise ValueError("Settings should not be None.")
    for dataset_str in settings["datasets"]:
        main_loop(dataset_str=dataset_str)


@with_settings
def evaluate_benchmark(bm_name: str = None, settings: Optional[Dict[str, Any]] = None,) -> None:
    if settings is None:
        raise ValueError("Settings should not be None.")
    if bm_name is None:
        bm_name = settings["benchmark_name"]
    for dataset_str in settings["datasets"]:
        evaluate_results(dataset_str, bm_name, settings['initial_datapoints'])


if __name__ == "__main__":
    perform_benchmark()
    evaluate_benchmark()
