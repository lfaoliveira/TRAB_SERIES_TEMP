import logging
from pathlib import Path
from typing import Any, Callable, List, Literal
from darts import TimeSeries

# import mlflow
import numpy as np
import pandas as pd

# from mlflow.artifacts import download_artifacts
import os


# # Data Processing Functions
# def _grab_values(
#     available_metrics: list[str],
#     client: mlflow.MlflowClient,
#     run_id: str,
# ) -> dict[str, dict[int, float]]:
#     """
#     Collects metrics from MLflow without plotting.

#     :param available_metrics: metrics list from mlflow
#     :param client: MLFlow Client
#     :param run_id: run ID
#     :return: Dictionary with metrics data
#     """
#     metrics_dict: dict[str, dict[int, float]] = {}

#     for metric_name in available_metrics:
#         metric_history = client.get_metric_history(run_id, metric_name)
#         metric_history = sorted(metric_history, key=lambda m: m.step)

#         if metric_history:
#             # Indexed by epoch number (sequence index) instead of mlflow step
#             metrics_dict[metric_name] = {epoch: m.value for epoch, m in enumerate(metric_history)}

#     return metrics_dict


# def _calculate_metric_averages(
#     metrics_dict: dict[str, dict[int, float]],
# ) -> dict[str, float]:
#     """Calculate average values for all metrics."""
#     averages: dict[str, float] = {}

#     for metric_name, epochs_values in metrics_dict.items():
#         values = list(epochs_values.values())
#         if values:
#             averages[f"{metric_name}_avg"] = float(np.average(values))
#     return averages


# def residual_analysis(
#     client: mlflow.MlflowClient,
#     best_run: Any,
#     name: str,
#     processer: ResultsProcesser,
#     plot=False,  # wether to plot agreggate confusion matrix results
# ):
#     # Get artifacts from the run and search for test_results_{run_id}.csv
#     artifacts = client.list_artifacts(best_run.run_id)
#     test_results_file = f"test_results_{best_run.run_id}.csv"
#     artifact_found = any(artifact.path == test_results_file for artifact in artifacts)

#     if not artifact_found:
#         logging.info(f"Artifact '{test_results_file}' not found in run {best_run.run_id}")
#         raise Exception("ARTIFACT NOT FOUND!")

#     # Pass the loaded model to your analysis function
#     df_path = download_artifacts(artifact_path=test_results_file, run_id=best_run.run_id)

#     prediction_df = pd.read_csv(df_path).dropna(how="any")
#     if plot:
#         plot_classification_errors(prediction_df, "stroke", "pred")
#     error_model = ErrorModel(prediction_df)
#     processer.update(name, error_model, prediction_df)


# def metric_filter(metric_name: str, not_list: list[str]):
#     """Returns true if any unallowed string is in metric_name"""
#     allowed = all([(unallowed not in metric_name) for unallowed in not_list])
#     return allowed


# def final_analysis(
#     models: list,
#     output_dir: Path,
#     sort_metric: Literal["val_f_beta_avg", "val_f1_avg", "val_loss_avg"],
#     unwanted_metrics: list[str],
#     residual=True,
#     exp_name: str = "PROD_TRAINING",
# ) -> tuple[pd.DataFrame, ResultsProcesser]:
#     """Generate metrics and plots for trained models. residual indicates wether to
#     store basic residual analysis information for later use.

#     Args:
#         exp_name:
#         MLflow experiment name. All models' runs are expected in this single experiment,
#         filtered by run name containing the model choice (e.g. 'PROD_MLP', 'PROD_KAN').
#     """

#     is_optuna = bool(os.environ.get("OPTUNA", False))
#     all_models_metrics = []
#     output_dir.mkdir(exist_ok=True)
#     client = mlflow.MlflowClient()

#     processer = ResultsProcesser()

#     experiment = mlflow.get_experiment_by_name(exp_name)
#     if not experiment:
#         logging.info(f"Experiment '{exp_name}' not found")
#         return pd.DataFrame(), processer

#     all_experiment_runs = pd.DataFrame(mlflow.search_runs(experiment_ids=[experiment.experiment_id]))
#     assert isinstance(all_experiment_runs, pd.DataFrame)

#     for choice in models:
#         # Filter runs for this model by run name
#         if "tags.mlflow.runName" in all_experiment_runs.columns:
#             runs = all_experiment_runs[
#                 all_experiment_runs["tags.mlflow.runName"].str.contains(choice, case=False, na=False)
#             ].copy()
#         else:
#             runs = all_experiment_runs.copy()

#         if runs.empty:
#             logging.info(f"No runs found for model '{choice}' in experiment '{exp_name}'")
#             continue

#         model_metrics = {"model": choice}
#         all_runs_metrics_dict = {}

#         # Process each run
#         for idx, run_id in enumerate(runs["run_id"]):
#             run = client.get_run(run_id)
#             available_metrics = list(run.data.metrics.keys())
#             available_metrics = list(
#                 filter(
#                     lambda elem: metric_filter(elem, unwanted_metrics),
#                     available_metrics,
#                 )
#             )
#             # Collect metrics
#             run_metrics_dict = _grab_values(available_metrics, client, run_id)
#             all_runs_metrics_dict[run_id] = run_metrics_dict
#             # Calculate averages
#             averages = _calculate_metric_averages(run_metrics_dict)

#             # Inject averages into the runs DataFrame
#             for metric_name, avg_value in averages.items():
#                 runs.loc[runs["run_id"] == run_id, f"metrics.{metric_name}"] = avg_value

#             model_metrics.update(averages)

#             # Plot individual runs only when NOT using Optuna
#             if not is_optuna and run_metrics_dict:
#                 plot_single_run(run_metrics_dict, choice, str(idx), output_dir)

#         # stores residual model and dataframe
#         if residual and not runs.empty:
#             ascending = "loss" in sort_metric
#             best_run = runs.sort_values(f"metrics.{sort_metric}", ascending=ascending).iloc[0]
#             residual_analysis(client, best_run, choice, processer, plot=True)

#         # Always plot combined view
#         if is_optuna and all_runs_metrics_dict:
#             plot_all_runs_per_model(all_runs_metrics_dict, choice, output_dir)

#         all_models_metrics.append(model_metrics)
#         logging.info(f"Graphs exported to: {output_dir}")

#     return pd.DataFrame(all_models_metrics).set_index("model"), processer


# ---------------------------------------------------------------------------
# Utilitário centralizado de janelamento
# ---------------------------------------------------------------------------
type WindowPredicate = Callable[[np.ndarray, Any], pd.Series[bool]]


def extract_windows(
    values: np.ndarray,
    window_size: int,
    centered: bool = False,
    padding_mode: str = "edge",
) -> np.ndarray:
    """
    Extrai janelas deslizantes de um array 1-D.

    Dois modos de operação:

    - **centered=False** (padrão): retorna um array ``(n_windows, window_size)``
      com janelas consecutivas sem stride::

          windows[k] = values[k : k + window_size]

      Neste modo o número de janelas é ``n_windows = n - window_size + 1``.

    - **centered=True**: retorna um array ``(n, window_size)`` onde cada
      linha *i* é uma janela centrada no ponto ``values[i]``. Quando a
      janela extrapola as bordas da série, o padding é feito com o modo
      indicado por *padding_mode*.

    Parameters
    ----------
    values : np.ndarray
        Array 1-D de valores.
    window_size : int
        Número de elementos por janela.
    centered : bool
        Se ``True``, cada ponto ganha uma janela centrada nele com padding
        nas bordas. Se ``False`` (padrão), janelas consecutivas sem stride.
    padding_mode : str
        Modo de padding ``np.pad`` usado nas bordas quando ``centered=True``.

    Returns
    -------
    np.ndarray
        Array de janelas. Shape ``(n_windows, window_size)`` para
        ``centered=False`` ou ``(n, window_size)`` para ``centered=True``.
        Retorna um array vazio ``(0, window_size)`` se os dados forem
        mais curtos que a janela no modo não-centrado.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    ws = window_size

    if not centered:
        if n < ws:
            return np.empty((0, ws), dtype=float)
        return np.lib.stride_tricks.sliding_window_view(values, window_shape=ws)

    # Modo centrado: cada ponto i recebe uma janela com ws elementos
    half = ws // 2
    windows = np.empty((n, ws), dtype=float)

    for i in range(n):
        left = i - half
        right = i + (ws - half - 1)  # left + 1 + right == ws
        start = max(0, left)
        end = min(n, right + 1)
        window = values[start:end]

        if len(window) < ws:
            pad_before = max(0, -left)
            pad_after = max(0, right + 1 - n)
            window = np.pad(window, (pad_before, pad_after), mode=padding_mode)

        windows[i] = window

    return windows


def rolling_window_apply(
    series_list: List[TimeSeries],
    window_size: int,
    predicate: WindowPredicate,
    overlap=False,
) -> List[np.ndarray]:
    """
    Aplica um rolling window de tamanho fixo sobre cada TimeSeries e avalia
    *predicate* em cada janela, retornando um array booleano por série.

    Parameters
    ----------
    series_list : List[TimeSeries]
        Lista de sequências numéricas 1-D.
    window_size : int
        Número de observações em cada janela.
    predicate : Callable[[np.ndarray], bool]
        Função que recebe um np.ndarray (uma janela) e retorna bool.
    overlap : bool
        Se ``False`` (padrão), janelas não se sobrepõem (stride = window_size).

    Returns
    -------
    List[np.ndarray]
        Um np.ndarray de bool por série de entrada.
    """
    if window_size < 1:
        raise ValueError(f"window_size deve ser >= 1, recebeu {window_size}")

    results: List[np.ndarray] = []

    for series in series_list:
        vals = np.asarray(series.univariate_values(), dtype=float)
        stride = 1 if overlap else window_size
        windows = extract_windows(vals, window_size, centered=False)

        if not overlap:
            # extract_windows sem stride dá sliding window pura;
            # queremos stride = window_size, então amostramos de stride em stride
            windows = windows[::stride]

        bool_vals = np.array([predicate(w) for w in windows], dtype=bool)

        # Expande o resultado para o comprimento original via repeat
        if not overlap:
            n = len(vals)
            expanded = np.full(n, False, dtype=bool)
            for idx, w in enumerate(windows):
                start = idx * window_size
                end = min(start + window_size, n)
                if bool_vals[idx]:
                    expanded[start:end] = True
            results.append(expanded)
        else:
            results.append(bool_vals)

    return results
