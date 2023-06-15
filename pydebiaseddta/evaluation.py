from typing import List, Dict
from itertools import combinations

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def ci(gold_truths: List[float], predictions: List[float]) -> float:
    """Computes concordance index (CI) between the expected values and predictions. 
    See [Gönen and Heller (2005)](https://www.jstor.org/stable/20441249) for the details of the metric.

    Parameters
    ----------
    gold_truths : List[float]
        The gold labels in the dataset.  
    predictions : List[float]
        Predictions of a model.
    
    Returns
    -------
    float
        Concordance index.
    """
    gold_combs, pred_combs = combinations(gold_truths, 2), combinations(predictions, 2)
    nominator, denominator = 0, 0
    for (g1, g2), (p1, p2) in zip(gold_combs, pred_combs):
        if g2 > g1:
            nominator = nominator + 1 * (p2 > p1) + 0.5 * (p2 == p1)
            denominator = denominator + 1

    return float(nominator / denominator)


def mse(gold_truths: List[float], predictions: List[float]) -> float:
    """Computes mean squared error between expected and predicted values.

    Parameters
    ----------
    gold_truths : List[float]
        The gold labels in the dataset.
    predictions : List[float]
        Predictions of a model.

    Returns
    -------
    float
        Mean squared error.
    """
    return float(mean_squared_error(gold_truths, predictions, squared=True))


def rmse(gold_truths: List[float], predictions: List[float]) -> float:
    """Computes root mean squared error between expected and predicted values.

    Parameters
    ----------
    gold_truths : List[float]
        The gold labels in the dataset.
    predictions : List[float]
        Predictions of a model.

    Returns
    -------
    float
        Root mean squared error.
    """
    return float(mean_squared_error(gold_truths, predictions, squared=False))


def r2(gold_truths: List[float], predictions: List[float]) -> float:
    """Compute $R^2$ (coefficient of determinant) between expected and predicted values.

    Parameters
    ----------
    gold_truths : List[float]
        The gold labels in the dataset.
    predictions : List[float]
        Predictions of a model.

    Returns
    -------
    float
        $R^2$ (coefficient of determinant) score.
    """
    return float(r2_score(gold_truths, predictions))


def mae(gold_truths: List[float], predictions: List[float]) -> float:
    """Computes mean absolute error between expected and predicted values.

    Parameters
    ----------
    gold_truths : List[float]
        The gold labels in the dataset.
    predictions : List[float]
        Predictions of a model.

    Returns
    -------
    float
        Mean squared error.
    """
    return float(mean_absolute_error(gold_truths, predictions))


def evaluate_predictions(
    gold_truths: List[float], predictions: List[float], metrics: List[str] = None
) -> Dict[str, float]:
    """Computes multiple metrics with a single call for convenience. 

    Parameters
    ----------
    gold_truths : List[float]
        The gold labels in the dataset.
    predictions : List[float]
        Predictions of a model.
    metrics : List[str]
        Name of the evaluation metrics to compute. Possible values are: `{"ci", "r2", "rmse", "mse"}`.
        All metrics are computed if no value is provided.

    Returns
    -------
    Dict[str,float]
        A dictionary that maps each metric name to the computed value.
    """
    if metrics is None:
        metrics = ["ci", "r2", "rmse", "mse", "mae"]

    metrics = [metric.lower() for metric in metrics]
    name_to_fn = {"ci": ci, "r2": r2, "rmse": rmse, "mse": mse, "mae": mae}
    return {metric: name_to_fn[metric](gold_truths, predictions) for metric in metrics}
