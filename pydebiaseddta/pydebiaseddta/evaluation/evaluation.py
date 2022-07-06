from typing import List, Dict
from itertools import combinations

from sklearn.metrics import mean_squared_error, r2_score


def ci(gold_truths: List[float], predictions: List[float]) -> float:
    """Compute concordance index between the expected values and predictions.
    Concordance index is calculated via:
    
    .. math::
        \frac {1} {Z} \sum_{\delta_x > \delta_y} h (b_x - b_y)

    where :math:`b_x` is the prediction for the larger affinity :math:`\delta_x`, 
    :math:`b_y` is the prediction for the smaller affinity :math:`\delta_y`, 
    :math:`Z` is a normalization constant,
    :math:`h(m)` is the step function.

    Parameters
    ----------
    gold_truths : List[float]
        The gold labels in the dataset
    predictions : List[float]
         Predictions of a model.

    Returns
    -------
    float
        float: Concordance index.
    """    
    gold_combs, pred_combs = combinations(gold_truths, 2), combinations(predictions, 2)
    nominator, denominator = 0, 0
    for (g1, g2), (p1, p2) in zip(gold_combs, pred_combs):
        if g2 > g1:
            nominator = nominator + 1 * (p2 > p1) + 0.5 * (p2 == p1)
            denominator = denominator + 1

    return float(nominator / denominator)


def mse(gold_truths: List[float], predictions: List[float]) -> float:
    """ Compute mean squared error between expected and predicted values

    Args:
        gold_truths (List[float]): The gold labels in the dataset.  
        predictions (List[float]): Predictions of a model.

    Returns:
        float: Mean squared error.
    """
    return float(mean_squared_error(gold_truths, predictions, squared=True))


def rmse(gold_truths: List[float], predictions: List[float]) -> float:
    """ Compute root mean squared error between expected and predicted values

    Args:
        gold_truths (List[float]): The gold labels in the dataset.  
        predictions (List[float]): Predictions of a model.

    Returns:
        float: Root mean squared error.
    """
    return float(mean_squared_error(gold_truths, predictions, squared=False))


def r2(gold_truths: List[float], predictions: List[float]) -> float:
    """ Compute :math:`R^2` (coefficient of determinant) between expected and predicted values

    Args:
        gold_truths (List[float]): The gold labels in the dataset.  
        predictions (List[float]): Predictions of a model.

    Returns:
        float: :math:`R^2` score.
    """
    return float(r2_score(gold_truths, predictions))


def evaluate_predictions(
    gold_truths: List[float], predictions: List[float], metrics: List[str] = None
) -> Dict[str, float]:
    """ A convenience function to compute several metrics in a single line. 

    Args:
        gold_truths (List[float]): The gold labels in the dataset.  
        predictions (List[float]): Predictions of a model.
        metrics (List[str]): Name of the evaluation metrics to compute. 
            The valid values are: {"ci", "r2", "rmse", "mse"}. 
            All metrics are computed if no value is provided.
    """
    if metrics is None:
        metrics = ["ci", "r2", "rmse", "mse"]

    metrics = [metric.lower() for metric in metrics]
    name_to_fn = {"ci": ci, "r2": r2, "rmse": rmse, "mse": mse}
    return {metric: name_to_fn[metric](gold_truths, predictions) for metric in metrics}
