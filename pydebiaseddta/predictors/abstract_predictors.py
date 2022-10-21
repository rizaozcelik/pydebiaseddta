from typing import Any, Dict, List
import json
from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf

from ..evaluation import evaluate_predictions


def create_uniform_weights(n_samples: int, n_epochs: int) -> List[np.array]:
    """Create a lists of weights such that every training instance has the equal weight across all epoch,
    *i.e.*, no sample weighting is used.

    Parameters
    ----------
    n_samples : int
        Number of training instances.
    n_epochs : int
        Number of epochs to train the model.

    Returns
    -------
    List[np.array]
        Sample weights across epochs. Each instance has a weight of 1 for all epochs.
    """
    return [np.array([1] * n_samples) for _ in range(n_epochs)]


tf.get_logger().setLevel("WARNING")


class Predictor(ABC):
    """An abstract class that implements the interface of a predictor in `pydebiaseddta`.
    The predictors are characterized by an `n_epochs` attribute and a `train` function, 
    whose signatures are implemented by this class. 
    Any instance of `Predictor` class can be trained in the `DebiasedDTA` training framework,
    and therefore, `Predictor` can be inherited to debias custom DTA prediction models.
    """

    @abstractmethod
    def __init__(self, n_epochs: int, *args, **kwargs) -> None:
        """An abstract constructor for `Predictor` to display that `n_epochs` is a necessary attribute for children classes.

        Parameters
        ----------
        n_epochs : int
            Number of epochs to train the model.
        """
        self.n_epochs = n_epochs

    @abstractmethod
    def train(
        self,
        train_ligands: List[Any],
        train_proteins: List[Any],
        train_labels: List[float],
        val_ligands: List[Any] = None,
        val_proteins: List[Any] = None,
        val_labels: List[float] = None,
        sample_weights_by_epoch: List[np.array] = None,
    ) -> Any:
        """An abstract method to train DTA prediction models.
        The inputs can be of any biomolecule representation type.
        However, the training procedure must support sample weighting in every epoch.

        Parameters
        ----------
        train_ligands : List[Any]
            The training ligands as a List.
        train_proteins : List[Any]
            The training proteins as a List.
        train_labels : List[float]
            Affinity scores of the training protein-compound pairs
        val_ligands : List[Any], optional
            Validation ligands as a List, in case validation scores are measured during training, by default `None`
        val_proteins : List[Any], optional
            Validation proteins as a List, in case validation scores are measured during training, by default `None`
        val_labels : List[float], optional
            Affinity scores of validation protein-compound pairs as a List, in case validation scores are measured during training, by default `None`

        Returns
        -------
        Any
            The function is free to return any value after its training, including `None`.
        """
        pass


class TFPredictor(Predictor):
    """The models in DebiasedDTA study (BPE-DTA, LM-DTA, DeepDTA) are implemented in Tensorflow.
    `TFPredictor` class provides an abstraction to these models to minimize code duplication.
    The children classes only implement model building, biomolecule vectorization, and `__init__` functions.  
    Model training, prediction, and save/load functions are inherited from this class.
    """

    @abstractmethod
    def __init__(self, n_epochs: int, learning_rate: float, batch_size: int, **kwargs):
        """An abstract constructor for BPE-DTA, LM-DTA, and DeepDTA.
        The constructor sets the common attributes and call the `build` function.  

        Parameters
        ----------
        n_epochs : int
            Number of epochs to train the model.
        learning_rate : float
            The learning rate of the optimization algorithm.
        batch_size : _type_
            Batch size for training.
        """
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.history = dict()
        self.model = self.build()

    @abstractmethod
    def build(self):
        """An abstract function to create the model architecture.
        Every child has to implement this function.
        """
        pass

    @abstractmethod
    def vectorize_ligands(self, ligands):
        """An abstract function to vectorize ligands.
        Every child has to implement this function.
        """
        pass

    @abstractmethod
    def vectorize_proteins(self, proteins):
        """An abstract function to vectorize proteins.
        Every child has to implement this function.
        """
        pass

    @classmethod
    def from_file(cls, path: str):
        """A utility function to load a DTA prediction model from disk.
        All attributes, including the model weights, are loaded.

        Parameters
        ----------
        path : str
            Path to load the prediction model from.

        Returns
        -------
        TFPredictor
            The previously saved model.
        """
        with open(f"{path}/params.json") as f:
            dct = json.load(f)

        instance = cls(**dct)

        instance.model = tf.keras.models.load_model(f"{path}/model")

        with open(f"{path}/history.json") as f:
            instance.history = json.load(f)
        return instance

    def train(
        self,
        train_ligands: List[str],
        train_proteins: List[str],
        train_labels: List[float],
        val_ligands: List[str] = None,
        val_proteins: List[str] = None,
        val_labels: List[float] = None,
        sample_weights_by_epoch: List[np.array] = None,
    ) -> Dict:
        """The common model training procedure for BPE-DTA, LM-DTA, and DeepDTA.
        The models adopt different biomolecule representation methods and model architectures,
        so, the training results are different.
        The training procedure supports validation for tracking, and sample weighting for debiasing.

        Parameters
        ----------
        train_ligands : List[str]
            SMILES strings of the training ligands.
        train_proteins : List[str]
            Amino-acid sequences of the training proteins.
        train_labels : List[float]
            Affinity scores of the training protein-ligand pairs.
        val_ligands : List[str], optional
            SMILES strings of the validation ligands, by default None and no validation is used.
        val_proteins : List[str], optional
            Amino-acid sequences of the validation proteins, by default None and no validation is used.
        val_labels : List[float], optional
            Affinity scores of the validation pairs, by default None and no validation is used.
        sample_weights_by_epoch : List[np.array], optional
            Weight of each training protein-ligand pair during training across epochs.
            This variable must be a List of size $E$ (number of training epochs),
            in which each element is a `np.array` of $N\times 1$, where $N$ is the training set size and 
            each element corresponds to the weight of a training sample.
            By default `None` and no weighting is used.

        Returns
        -------
        Dict
            Training history.
        """
        if sample_weights_by_epoch is None:
            sample_weights_by_epoch = create_uniform_weights(
                len(train_ligands), self.n_epochs
            )

        train_ligand_vectors = self.vectorize_ligands(train_ligands)
        train_protein_vectors = self.vectorize_proteins(train_proteins)
        train_labels = np.array(train_labels)

        val_tuple = None
        if (
            val_ligands is not None
            and val_proteins is not None
            and val_labels is not None
        ):
            val_ligand_vectors = self.vectorize_ligands(val_ligands)
            val_protein_vectors = self.vectorize_proteins(val_proteins)
            val_tuple = (
                [val_ligand_vectors, val_protein_vectors],
                np.array(val_labels),
            )

        train_stats_over_epochs = {"mse": [], "rmse": [], "r2": []}
        val_stats_over_epochs = train_stats_over_epochs.copy()
        for e in range(self.n_epochs):
            self.model.fit(
                x=[train_ligand_vectors, train_protein_vectors],
                y=train_labels,
                sample_weight=sample_weights_by_epoch[e],
                validation_data=val_tuple,
                batch_size=self.batch_size,
                epochs=1,
            )

            train_stats = evaluate_predictions(
                gold_truths=train_labels,
                predictions=self.predict(train_ligands, train_proteins),
                metrics=list(train_stats_over_epochs.keys()),
            )
            for metric, stat in train_stats.items():
                train_stats_over_epochs[metric].append(stat)

            if val_tuple is not None:
                val_stats = evaluate_predictions(
                    y_true=val_labels,
                    y_preds=self.predict(val_tuple[0], val_tuple[1]),
                    metrics=list(val_stats_over_epochs.keys()),
                )
                for metric, stat in val_stats.items():
                    val_stats_over_epochs[metric].append(stat)

        self.history["train"] = train_stats_over_epochs
        if val_stats_over_epochs is not None:
            self.history["val"] = val_stats_over_epochs

        return self.history

    def predict(self, ligands, proteins):
        ligand_vectors = self.vectorize_ligands(ligands)
        protein_vectors = self.vectorize_proteins(proteins)
        return self.model.predict([ligand_vectors, protein_vectors]).tolist()

    def save(self, path):
        self.model.save(f"{path}/model")

        with open(f"{path}/history.json", "w") as f:
            json.dump(self.history, f, indent=4)

        donot_copy = {"model", "history"}
        dct = {k: v for k, v in self.__dict__.items() if k not in donot_copy}
        with open(f"{path}/params.json", "w") as f:
            json.dump(dct, f, indent=4)

