from __future__ import annotations
from typing import Any, Dict, List, Union
from pathlib import Path
import json
from abc import ABC, abstractmethod
import shutil
from tqdm import tqdm

import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from ..evaluation import evaluate_predictions
import pandas as pd


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
        val_splits: Dict[str, List[Union[List[str], List[float]]]] = {},
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
        val_splits : Dict[str, List[Union[List[str], List[float]]]], optional
            Dictionary that includes all desired validation splits. Keys denote the split name e.g.
            val_cold_both, and values include a list that include the ligands, proteins, and labels
            for the said split, in the style of the training lists provided to this function.
        sample_weights_by_epoch : List[np.array], optional
            Weight of each training protein-ligand pair during training across epochs.
            This variable must be a List of size $E$ (number of training epochs),
            in which each element is a `np.array` of $N\times 1$, where $N$ is the training set size and 
            each element corresponds to the weight of a training sample.

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
    def __init__(self, n_epochs: int, learning_rate: float, batch_size: int, seed: int = 0, **kwargs):
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
        tf.random.set_seed(seed)
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
    def from_file(cls, path: str, params: Dict[str, Any] = None) -> TFPredictor:
        """A utility function to load a `TFPredictor` instance from disk.

        All attributes, including the model weights, are loaded. If a non-empty params
        dictionary is provided, previously used hyperparameters are ignored and new ones
        are instead used. This is done so that pretrained models can be used with new
        hyperparameters for any other purposes. To prevent incompatibilities due to previous
        tracked metrics and the new ones, the loaded history (if exists) is saved into
        a different property called `pretraining_history`.

        Parameters
        ----------
        path : str
            Path to load the prediction model from.
        params : Dict[str, Any]
            Dictionary of hyperparameters to be used. 

        Returns
        -------
        TFPredictor
            The previously saved model.
        """
        if params:
            print("New hyperparameters used for the pretrained predictor, any saved hyperparamaters are ignored.")
            dct = params
        else:
            try:
                with open(f"{path}/params.json") as f:
                    dct = json.load(f)
            except FileNotFoundError:
                print("Saved hyperparameters file not found, initializing pretrained predictor with default hyperparameters.")
                dct = {}

        instance = cls(**dct)

        try:
            with open(f"{path}/history.json") as f:
                instance.pretraining_history = json.load(f)
        except FileNotFoundError:
            print("Predictor pretraining history file not found, `pretraining_history` history is set to None.")
            instance.pretraining_history = None

        try:
            instance.model = tf.keras.models.load_model(f"{path}/model")        
        except FileNotFoundError:
            raise FileNotFoundError("Model not found. You cannot initialize a Predictor from file with no saved models.")
        return instance

    def train(
        self,
        train_ligands: List[str],
        train_proteins: List[str],
        train_labels: List[float],
        val_splits: Dict[str, List[Union[List[str], List[float]]]] = {},
        sample_weights_by_epoch: List[np.array] = None,
        metrics_tracked: List[str] = None,
        predictor_save_folder: str = None,
        seed: int = 0,
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
        val_splits : Dict[str, List[Union[List[str], List[float]]]], optional
            Dictionary that includes all desired validation splits. Keys denote the split name e.g.
            val_cold_both, and values include a list that include the ligands, proteins, and labels
            for the said split, in the style of the training lists provided to this function.
        sample_weights_by_epoch : List[np.array], optional
            Weight of each training protein-ligand pair during training across epochs.
            This variable must be a List of size $E$ (number of training epochs),
            in which each element is a `np.array` of $N\times 1$, where $N$ is the training set size and 
            each element corresponds to the weight of a training sample.
            By default `None` and no weighting is used.
        metrics_tracked : List[str], optional
            List of metrics that are tracked during training. Available options are
            "mse", "rmse", "mae", "r2", "ci".
        predictor_save_folder : str, optional
            If provided, this folder is used to save the trained predictor at the end of training.
            It is also used to cache and load the incumbent best performing model if early stopping is used.
        seed : int, optional
            Seed for the training procedure.

        Returns
        -------
        Dict
            Training history.
        """
        tf.random.set_seed(seed)

        if not metrics_tracked:
            metrics_tracked = ["mse", "mae"]

        if predictor_save_folder:
            Path(predictor_save_folder).mkdir(parents=True, exist_ok=True)

        if sample_weights_by_epoch is None:
            sample_weights_by_epoch = create_uniform_weights(
                len(train_ligands), self.n_epochs
            )

        train_ligand_vectors = self.vectorize_ligands(train_ligands)
        train_protein_vectors = self.vectorize_proteins(train_proteins)
        train_labels = np.array(train_labels)

        train_stats_over_epochs = {metric: [] for metric in metrics_tracked}
        val_stats_over_epochs = {split: {metric: [] for metric in metrics_tracked} for split in val_splits.keys()}
        assert self.early_stopping_metric in ["mse", "mae"]
        best_metric = 1e6
        best_metric_epoch = 0
        for e in tqdm(range(self.n_epochs)):
            self.model.fit(
                x=[train_ligand_vectors, train_protein_vectors],
                y=train_labels,
                sample_weight=sample_weights_by_epoch[e],
                batch_size=self.batch_size,
                epochs=1,
                verbose=0
            )
            train_stats = evaluate_predictions(
                gold_truths=train_labels,
                predictions=self.predict(train_ligands, train_proteins),
                metrics=list(train_stats_over_epochs.keys()),
            )
            for metric, stat in train_stats.items():
                train_stats_over_epochs[metric].append(np.round(stat, 6))

            for split in val_splits.keys():
                val_stats = evaluate_predictions(
                    gold_truths=val_splits[split][2],
                    predictions=self.predict(val_splits[split][0], val_splits[split][1]),
                    metrics=list(val_stats_over_epochs[split].keys()),
                )
                for metric, stat in val_stats.items():
                    val_stats_over_epochs[split][metric].append(np.round(stat, 6))
            
            if self.early_stopping_num_epochs > 0:
                current_metric = train_stats[self.early_stopping_metric] if self.early_stopping_split == "train" else val_stats_over_epochs[self.early_stopping_split][self.early_stopping_metric][-1]
                if current_metric < best_metric:
                    best_metric = current_metric
                    best_metric_epoch = e
                    if predictor_save_folder:
                        self.model.save(predictor_save_folder + "/temp_best_model")
                else:
                    if (e > self.min_epochs) and ((e - best_metric_epoch) == self.early_stopping_num_epochs):
                        tqdm.write(f"Early stopping due to no increase to {self.early_stopping_metric} in {self.early_stopping_split} split for {self.early_stopping_num_epochs} epochs.")
                        if predictor_save_folder:
                            self.model = tf.keras.models.load_model(predictor_save_folder + "/temp_best_model")
                            tqdm.write(f"Retrieved the best model from epoch {best_metric_epoch}.")
                            shutil.rmtree(predictor_save_folder + "/temp_best_model")
                        else:
                            tqdm.write("No save folder provided, using the final model.")
                        break
            
            if (self.early_stopping_metric_threshold) and (e > self.min_epochs):
                if self.early_stopping_split == "train":
                    current_metric = train_stats[self.early_stopping_metric]
                else:
                    current_metric = val_stats_over_epochs[self.early_stopping_split][self.early_stopping_metric][-1]
                if (current_metric < self.early_stopping_metric_threshold):
                    tqdm.write(f"Early stopping training due to convergence on the {self.early_stopping_split} split.")
                    break


        self.history["train"] = train_stats_over_epochs
        if val_stats_over_epochs is not None:
            self.history["val_splits"] = val_stats_over_epochs

        if predictor_save_folder:
            self.save(predictor_save_folder)
            print(f"Saved predictor to the folder {predictor_save_folder}.")
        return self.history

    def predict(self, ligands: List[str], proteins: List[str]) -> List[float]:
        """Predicts the affinities of a `List` of protein-ligand pairs via the trained DTA prediction model,
        *i.e.*, BPE-DTA, LM-DTA, and BPE-DTA. 

        Parameters
        ----------
        ligands : List[str]
            SMILES strings of the ligands.
        proteins : List[str]
            Amino-acid sequences of the proteins.

        Returns
        -------
        List[float]
            Predicted affinity scores by DTA prediction model.
        """        
        ligand_vectors = self.vectorize_ligands(ligands)
        protein_vectors = self.vectorize_proteins(proteins)
        return self.model.predict([ligand_vectors, protein_vectors]).tolist()

    def save(self, path: str):
        """A utility function to save a `TFPredictor` instance to the disk.
        All attributes, including the model weights, are saved.

        Parameters
        ----------
        path : str
            Path to save the predictor.
        """
        self.model.save(f"{path}/model")

        with open(f"{path}/history.json", "w") as f:
            json.dump(self.history, f, indent=4)

        donot_copy = {"model", "history"}
        dct = {k: v for k, v in self.__dict__.items() if k not in donot_copy}
        with open(f"{path}/params.json", "w") as f:
            json.dump(dct, f, indent=4)
