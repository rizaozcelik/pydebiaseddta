import random
from typing import Any, Dict, List, Tuple, Type, Union
from pathlib import Path

import numpy as np

from pydebiaseddta import guides, predictors

import pandas as pd
from ..utils import get_ranks


class DebiasedDTA:
    def __init__(
        self,
        guide_cls: Union[Type[guides.Guide], Type[predictors.Predictor]],
        predictor_cls: Union[Type[guides.Guide], Type[predictors.Predictor]],
        mini_val_frac: float = 0.2,
        n_bootstrapping: int = 10,
        guide_params: Dict = None,
        predictor_params: Dict = None,
        guide_error_exponent: float = 2.,
        weight_temperature: float = 1.,
        weight_tempering_exponent: float = 1.,
        weight_tempering_num_epochs: int = int(1e6),
        weight_prior: float = 0.01,
        weight_rank_based: bool = False,
        seed: int = 0,
    ):
        """Constructor to initiate a DebiasedDTA training framework. 
        
        Parameters
        ----------
        guide_cls : Union[Type[guides.Guide], Type[predictors.Predictor]]
            The `Guide` class for debiasing. Note that the input is not an instance, but a class, *e.g.*, `BoWDTA`, not `BoWDTA()`.
            The instance is created during the model training by the DebiasedDTA module. Passing a `Predictor` class will
            result in utilization of the training errors of this predictor.
        predictor_cls : Union[Type[guides.Guide], Type[predictors.Predictor]]
            Class of the `Predictor` to debias. Note that the input is not an instance, but a class, *e.g.*, `BPEDTA`, not `BPEDTA()`.
            The instance is created during the model training by the DebiasedDTA module. Passing a `Guide` class will
            result in use of that guide class as the predictor to allow baseline computations, however this will result in
            an error if the guide_cls is not None.
        mini_val_frac : float, optional
            Fraction of the validation data to separate for guide evaluation, by default 0.2
        n_bootstrapping : int, optional
            Number of times to train guides on the training set, by default 10
        guide_params : Dict, optional
            Parameter dictionary necessary to create the `Guide`. 
            The dictionary should map the name of the constructor parameters to their values. 
            An empty dictionary is used during the creation by default.
        predictor_params : Dict, optional
            Parameter dictionary necessary to create the `Predictor`. 
            The dictionary should map the name of the constructor parameters to their values, and
            `n_epochs` **must** be among the parameters for debiasing to work.
            An empty dictionary is used during the creation by default.
        guide_error_exponent : float, optional
            Exponent for computing the errors incurred by guide's predictions.
        weight_temperature : float, optional
            Temperature parameter for importance weights. After weights are
            prepared, they are exponentiated by 1/weight_temperature and 
            renormalized.
        weight_tempering_exponent : float, optional
            Controls the speed of tempering process. Lower numbers lead to quicker transition from uniform
            to computed importance weights.
        weight_tempering_num_epochs : int, optional
            Controls the total number of epochs in which to transition to computed importance weights.
            Especially relevant when early stopping is desired.
        weight_prior : float, optional
            Adds the given ratio of the maximum importance weight to all importance weights, i.e., sets importance
            weights to importance_weights + importance_weights.max() * weight_prior.
        weight_rank_based : bool, optional
            Instead of computing weights as directly proportional to guide error, setting them to percentile ranks
            of the errors of training inputs.
        seed : int, optional
            Seed for reproducibility of experiments.

        Raises
        ------
        ValueError
            A `ValueError` is raised if `n_epochs` is not among the predictor parameters.
        """
        self.guide_cls = guide_cls
        self.predictor_cls = predictor_cls
        self.mini_val_frac = mini_val_frac
        self.n_bootstrapping = n_bootstrapping

        self.guide_params = dict() if guide_params is None else guide_params
        self.predictor_params = {} if predictor_params is None else predictor_params
        self.predictor_instance = self.predictor_cls(**self.predictor_params)
        if ("n_epochs" not in self.predictor_instance.__dict__) and (self.guide_cls):
            raise ValueError(
                'The predictor must have a field named "n_epochs" to be debiased'
            )

        self.guide_error_exponent = guide_error_exponent    
        self.weight_tempering_exponent = weight_tempering_exponent
        self.weight_tempering_num_epochs = weight_tempering_num_epochs
        self.weight_temperature = weight_temperature
        self.weight_prior = weight_prior
        self.weight_rank_based=weight_rank_based
        self.predictor_instance = self.predictor_cls(seed=seed, **predictor_params)
        self.seed = seed

    @staticmethod
    def save_importance_weights(
        interactions: List[Tuple[int, Any, Any, float]], importance_weights: List[float], weights_save_path: str
    ):
        """Saves the importance weights learned by the `guide`.

        Parameters
        ----------
        interactions : List[Tuple[int, Any, Any, float]]
            The List of training interactions as a Tuple of interaction id (assigned by the guide),
            ligand, protein, and affinity score.
        importance_weights : List[float]
            The importance weight for each interaction.
        weights_save_path : str
            Path to save the weights.
        """
        try:
            Path(weights_save_path).mkdir(parents=True, exist_ok=True)
        except FileExistsError:
            pass
        dump_content = []
        for interaction_id, ligand, protein, label in interactions:
            importance_weight = importance_weights[interaction_id]
            dump_content.append(f"{ligand},{protein},{label},{importance_weight}")
        dump = "\n".join(dump_content)
        with open(weights_save_path, "w") as f:
            f.write(dump)

    def learn_importance_weights(
        self,
        train_ligands: List[Any],
        train_proteins: List[Any],
        train_labels: List[float],
        weights_save_path: str = None,
        weights_load_path = None
    ) -> List[float]:
        """Learns importance weights using the `Guide` model specified during the construction.

        Parameters
        ----------
        train_ligands : List[Any]
            List of the training ligands used by the `Guide` model. 
            DebiasedDTA training framework imposes no restriction on the representation type of the ligands.
        train_proteins : List[Any]
            List of the training proteins used by the `Guide` model. 
            DebiasedDTA training framework imposes no restriction on the representation type of the proteins.
        train_labels : List[float]
            Affinity scores of the training protein-ligand pairs.
        weights_save_path : str, optional
            Path to save the learned importance weights. By default `None` and the weights are not saved.
        weights_load_path : str, optional
            Path to load previously computed importance weights. By default `None` and the weights are newly computed.

        Returns
        -------
        List[float]
            The importance weights learned by the guide.
        """
        train_size = len(train_ligands)
        train_interactions = list(
            zip(range(train_size), train_ligands, train_proteins, train_labels,)
        )
        mini_val_data_size = int(train_size * self.mini_val_frac) + 1
        all_mini_val_errors = [[] for _ in range(train_size)]

        assert self.mini_val_frac > 0
        
        if weights_load_path:
            return pd.read_csv(weights_load_path, header=None).loc[:, 3].values
        elif self.guide_cls is None:
            # if a guide is being used as predictor to compute a baseline
            if issubclass(self.predictor_cls, guides.Guide):
                self.predictor_instance.n_epochs = 1
            return [1 for i in range(len(train_ligands))]
        elif self.guide_cls.__name__ == "OutDTA":
            self.guide_instance = self.guide_cls(ligands=train_ligands, proteins=train_proteins, **self.guide_params)
            self.guide_instance.train()
            importance_weights = self.guide_instance.get_importance_weights()
        elif self.guide_cls.__name__ == "RFDTA":
            guide_instance = self.guide_cls(**self.guide_params)
            guide_instance.train(train_ligands, train_proteins, train_labels)
            train_preds = guide_instance.predict_oob()
            importance_weights = np.abs(np.array(train_labels) - np.array(train_preds)) ** self.guide_error_exponent
        elif issubclass(self.guide_cls, predictors.Predictor):
            print(f"Using predictor {self.guide_cls.__name__} as guide.")
            self.guide_instance = self.guide_cls(**self.guide_params)
            self.guide_instance.train(train_ligands, train_proteins, train_labels)
            preds = self.guide_instance.predict(train_ligands, train_proteins)
            importance_weights = np.abs((np.array(train_labels) - np.array(preds).flatten())) ** self.guide_error_exponent
            print("Guide training completed.")
        else:
            for _ in range(self.n_bootstrapping):
                random.shuffle(train_interactions)
                n_mini_val = int(1 / self.mini_val_frac)
                assert n_mini_val > 1
                for mini_val_ix in range(n_mini_val):
                    val_start_ix = mini_val_ix * mini_val_data_size
                    val_end_ix = val_start_ix + mini_val_data_size
                    mini_val_interactions = train_interactions[val_start_ix:val_end_ix]
                    mini_train_interactions = (
                        train_interactions[:val_start_ix] + train_interactions[val_end_ix:]
                    )

                    mini_train_ligands = [
                        interaction[1] for interaction in mini_train_interactions
                    ]
                    mini_train_proteins = [
                        interaction[2] for interaction in mini_train_interactions
                    ]
                    mini_train_labels = [
                        interaction[3] for interaction in mini_train_interactions
                    ]
                    guide_instance = self.guide_cls(**self.guide_params)
                    guide_instance.train(
                        mini_train_ligands, mini_train_proteins, mini_train_labels,
                    )

                    mini_val_interaction_ids, mini_val_ligands, mini_val_proteins, mini_val_labels = [
                        [interaction[i] for interaction in mini_val_interactions] for i in range(4)
                        ]
                    mini_val_preds = guide_instance.predict(mini_val_ligands, mini_val_proteins)
                    mini_val_errors = np.abs(np.array(mini_val_labels) - np.array(mini_val_preds)) ** self.guide_error_exponent
                    for interaction_id, val_error in zip(mini_val_interaction_ids, mini_val_errors):
                        all_mini_val_errors[interaction_id].append(val_error)
            
            assert all([len(mini_val_errors) == self.n_bootstrapping for mini_val_errors in all_mini_val_errors])

            importance_weights = np.array([np.median(errors) for errors in all_mini_val_errors])

        if self.weight_rank_based:
            importance_weights = get_ranks(importance_weights)
        importance_weights += importance_weights.max() * self.weight_prior
        importance_weights = importance_weights ** (1/self.weight_temperature)
    
        importance_weights = list((importance_weights / importance_weights.sum()) * len(importance_weights))

        if weights_save_path is not None:
            DebiasedDTA.save_importance_weights(
                train_interactions, importance_weights, weights_save_path
            )
        return importance_weights

    def train(
        self,
        train_ligands: List[Any],
        train_proteins: List[Any],
        train_labels: List[float],
        val_splits: Dict[str, List[Union[List[str], List[float]]]] = {},
        weights_save_path: str = None,
        weights_load_path: str = None,
        metrics_tracked: List[str] = ["mse", "mae"],  
    ) -> Any:
        """Starts the DebiasedDTA training framework.
        The importance weights are learned with the guide and used to weight the samples during the predictor's training.
        Performance on the validation set is also measured, if provided.
        Parameters
        ----------
        train_ligands : List[Any]
            List of the training ligands used by the `Predictor`. 
            DebiasedDTA training framework imposes no restriction on the representation type of the ligands.
        train_proteins : List[Any]
            List of the training ligands used by the `Predictor`. 
            DebiasedDTA training framework imposes no restriction on the representation type of the proteins.
        train_labels : List[float]
            Affinity scores of the training protein-ligand pairs.
        val_splits : Dict[str, List[Union[List[str], List[float]]]], optional
            Dictionary that includes all desired validation splits. Keys denote the split name e.g.
            val_cold_both, and values include a list that include the ligands, proteins, and labels
            for the said split, in the style of the training lists provided to this function.
        weights_save_path : str, optional
            Path to save the learned importance weights. By default `None` and the weights are not saved.
        weights_load_path : str, optional
            Path to load previously computed importance weights. By default `None` and the weights are newly computed.
        metrics_tracked : List[str], optional
            List of metrics that are tracked during training. Available options are
            "mse", "rmse", "mae", "r2", "ci".


        Returns
        -------
        Any
            Output of the train function of the predictor.

        """
        random.seed(self.seed)
        np.random.seed(self.seed)

        train_ligands = train_ligands.copy()
        train_proteins = train_proteins.copy()
        assert len(train_ligands) == len(train_proteins)

        importance_weights = self.learn_importance_weights(
            train_ligands,
            train_proteins,
            train_labels,
            weights_save_path=weights_save_path,
            weights_load_path=weights_load_path,
        )
        n_epochs = self.predictor_instance.n_epochs
        iw = np.array(importance_weights)
        final_num_tem_epochs = min(n_epochs, self.weight_tempering_num_epochs)
        weights_by_epoch = [
            1 - ((min(e, final_num_tem_epochs) / final_num_tem_epochs) ** self.weight_tempering_exponent) + iw * ((min(e, final_num_tem_epochs) / final_num_tem_epochs) ** self.weight_tempering_exponent) for e in range(n_epochs)
        ] if final_num_tem_epochs > 0.5 else [iw for e in range(n_epochs)]

        return self.predictor_instance.train(
            train_ligands,
            train_proteins,
            train_labels,
            val_splits=val_splits,
            sample_weights_by_epoch=weights_by_epoch,
            metrics_tracked=metrics_tracked
        )
