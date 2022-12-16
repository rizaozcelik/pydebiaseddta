import random
from typing import Any, Dict, List, Tuple, Type

import numpy as np

from pydebiaseddta import guides, predictors


class DebiasedDTA:
    def __init__(
        self,
        guide_cls: Type[guides.Guide],
        predictor_cls: Type[predictors.BasePredictor],
        mini_val_frac: float = 0.2,
        n_bootstrapping: int = 10,
        guide_params: Dict = None,
        predictor_params: Dict = None,
    ):
        """Constructor to initiate a DebiasedDTA training framework. 
        
        Parameters
        ----------
        guide_cls : Type[guides.AbstractGuide]
            The `Guide` class for debiasing. Note that the input is not an instance, but a class, *e.g.*, `BoWDTA`, not `BoWDTA()`.
            The instance is created during the model training by the DebiasedDTA module.
        predictor_cls : Type[predictors.BasePredictor]
            Class of the `Predictor` to debias. Note that the input is not an instance, but a class, *e.g.*, `BPEDTA`, not `BPEDTA()`.
            The instance is created during the model training by the DebiasedDTA module.
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
        if "n_epochs" not in self.predictor_instance.__dict__:
            raise ValueError(
                'The predictor must have a field named "n_epochs" to be debiased'
            )

    @staticmethod
    def save_importance_coefficients(
        interactions: List[Tuple[int, Any, Any, float]], importance_coefficients: List[float], savedir: str
    ):
        """Saves the importance coefficients learned by the `guide`.

        Parameters
        ----------
        interactions : List[Tuple[int, Any, Any, float]]
            The List of training interactions as a Tuple of interaction id (assigned by the guide),
            ligand, chemical, and affinity score.
        importance_coefficients : List[float]
            The importance coefficient for each interaction.
        savedir : str
            Path to save the coefficients.
        """    
        dump_content = []
        for interaction_id, ligand, protein, label in interactions:
            importance_coefficient = importance_coefficients[interaction_id]
            dump_content.append(f"{ligand},{protein},{label},{importance_coefficient}")
        dump = "\n".join(dump_content)
        with open(savedir) as f:
            f.write(dump)

    def learn_importance_coefficients(
        self,
        train_ligands: List[Any],
        train_proteins: List[Any],
        train_labels: List[float],
        savedir: str = None,
    ) -> List[float]:
        """Learns importance coefficients using the `Guide` model specified during the construction.

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
        savedir : str, optional
            Path to save the learned importance coefficients. By default `None` and the coefficients are not saved.

        Returns
        -------
        List[float]
            The importance coefficients learned by the guide.
        """
        train_size = len(train_ligands)
        train_interactions = list(
            zip(range(train_size), train_ligands, train_proteins, train_labels,)
        )
        mini_val_data_size = int(train_size * self.mini_val_frac) + 1
        interaction_id_to_sq_diff = [[] for _ in range(train_size)]

        for _ in range(self.n_bootstrapping):
            random.shuffle(train_interactions)
            n_mini_val = int(1 / self.mini_val_frac)
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

                mini_val_ligands = [
                    interaction[1] for interaction in mini_val_interactions
                ]
                mini_val_proteins = [
                    interaction[2] for interaction in mini_val_interactions
                ]
                preds = guide_instance.predict(mini_val_ligands, mini_val_proteins)
                mini_val_labels = [
                    interaction[3] for interaction in mini_val_interactions
                ]
                mini_val_sq_diffs = (np.array(mini_val_labels) - np.array(preds)) ** 2
                mini_val_interaction_ids = [
                    interaction[0] for interaction in mini_val_interactions
                ]
                for interaction_id, sq_diff in zip(
                    mini_val_interaction_ids, mini_val_sq_diffs
                ):
                    interaction_id_to_sq_diff[interaction_id].append(sq_diff)

        interaction_id_to_med_diff = [
            np.median(diffs) for diffs in interaction_id_to_sq_diff
        ]
        importance_coefficients = [
            med / sum(interaction_id_to_med_diff) for med in interaction_id_to_med_diff
        ]

        if savedir is not None:
            DebiasedDTA.save_importance_coefficients(
                train_interactions, importance_coefficients, savedir
            )

        return importance_coefficients

    def train(
        self,
        train_ligands: List[Any],
        train_proteins: List[Any],
        train_labels: List[float],
        val_ligands: List[Any] = None,
        val_proteins: List[Any] = None,
        val_labels: List[float] = None,
        coeffs_save_path: str = None,
    ) -> Any:
        """Starts the DebiasedDTA training framework.
        The importance coefficients are learned with the guide and used to weight the samples during the predictor's training.
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
        val_ligands : List[Any], optional
            Validation ligands to measure predictor performance, by default `None` and no validation is applied.
        val_proteins : List[Any], optional
            Validation proteins to measure predictor performance, by default `None` and no validation is applied.
        val_labels : List[float], optional
            Affinity scores of the Validatio pairs, by default `None` and no validation is applied.
        coeffs_save_path : str, optional
            Path to save importance coefficients learned by the `guide`. Defaults to `None` and no saving is performed.

        Returns
        -------
        Any
            Output of the train function of the predictor.

        """
        train_ligands = train_ligands.copy()
        train_proteins = train_proteins.copy()

        importance_coefficients = self.learn_importance_coefficients(
            train_ligands, train_proteins, train_labels, savedir=coeffs_save_path,
        )

        n_epochs = self.predictor_instance.n_epochs
        ic = np.array(importance_coefficients)
        weights_by_epoch = [
            1 - (e / n_epochs) + ic * (e / n_epochs) for e in range(n_epochs)
        ]

        if (
            val_ligands is not None
            and val_proteins is not None
            and val_labels is not None
        ):
            return self.predictor_instance.train(
                train_ligands,
                train_proteins,
                train_labels,
                val_ligands=val_ligands,
                val_proteins=val_proteins,
                val_labels=val_labels,
                sample_weights_by_epoch=weights_by_epoch,
            )

        return self.predictor_instance.train(
            train_ligands,
            train_proteins,
            train_labels,
            sample_weights_by_epoch=weights_by_epoch,
        )
