from abc import ABC, abstractmethod
from typing import List, Any


class Guide(ABC):
    """An abstract class that implements the interface of a guide in `pydebiaseddta`.
    The guides are characterized by a `train` function and a `predict` function, 
    whose signatures are implemented by this class. 
    Any instance of the `Guide` class can be trained in the `DebiasedDTA` training framework,
    and therefore, `Guide` can be inherited to design custom guide models.
    """

    @abstractmethod
    def train(
        train_ligands: List[Any], train_proteins: List[Any], train_labels: List[float],
    ):
        """An abstract method to define the training interface of the guides.

        Parameters
        ----------
        train_ligands : List[Any]
            Training ligands in any representation.
        train_proteins : List[Any]
            Training proteins in any representation.
        train_labels : List[float]
            Affinity scores of the training protein-ligand pairs.
        """
        pass

    @abstractmethod
    def predict(ligands: List[Any], proteins: List[Any]) -> List[float]:
        """An abstract method to define the prediction interface of the guides.

        Parameters
        ----------
        ligands : List[Any]
            Ligands in any representation.
        proteins : List[Any]
            Proteins in any representation.

        Returns
        -------
        List[float]
            The predicted affinities.
        """
        pass
