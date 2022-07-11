from abc import ABC, abstractmethod
from typing import List, Any


class BaseGuide(ABC):
    @abstractmethod
    def train(
        train_chemicals: List[Any],
        train_proteins: List[Any],
        train_labels: List[float],
    ):
        pass

    @abstractmethod
    def predict(chemicals: List[Any], proteins: List[Any]) -> List[float]:
        pass
