import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

from .base_guide import BaseGuide


def _list_to_numpy(lst):
    return np.array(lst).reshape(-1, 1)


class IDDTA(BaseGuide):
    def __init__(self):
        self.prediction_model = DecisionTreeRegressor()
        self.chemical_encoder = OneHotEncoder(handle_unknown="ignore")
        self.protein_encoder = OneHotEncoder(handle_unknown="ignore")

    def vectorize_chemicals(self, chemicals):
        chemicals = np.array(chemicals).reshape(-1, 1)
        return self.chemical_encoder.transform(chemicals).todense()

    def vectorize_proteins(self, proteins):
        proteins = np.array(proteins).reshape(-1, 1)
        return self.protein_encoder.transform(proteins).todense()

    def train(self, train_chemicals, train_proteins, train_labels):
        chemical_vecs = self.chemical_encoder.fit_transform(
            _list_to_numpy(train_chemicals)
        ).todense()
        protein_vecs = self.protein_encoder.fit_transform(
            _list_to_numpy(train_proteins)
        ).todense()

        X_train = np.hstack([chemical_vecs, protein_vecs])
        self.prediction_model.fit(X_train, train_labels)

    def predict(self, chemicals, proteins):
        chemical_vecs = self.vectorize_chemicals(chemicals)
        protein_vecs = self.vectorize_proteins(proteins)
        X_test = np.hstack([chemical_vecs, protein_vecs])
        return self.prediction_model.predict(X_test)
