from typing import List
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from tensorflow.keras.preprocessing.text import Tokenizer

from .base_guide import BaseGuide
from ..sequence.word_identification import (
    load_chemical_word_identifier,
    load_protein_word_identifier,
)
from ..sequence.smiles_processing import (
    smiles_to_unichar_batch,
    load_smiles_to_unichar_encoding,
)


class BoWDTA(BaseGuide):
    def __init__(self):
        self.chemical_bow_vectorizer = Tokenizer(
            filters=None, lower=False, oov_token="C"
        )
        self.protein_bow_vectorizer = Tokenizer(
            filters=None, lower=False, oov_token="$"
        )
        self.prediction_model = DecisionTreeRegressor()

    def tokenize_chemicals(self, smiles: List[str]):
        smi_to_unichar_encoding = load_smiles_to_unichar_encoding()
        unichars = smiles_to_unichar_batch(smiles, smi_to_unichar_encoding)
        word_identifier = load_chemical_word_identifier(vocab_size=8000)

        return word_identifier.encode_sequences(unichars, 100)

    def tokenize_proteins(self, proteins: List[str]):
        word_identifier = load_protein_word_identifier(vocab_size=32000)
        return word_identifier.encode_sequences(proteins, 1000)

    def vectorize_chemicals(self, smiles_words):
        return self.chemical_bow_vectorizer.texts_to_matrix(
            smiles_words, mode="freq"
        )

    def vectorize_proteins(self, proteins):
        return self.protein_bow_vectorizer.texts_to_matrix(
            proteins, mode="freq"
        )

    def train(
        self,
        train_chemicals: List[str],
        train_proteins: List[str],
        train_labels: List[float],
    ):
        tokenized_chemicals = self.tokenize_chemicals(train_chemicals)
        tokenized_proteins = self.tokenize_proteins(train_proteins)
        self.chemical_bow_vectorizer.fit_on_texts(tokenized_chemicals)
        self.protein_bow_vectorizer.fit_on_texts(tokenized_proteins)

        chemical_vectors = self.vectorize_chemicals(tokenized_chemicals)
        protein_vectors = self.vectorize_proteins(tokenized_proteins)
        X_train = np.hstack([chemical_vectors, protein_vectors])
        self.prediction_model.fit(X_train, train_labels)

    def predict(
        self, chemicals: List[str], proteins: List[str]
    ) -> List[float]:
        tokenized_chemicals = self.tokenize_chemicals(chemicals)
        tokenized_proteins = self.tokenize_proteins(proteins)

        chemical_vectors = self.vectorize_chemicals(tokenized_chemicals)
        protein_vectors = self.vectorize_proteins(tokenized_proteins)

        interaction = np.hstack([chemical_vectors, protein_vectors])
        return self.prediction_model.predict(interaction).tolist()


if __name__ == "__main__":

    from pydebiaseddta.utils import load_sample_dta_data

    train_chemicals, train_proteins, train_labels = load_sample_dta_data(
        mini=True
    )["train"]
    bowdta = BoWDTA()
    bowdta.train(train_chemicals, train_proteins, train_labels)
    preds = bowdta.predict(train_chemicals, train_proteins)
