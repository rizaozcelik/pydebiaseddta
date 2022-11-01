from typing import List
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from tensorflow.keras.preprocessing.text import Tokenizer

from .abstract_guide import Guide
from ..sequence.word_identification import (
    load_ligand_word_identifier,
    load_protein_word_identifier,
)
from ..sequence.smiles_processing import (
    smiles_to_unichar_batch,
    load_smiles_to_unichar_encoding,
)


class BoWDTA(Guide):
    def __init__(self):
        """Constructor to create a BoWDTA model.
        BoWDTA represents the proteins and ligands as "bag-of-words`
        and uses a decision tree for prediction. BoWDTA uses the same biomolecule vocabulary
        as BPEDTA.
        """        
        self.ligand_bow_vectorizer = Tokenizer(
            filters=None, lower=False, oov_token="C"
        )
        self.protein_bow_vectorizer = Tokenizer(
            filters=None, lower=False, oov_token="$"
        )
        self.prediction_model = DecisionTreeRegressor()

    def tokenize_ligands(self, smiles: List[str]) -> List[List[int]]:
        """Segments SMILES strings of the ligands into their ligand words and applies label encoding.

        Parameters
        ----------
        smiles : List[str]
            The SMILES strings of the ligands

        Returns
        -------
        List[List[int]]
            Label encoded sequences of ligand words.
        """
        smi_to_unichar_encoding = load_smiles_to_unichar_encoding()
        unichars = smiles_to_unichar_batch(smiles, smi_to_unichar_encoding)
        word_identifier = load_ligand_word_identifier(vocab_size=8000)

        return word_identifier.encode_sequences(unichars, 100)

    def tokenize_proteins(self, aa_sequences: List[str]) -> List[List[int]]:
        """Segments amino-acid sequences of the proteins into their protein words and applies label encoding.

        Parameters
        ----------
        aa_sequences : List[str]
            The amino-acid sequences of the proteins.

        Returns
        -------
        List[List[int]]
            Label encoded sequences of protein words.
        """
        word_identifier = load_protein_word_identifier(vocab_size=32000)
        return word_identifier.encode_sequences(aa_sequences, 1000)

    def vectorize_ligands(self, smiles_words: List[List[int]]) -> np.array:
        """Computes bag-of-words vectors of the ligands based on their frequency.

        Parameters
        ----------
        smiles_words : List[List[int]]
            ligand words of each ligand as a sequence of sequences.

        Returns
        -------
        np.array
            Bag-of-words vectors stacked in a matrix.
        """        
        return self.ligand_bow_vectorizer.texts_to_matrix(
            smiles_words, mode="freq"
        )

    def vectorize_proteins(self, protein_words: List[List[int]]) -> np.array:
        """Computes bag-of-words vectors of the proteins based on their frequency.

        Parameters
        ----------
        protein_words : List[List[int]]
            Protein words of each protein as a sequence of sequences.

        Returns
        -------
        np.array
            Bag-of-words vectors stacked in a matrix.
        """  
        return self.protein_bow_vectorizer.texts_to_matrix(
            protein_words, mode="freq"
        )

    def train(
        self,
        train_ligands: List[str],
        train_proteins: List[str],
        train_labels: List[float],
    ):
        """Trains a BoWDTA model on the provided protein-ligand interactions.
        The biomolecules are represented as bag of their biomolecule words and a
        decision tree is used for affinity prediction.

        Parameters
        ----------
        train_ligands : List[str]
            SMILES strings of the training ligands.
        train_proteins : List[str]
            Amino-acid sequences of the training ligands.
        train_labels : List[float]
            Affinity scores of the training interactions.
        """    
        tokenized_ligands = self.tokenize_ligands(train_ligands)
        tokenized_proteins = self.tokenize_proteins(train_proteins)
        self.ligand_bow_vectorizer.fit_on_texts(tokenized_ligands)
        self.protein_bow_vectorizer.fit_on_texts(tokenized_proteins)

        ligand_vectors = self.vectorize_ligands(tokenized_ligands)
        protein_vectors = self.vectorize_proteins(tokenized_proteins)
        X_train = np.hstack([ligand_vectors, protein_vectors])
        self.prediction_model.fit(X_train, train_labels)

    def predict(
        self, ligands: List[str], proteins: List[str]
    ) -> List[float]:
        """Predicts the affinities of a list of protein-ligand pairs.

        Parameters
        ----------
        ligands : List[str]
            SMILES strings of the ligands.
        proteins : List[str]
            Amino-acid sequences of the proteins.

        Returns
        -------
        List[float]
            Predicted affinities.
        """    
        tokenized_ligands = self.tokenize_ligands(ligands)
        tokenized_proteins = self.tokenize_proteins(proteins)

        ligand_vectors = self.vectorize_ligands(tokenized_ligands)
        protein_vectors = self.vectorize_proteins(tokenized_proteins)

        interaction = np.hstack([ligand_vectors, protein_vectors])
        return self.prediction_model.predict(interaction).tolist()


if __name__ == "__main__":

    from pydebiaseddta.utils import load_sample_dta_data

    train_ligands, train_proteins, train_labels = load_sample_dta_data(
        mini=True
    )["train"]
    bowdta = BoWDTA()
    bowdta.train(train_ligands, train_proteins, train_labels)
    preds = bowdta.predict(train_ligands, train_proteins)
