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
VOCAB_SIZES_LIGAND = {"high": 8000, "mid": 400, "low":94}
VOCAB_SIZES_PROTEIN = {"high": 32000, "mid": 1600, "low":26}

class BoWDTA(Guide):
    def __init__(
            self,
            max_depth: int = None,
            min_samples_split: int = 2,
            min_samples_leaf: int = 1,
            criterion: str = "squared_error",
            vocab_size: str = "high",
            ligand_vector_mode: str = "freq",
            prot_vector_mode: str = "freq",
            input_rank=0,
            **kwargs):
        """Constructor to create a BoWDTA model.
        BoWDTA represents the proteins and ligands as "bag-of-words`
        and uses a decision tree for prediction. BoWDTA uses the same biomolecule vocabulary
        as BPEDTA.

        Parameters
        ----------
        max_depth : int, optional
            Determines the maximum depth of the decision tree regressor.
        min_samples_split : int, optional
            Determines the minimum samples to split a leaf for the decision
            tree regressor.
        min_samples_leaf : int, optional
            Determines the minimum samples a leaf can have for the decision
            tree regressor.
        criterion : str, optional
            Criterion according to which the decision tree regressor will be trained.
        vocab_size : str, optional
            Vocabulary size that will be used for tokenizing ligands and proteins.
        ligand_vector_mode : str, optional
            Method to use when creating the matrix representation for ligand tokens.
        prot_vector_mode : str, optional
            Method to use when creating the matrix representation for protein tokens.
        input_rank : int, optional
            If set > 0, uses a low-rank approximation of the input representation with
            the given rank.
        """        
        self.ligand_bow_vectorizer = Tokenizer(
            filters=None, lower=False, oov_token="C"
        )
        self.protein_bow_vectorizer = Tokenizer(
            filters=None, lower=False, oov_token="$"
        )
        self.prediction_model = DecisionTreeRegressor(
            max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, criterion=criterion
            )
        self.input_rank = input_rank
        self.vocab_size = vocab_size
        self.ligand_vector_mode = ligand_vector_mode
        self.prot_vector_mode = prot_vector_mode
    
    def tokenize_ligands(self, smiles: List[str], vocab_size: str = "high") -> List[List[int]]:
        """Segments SMILES strings of the ligands into their ligand words and applies label encoding.

        Parameters
        ----------
        smiles : List[str]
            The SMILES strings of the ligands
        vocab_size : str, optional
            Vocabulary size that will be used for tokenizing ligands.

        Returns
        -------
        List[List[int]]
            Label encoded sequences of ligand words.
        """
        smi_to_unichar_encoding = load_smiles_to_unichar_encoding()
        unichars = smiles_to_unichar_batch(smiles, smi_to_unichar_encoding)
        word_identifier = load_ligand_word_identifier(vocab_size=VOCAB_SIZES_LIGAND[vocab_size])

        return word_identifier.encode_sequences(unichars, 100)

    def tokenize_proteins(self, aa_sequences: List[str], vocab_size: str = "high"):
        """Segments amino-acid sequences of the proteins into their protein words and applies label encoding.

        Parameters
        ----------
        aa_sequences : List[str]
            The amino-acid sequences of the proteins.
        vocab_size : str, optional
            Vocabulary size that will be used for tokenizing proteins.

        Returns
        -------
        List[List[int]]
            Label encoded sequences of protein words.
        """
        word_identifier = load_protein_word_identifier(vocab_size=VOCAB_SIZES_PROTEIN[vocab_size])
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
            smiles_words, mode=self.ligand_vector_mode
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
            protein_words, mode=self.prot_vector_mode
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
        tokenized_ligands = self.tokenize_ligands(train_ligands, self.vocab_size)
        tokenized_proteins = self.tokenize_proteins(train_proteins, self.vocab_size)
        self.ligand_bow_vectorizer.fit_on_texts(tokenized_ligands)
        self.protein_bow_vectorizer.fit_on_texts(tokenized_proteins)

        ligand_vectors = self.vectorize_ligands(tokenized_ligands)
        protein_vectors = self.vectorize_proteins(tokenized_proteins)

        if self.input_rank > 0:
            R = self.input_rank
            U, s, Vh = np.linalg.svd(ligand_vectors, full_matrices=False)
            ligand_vectors = U[:, :R] @ np.diag(s[:R]) @ Vh[:R]
            U, s, Vh = np.linalg.svd(protein_vectors, full_matrices=False)
            protein_vectors  = U[:, :R] @ np.diag(s[:R]) @ Vh[:R]

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
        tokenized_ligands = self.tokenize_ligands(ligands, self.vocab_size)
        tokenized_proteins = self.tokenize_proteins(proteins, self.vocab_size)

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
