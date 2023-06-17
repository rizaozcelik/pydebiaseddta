from typing import List
import numpy as np
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D
from tensorflow.keras.layers import Dense, Dropout, Concatenate
from tensorflow.keras.layers import Input, Embedding
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD

from .abstract_predictors import TFPredictor
from ..sequence.word_identification import (
    load_ligand_word_identifier,
    load_protein_word_identifier,
)
from ..sequence.smiles_processing import (
    smiles_to_unichar_batch,
    load_smiles_to_unichar_encoding,
)


class DeepDTA(TFPredictor):
    def __init__(
        self,
        max_smi_len: int = 100,
        max_prot_len: int = 1000,
        embedding_dim: int = 128,
        learning_rate: float = 0.001,
        batch_size: int = 256,
        n_epochs: int = 200,
        num_filters: int = 32,
        smi_filter_len: int = 4,
        prot_filter_len: int = 6,
        early_stopping_metric: str = "mse",
        early_stopping_metric_threshold: float = -1e6,
        early_stopping_num_epochs: int = 0,
        early_stopping_split: str = "train",
        optimizer: str = "adam",
        min_epochs: int = 0,
        seed: int = 0,
        **kwargs
    ):
        """Constructor to create a DeepDTA instance.
        DeepDTA segments SMILES strings of ligands and amino-acid sequences of proteins into characters,
        and applies three layers of convolutions to learn latent representations. 
        A fully-connected neural network with three layers is used afterwards to predict affinities.

        Parameters
        ----------
        max_smi_len : int, optional
            Maximum number of characters in a SMILES string, by default 100. 
            Longer SMILES strings are truncated.
        max_prot_len : int, optional
            Maximum number of amino-acids a protein sequence, by default 1000. 
            Longer sequences are truncated.
        embedding_dim : int, optional
            The dimension of the biomolecule characters, by default 128.
        learning_rate : float, optional
            Learning rate during optimization, by default 0.001.
        batch_size : int, optional
            Batch size during training, by default 256.
        n_epochs : int, optional
            Number of epochs to train the model, by default 200.
        num_filters : int, optional
            Number of filters in the first convolution block. The next blocks use two and three times of this number, respectively. y default 32.
        smi_filter_len : int, optional
            Length of filters in the convolution blocks for ligands, by default 4.
        prot_filter_len : int, optional
            Length of filters in the convolution blocks for proteins, by default 6.
        early_stopping_metric : str, optional
            Metric for early stopping of the training. Available options are "mse", "rmse", "mae", "r2", "ci".
        early_stopping_metric_threshold : float, optional
            If the performance in the specified metric in the specified split 
            fall below this value the training is stopped early. Available options are
            "mse" and "mae". Set to a negative value by default with no effects.
        early_stopping_num_epochs : int, optional
            If this value is set > 0, then if the training has not been better than the
            best recorded performance for this many epochs on the specified split, the
            training is stopped early. Set to 0 by default with no effect.
        early_stopping_split:
            The split for conducting early stopping checks. Available options are "train"
            and the keys in the val_split dictionary.
        optimizer : str, optional
            The optimizer used in training. Available options are "adam" and "sgd".
        min_epochs : int, optional
            Initial number of epochs for which the early stopping computations will be overrided.
        seed : int, optional
            Seed for the initialized model.
        """    
        self.max_smi_len = max_smi_len
        self.max_prot_len = max_prot_len
        self.embedding_dim = embedding_dim
        self.num_filters = num_filters
        self.smi_filter_len = smi_filter_len
        self.prot_filter_len = prot_filter_len

        self.optimizer = optimizer
        self.early_stopping_metric = early_stopping_metric
        self.early_stopping_metric_threshold = early_stopping_metric_threshold
        self.early_stopping_num_epochs = early_stopping_num_epochs
        self.early_stopping_split = early_stopping_split
        self.min_epochs = min_epochs

        self.chem_vocab_size = 94
        self.prot_vocab_size = 26
        TFPredictor.__init__(self, n_epochs, learning_rate, batch_size, seed=seed)

    def build(self):
        """Builds a `DeepDTA` predictor in `keras` with the parameters specified during construction.

        Returns
        -------
        tensorflow.keras.models.Model
            The built model.
        """    
        # Inputs
        ligands = Input(shape=(self.max_smi_len,), dtype="int32")

        # ligand representation
        ligand_representation = Embedding(
            input_dim=self.chem_vocab_size + 1,
            output_dim=self.embedding_dim,
            input_length=self.max_smi_len,
            mask_zero=True,
        )(ligands)
        ligand_representation = Conv1D(
            filters=self.num_filters,
            kernel_size=self.smi_filter_len,
            activation="relu",
            padding="valid",
            strides=1,
        )(ligand_representation)
        ligand_representation = Conv1D(
            filters=self.num_filters * 2,
            kernel_size=self.smi_filter_len,
            activation="relu",
            padding="valid",
            strides=1,
        )(ligand_representation)
        ligand_representation = Conv1D(
            filters=self.num_filters * 3,
            kernel_size=self.smi_filter_len,
            activation="relu",
            padding="valid",
            strides=1,
        )(ligand_representation)
        ligand_representation = GlobalMaxPooling1D()(ligand_representation)

        # Protein representation
        proteins = Input(shape=(self.max_prot_len,), dtype="int32")
        protein_representation = Embedding(
            input_dim=self.prot_vocab_size + 1,
            output_dim=self.embedding_dim,
            input_length=self.max_prot_len,
            mask_zero=True,
        )(proteins)
        protein_representation = Conv1D(
            filters=self.num_filters,
            kernel_size=self.prot_filter_len,
            activation="relu",
            padding="valid",
            strides=1,
        )(protein_representation)
        protein_representation = Conv1D(
            filters=self.num_filters * 2,
            kernel_size=self.prot_filter_len,
            activation="relu",
            padding="valid",
            strides=1,
        )(protein_representation)
        protein_representation = Conv1D(
            filters=self.num_filters * 3,
            kernel_size=self.prot_filter_len,
            activation="relu",
            padding="valid",
            strides=1,
        )(protein_representation)
        protein_representation = GlobalMaxPooling1D()(protein_representation)

        interaction_representation = Concatenate(axis=-1)(
            [ligand_representation, protein_representation]
        )

        # Fully connected layers
        FC1 = Dense(1024, activation="relu")(interaction_representation)
        FC1 = Dropout(0.1)(FC1)
        FC2 = Dense(1024, activation="relu")(FC1)
        FC2 = Dropout(0.1)(FC2)
        FC3 = Dense(512, activation="relu")(FC2)
        predictions = Dense(1, kernel_initializer="normal")(FC3)

        if self.optimizer == "adam":
            opt = Adam(self.learning_rate)
        elif self.optimizer == "sgd":
            opt = SGD(self.learning_rate)
        else:
            raise ValueError(f"The optimizer {self.optimizer} is not found.")
        deepdta = Model(inputs=[ligands, proteins], outputs=[predictions])
        deepdta.compile(
            optimizer=opt,
            loss="mean_squared_error",
            metrics=["mean_squared_error"],
        )
        return deepdta

    def vectorize_ligands(self, ligands: List[str]) -> np.array:
        """Segments SMILES strings of ligands into characters and applies label encoding.
        Truncation and padding are also applied to prepare ligands for training and/or prediction.

        Parameters
        ----------
        ligands : List[str]
            The SMILES strings of ligands.

        Returns
        -------
        np.array
            An $N \\times max\\_smi\\_len$ ($N$ is the number of the input ligands) matrix that contains label encoded sequences of SMILES tokens.
        """     
        smi_to_unichar_encoding = load_smiles_to_unichar_encoding()
        unichars = smiles_to_unichar_batch(ligands, smi_to_unichar_encoding)
        word_identifier = load_ligand_word_identifier(vocabulary_size=94)

        return np.array(
            word_identifier.encode_sequences(unichars, self.max_smi_len)
        )

    def vectorize_proteins(self, aa_sequences: List[str]) -> np.array:
        """Segments amino-acid sequences of proteins into characters and applies label encoding.
        Truncation and padding are also applied to prepare proteins for training and/or prediction.

        Parameters
        ----------
        aa_sequences : List[str]
            The amino-acid sequences of proteins.

        Returns
        -------
        np.array
            An $N \\times max\\_prot\\_len$ ($N$ is the number of the input proteins) matrix that contains label encoded sequences of amino-acids.
        """
        word_identifier = load_protein_word_identifier(vocabulary_size=26)
        return np.array(
            word_identifier.encode_sequences(aa_sequences, self.max_prot_len)
        )


if __name__ == "__main__":

    from pydebiaseddta.utils import load_sample_dta_data

    train_data = load_sample_dta_data(mini=True)["train"]
    train_ligands, train_proteins, train_labels = train_data["smiles"], train_data["aa_sequence"], train_data["affinity_score"]
    deepdta = DeepDTA(n_epochs=5)
    deepdta.train(train_ligands, train_proteins, train_labels)
