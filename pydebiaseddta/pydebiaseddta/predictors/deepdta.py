from typing import List
import numpy as np
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D
from tensorflow.keras.layers import Dense, Dropout, Concatenate
from tensorflow.keras.layers import Input, Embedding
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from .base_predictors import TFPredictor
from ..sequence.word_identification import (
    load_chemical_word_identifier,
    load_protein_word_identifier,
)
from ..sequence.smiles_processing import (
    smiles_to_unichar_batch,
    load_smiles_to_unichar_encoding,
)


class DeepDTA(TFPredictor):
    def __init__(
        self,
        max_smi_len=100,
        max_prot_len=1000,
        embedding_dim=128,
        learning_rate=0.001,
        batch_size=256,
        n_epochs=200,
        num_filters=32,
        smi_filter_len=4,
        prot_filter_len=6,
    ):
        self.max_smi_len = max_smi_len
        self.max_prot_len = max_prot_len
        self.embedding_dim = embedding_dim
        self.num_filters = num_filters
        self.smi_filter_len = smi_filter_len
        self.prot_filter_len = prot_filter_len

        self.chem_vocab_size = 94
        self.prot_vocab_size = 26
        TFPredictor.__init__(self, n_epochs, learning_rate, batch_size)

    def build(self):
        # Inputs
        chemicals = Input(shape=(self.max_smi_len,), dtype="int32")

        # chemical representation
        chemical_representation = Embedding(
            input_dim=self.chem_vocab_size + 1,
            output_dim=self.embedding_dim,
            input_length=self.max_smi_len,
            mask_zero=True,
        )(chemicals)
        chemical_representation = Conv1D(
            filters=self.num_filters,
            kernel_size=self.smi_filter_len,
            activation="relu",
            padding="valid",
            strides=1,
        )(chemical_representation)
        chemical_representation = Conv1D(
            filters=self.num_filters * 2,
            kernel_size=self.smi_filter_len,
            activation="relu",
            padding="valid",
            strides=1,
        )(chemical_representation)
        chemical_representation = Conv1D(
            filters=self.num_filters * 3,
            kernel_size=self.smi_filter_len,
            activation="relu",
            padding="valid",
            strides=1,
        )(chemical_representation)
        chemical_representation = GlobalMaxPooling1D()(chemical_representation)

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
            [chemical_representation, protein_representation]
        )

        # Fully connected layers
        FC1 = Dense(1024, activation="relu")(interaction_representation)
        FC1 = Dropout(0.1)(FC1)
        FC2 = Dense(1024, activation="relu")(FC1)
        FC2 = Dropout(0.1)(FC2)
        FC3 = Dense(512, activation="relu")(FC2)
        predictions = Dense(1, kernel_initializer="normal")(FC3)

        opt = Adam(self.learning_rate)
        deepdta = Model(inputs=[chemicals, proteins], outputs=[predictions])
        deepdta.compile(
            optimizer=opt,
            loss="mean_squared_error",
            metrics=["mean_squared_error"],
        )
        return deepdta

    def vectorize_chemicals(self, chemicals: List[str]):
        smi_to_unichar_encoding = load_smiles_to_unichar_encoding()
        unichars = smiles_to_unichar_batch(chemicals, smi_to_unichar_encoding)
        word_identifier = load_chemical_word_identifier(vocab_size=94)

        return np.array(
            word_identifier.encode_sequences(unichars, self.max_smi_len)
        )

    def vectorize_proteins(self, aa_sequences: List[str]):
        word_identifier = load_protein_word_identifier(vocab_size=26)
        return np.array(
            word_identifier.encode_sequences(aa_sequences, self.max_prot_len)
        )


if __name__ == "__main__":

    from pydebiaseddta.utils import load_sample_dta_data

    train_chemicals, train_proteins, train_labels = load_sample_dta_data(
        mini=True
    )["train"]
    deepdta = DeepDTA(n_epochs=5)
    deepdta.train(train_chemicals, train_proteins, train_labels)
