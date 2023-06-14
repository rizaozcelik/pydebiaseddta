import re
from functools import lru_cache
from typing import List

import numpy as np
import transformers
from tensorflow.keras.layers import Input, Concatenate, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD
from transformers import AutoTokenizer, AutoModel

from .abstract_predictors import TFPredictor


class LMDTA(TFPredictor):
    def __init__(
            self,
            n_epochs: int = 200,
            learning_rate: float = 0.001,
            batch_size: int = 256,
            early_stopping_metric: str = "mse",
            early_stopping_metric_threshold: float = -1e6,
            early_stopping_num_epochs: int = 0,
            early_stopping_split: str = "train",
            model_folder: str = "",
            optimizer: str = "adam",
            min_epochs: int = 0,
            ):
        """Constructor to create a LMDTA instance.
        LMDTA represents ligands and proteins with pre-trained language model embeddings
        obtained via [`ChemBERTa`](https://arxiv.org/abs/2010.09885) and  [`ProtBert`](https://www.biorxiv.org/content/biorxiv/early/2020/07/21/2020.07.12.199554.full.pdf) models, respectively. 
        A fully-connected neural network with two layers is used afterwards to predict affinities.

        Parameters
        ----------
        n_epochs : int, optional
            Number of epochs to train the model, by default 200.
        learning_rate : float, optional
            Learning rate during optimization, by default 0.001.
        batch_size : int, optional
             Batch size during training, by default 256.
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
        model_folder : str, optional
            Folder for saving the model. Empty by default and not saving any models, also used for retrieving the
            best model if early_stopping_num_epochs > 0.
        optimizer : str, optional
            The optimizer used in training. Available options are "adam" and "sgd".
        min_epochs : int, optional
            Initial number of epochs for which the early stopping computations will be overrided.
        """
        transformers.logging.set_verbosity(transformers.logging.CRITICAL)
        self.ligand_tokenizer = AutoTokenizer.from_pretrained(
            "seyonec/PubChem10M_SMILES_BPE_450k"
        )
        self.chemberta = AutoModel.from_pretrained("seyonec/PubChem10M_SMILES_BPE_450k")

        self.protein_tokenizer = AutoTokenizer.from_pretrained(
            "Rostlab/prot_bert", do_lower_case=False
        )
        self.protbert = AutoModel.from_pretrained("Rostlab/prot_bert")
        TFPredictor.__init__(self, n_epochs, learning_rate, batch_size)

        self.optimizer = optimizer
        self.early_stopping_metric = early_stopping_metric
        self.early_stopping_metric_threshold = early_stopping_metric_threshold
        self.early_stopping_num_epochs = early_stopping_num_epochs
        self.early_stopping_split = early_stopping_split
        self.min_epochs = min_epochs
        self.model_folder = model_folder

    def build(self):
        """Builds a `LMDTA` predictor in `keras` with the parameters specified during construction.

        Returns
        -------
        tensorflow.keras.models.Model
            The built model.
        """
        ligands = Input(shape=(768,), dtype="float32")
        proteins = Input(shape=(1024,), dtype="float32")

        interaction_representation = Concatenate(axis=-1)([ligands, proteins])

        FC1 = Dense(1024, activation="relu")(interaction_representation)
        FC1 = Dropout(0.1)(FC1)
        FC2 = Dense(512, activation="relu")(FC1)
        predictions = Dense(1, kernel_initializer="normal")(FC2)

        if self.optimizer == "adam":
            opt = Adam(self.learning_rate)
        elif self.optimizer == "sgd":
            opt = SGD(self.learning_rate)
        else:
            raise ValueError(f"The optimizer {self.optimizer} is not found.")
        lmdta = Model(inputs=[ligands, proteins], outputs=[predictions])
        lmdta.compile(
            optimizer=opt, loss="mean_squared_error", metrics=["mean_squared_error"]
        )
        return lmdta

    @lru_cache(maxsize=2048)
    def get_chemberta_embedding(self, smiles: str) -> np.array:
        """Computes the [`ChemBERTa`](https://arxiv.org/abs/2010.09885) vector for a ligand. 
        Since the creating the vector is computation-heavy, an `lru_cache` of size 2048 is used to store produced vectors.

        Parameters
        ----------
        smiles : str
            SMILES string of the ligand.

        Returns
        -------
        np.array
            [`ChemBERTa`](https://arxiv.org/abs/2010.09885) vector (768-dimensional) of the ligand.
        """        
        tokens = self.ligand_tokenizer(smiles, return_tensors="pt")
        output = self.chemberta(**tokens)
        return output.last_hidden_state.detach().numpy().mean(axis=1)

    def vectorize_ligands(self, ligands: List[str]) -> np.array:
        """Vectorizes the ligands with [`ChemBERTa`](https://arxiv.org/abs/2010.09885) embeddings.

        Parameters
        ----------
        ligands : List[str]
            The SMILES strings of ligands.

        Returns
        -------
        np.array
            An $N \\times 768$ ($N$ is the number of the input ligands) matrix that contains [`ChemBERTa`](https://arxiv.org/abs/2010.09885) vectors of the ligands.
        """        
        return np.vstack(
            [self.get_chemberta_embedding(ligand) for ligand in ligands]
        )

    @lru_cache(maxsize=1024)
    def get_protbert_embedding(self, aa_sequence: str) -> np.array:
        """Computes the [`ProtBert`](https://www.biorxiv.org/content/biorxiv/early/2020/07/21/2020.07.12.199554.full.pdf) vector for a protein. 
        Since the creating the vector is computation-heavy, an `lru_cache` of size 2048 is used to store produced vectors.

        Parameters
        ----------
        aa_sequence : str
            Amino-acid sequence of the protein.

        Returns
        -------
        np.array
            [`ProtBert`](https://www.biorxiv.org/content/biorxiv/early/2020/07/21/2020.07.12.199554.full.pdf) vector (1024-dimensional) of the protein.
        """    
        pp_sequence = " ".join(aa_sequence)
        cleaned_sequence = re.sub(r"[UZOB]", "X", pp_sequence)
        tokens = self.protein_tokenizer(cleaned_sequence, return_tensors="pt")
        output = self.protbert(**tokens)
        return output.last_hidden_state.detach().numpy().mean(axis=1)

    def vectorize_proteins(self, aa_sequences: List[str]) -> np.array:
        """Vectorizes the proteins with [`ProtBert`](https://www.biorxiv.org/content/biorxiv/early/2020/07/21/2020.07.12.199554.full.pdf) embeddings.

        Parameters
        ----------
        aa_sequences : List[str]
            The amino-acid sequences of the proteins.

        Returns
        -------
        np.array
            An $N \\times 1024$ ($N$ is the number of the input proteins) matrix that contains [`ProtBert`](https://www.biorxiv.org/content/biorxiv/early/2020/07/21/2020.07.12.199554.full.pdf) vectors of the ligands.
        """   
        return np.vstack(
            [self.get_protbert_embedding(aa_sequence) for aa_sequence in aa_sequences]
        )
