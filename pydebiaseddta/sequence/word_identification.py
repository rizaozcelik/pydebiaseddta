import json
from typing import List
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

from .. import package_path
from ..utils import load_json, save_json

FILE_EXTENSION = ".json"


class WordIdentifier:
    """A versatile class to identify biomolecule words in biomolecule strings. 
    `WordIdentifier` leverages the Byte Pair Encoding algorithm implemented in the `tokenizers` library
    to learn biomolecule vocabularies and segment biomolecule strings into their words. 
    """    
    def __init__(self, vocab_size: int):
        """Creates a `WordIdentifier` instance.

        Parameters
        ----------
        vocab_size : int
            Size of the biomolecule vocabulary.
        """        
        self.vocab_size = vocab_size
        self.tokenizer = Tokenizer(BPE())
        self.tokenizer.pre_tokenizer = Whitespace()

    @classmethod
    def from_file(cls, loadpath: str):
        """Loads a `WordIdentifier` from a file.

        Parameters
        ----------
        loadpath : str
            Path to the `WordIdentifier` file.

        Returns
        -------
        WordIdentifier
            Previously saved `WordIdentifier`
        """        
        if not loadpath.endswith(FILE_EXTENSION):
            loadpath = loadpath + FILE_EXTENSION

        dct = load_json(loadpath)
        vocab_size = len(dct["model"]["vocab"])
        instance = cls(vocab_size)
        instance.tokenizer = Tokenizer.from_str(json.dumps(dct))
        return instance

    def train(self, corpus_path: str):
        """Learns a biomolecule vocabulary from a file of biomolecule strings using Byte Pair Encoding Algorithm. 

        Parameters
        ----------
        corpus_path : str
            Path to the corpus of biomolecule strings. The corpus file must contain a biomolecule string per line.
        """        
        trainer = BpeTrainer(
            vocab_size=self.vocab_size, special_tokens=["[PAD]"]
        )
        self.tokenizer.train([corpus_path], trainer)
        if self.tokenizer.get_vocab_size() < self.vocab_size:
            print(
                f"Warning: The iterations stopped before the desired vocab size is reached. Learned vocab size={self.tokenizer.get_vocab_size()}. Desired size={self.vocab_size}"
            )

    def tokenize_sequences(self, sequences: List[str]) -> List[List[str]]:
        """Segments a List of biomolecule strings into biomolecule words via the learned vocabulary.

        Parameters
        ----------
        sequences : List[str]
            The List of biomolecule strings.

        Returns
        -------
        List[List[str]]
            List of biomolecule words of each input string.
        """        
        encodings = self.tokenizer.encode_batch(sequences)
        return [encoding.tokens for encoding in encodings]

    def encode_sequences(self, sequences: List[str], padding_len: int = None)-> List[List[int]]:
        """Segments a List of biomolecule strings into biomolecule words via the learned vocabulary and 
        returns the id of the biomolecule word, which is convenient to apply label encoding in the subsequent steps.
        Padding support is also available to ease training deep learning possible.

        Parameters
        ----------
        sequences : List[str]
            The List of biomolecule strings.
        padding_len : int, optional
            The desired length of sequences, by default `None`. No padding is applied when set to `None`.

        Returns
        -------
        List[List[int]]
            List of the id of the biomolecule words of each input string.
        """        
        encodings = self.tokenizer.encode_batch(sequences)
        if isinstance(padding_len, int):
            for encoding in encodings:
                encoding.pad(
                    padding_len, direction="right", pad_id=0, pad_token="[PAD]"
                )
                encoding.truncate(padding_len)

        return [encoding.ids for encoding in encodings]

    def save(self, savepath: str):
        """Saves a `WordIdentifier` instance to disk.

        Parameters
        ----------
        savepath : str
            The path to dump the instance. File extension is added automatically.
        """        
        if not savepath.endswith(FILE_EXTENSION):
            savepath = savepath + FILE_EXTENSION
        save_json(json.loads(self.tokenizer.to_str()), savepath)


def load_chemical_word_identifier(vocab_size: int) -> type[WordIdentifier]:
    """A convenience function to load word vocabularies learned for SMILES strings in the study.
    The possible vocabularies to load are for DeepDTA and BPE-DTA. 

    Parameters
    ----------
    vocab_size : int
        Size of the learned SMILES word vocabulary. The allowed values are 94 and 8000, for DeepDTA and BPE-DTA, respectively.

    Returns
    -------
    type[WordIdentifier]
        The `WordIdentifier` instance used by the DTA models.

    Raises
    ------
    ValueError
        If vocabulary size besides 94 and 8000 is passed, a `ValueError` is raised.
    """    
    if vocab_size not in [94, 8000]:
        raise ValueError("Supported vocab sizes are 94 and 8000")

    protein_vocab_path = f"{package_path}/data/word_identification/chemical"
    vocab_path = f"{protein_vocab_path}/chembl27_enc_94.json"
    if vocab_size == 8000:
        vocab_path = f"{protein_vocab_path}/chembl27_enc_bpe_8000.json"

    return WordIdentifier.from_file(vocab_path)


def load_protein_word_identifier(vocab_size: int)-> type[WordIdentifier]:
    """A convenience function to load word vocabularies learned for amino-acid sequences in the study.
    The possible vocabularies to load are for DeepDTA and BPE-DTA. 

    Parameters
    ----------
    vocab_size : int
        Size of the learned SMILES word vocabulary. The allowed values are 26 and 32000, for DeepDTA and BPE-DTA, respectively.

    Returns
    -------
    type[WordIdentifier]
        The `WordIdentifier` instance used by the DTA models.

    Raises
    ------
    ValueError
        If vocabulary size besides 26 and 32000 is passed, a `ValueError` is raised.
    """    
    if vocab_size not in [26, 32000]:
        raise ValueError("Supported vocab sizes are 26 and 32000")

    protein_vocab_path = f"{package_path}/data/word_identification/protein"
    vocab_path = f"{protein_vocab_path}/uniprot_26.json"
    if vocab_size == 32000:
        vocab_path = f"{protein_vocab_path}/uniprot_bpe_32000.json"

    return WordIdentifier.from_file(vocab_path)


if __name__ == "__main__":
    word_identifier = WordIdentifier(vocab_size=1024)
    word_identifier.train(f"{package_path}/data/sequence/chembl27.mini.smiles")

    with open(f"{package_path}/data/sequence/chembl27.mini.smiles") as f:
        sequences = f.readlines()

    tok_seqs = word_identifier.tokenize_sequences(sequences)

    merged = ["".join(seq) for seq in tok_seqs]
    stripped = [seq.strip() for seq in sequences]
    if merged != stripped:
        raise ValueError("Tokenization broke the sequences")

    word_identifier.save(
        f"{package_path}/data/word_identification/chemical_mini_test.json"
    )
    loaded_identifier = WordIdentifier.from_file(
        f"{package_path}/data/word_identification/chemical_mini_test.json"
    )

    is_save_load_ok = (
        word_identifier.tokenizer.to_str()
        == loaded_identifier.tokenizer.to_str()
        and word_identifier.vocab_size == loaded_identifier.vocab_size
    )
    if not is_save_load_ok:
        raise ValueError("Saved and laoded objects are not the same")
