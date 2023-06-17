import re
from collections import Counter
from typing import List, Union, Dict

from .. import package_path
from ..utils import load_json


OOV_TOKEN = "X"
_ELEMENTS_STR = r"(?<=\[)Cs(?=\])|Si|Xe|Ba|Rb|Ra|Sr|Dy|Li|Kr|Bi|Mn|He|Am|Pu|Cm|Pm|Ne|Th|Ni|Pr|Fe|Lu|Pa|Fm|Tm|Tb|Er|Be|Al|Gd|Eu|te|As|Pt|Lr|Sm|Ca|La|Ti|Te|Ac|Cf|Rf|Na|Cu|Au|Nd|Ag|Se|se|Zn|Mg|Br|Cl|Pb|U|V|K|C|B|H|N|O|S|P|F|I|b|c|n|o|s|p"
__REGEXES = {
    "segmentation": rf"(\[[^\]]+]|{_ELEMENTS_STR}|"
    + r"\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%\d{2}|\d)",
    "segmentation_sq": rf"(\[|\]|{_ELEMENTS_STR}|"
    + r"\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%\d{2}|\d)",
}
_RE_PATTERNS = {name: re.compile(pattern) for name, pattern in __REGEXES.items()}


def segment_smiles(smiles: str, segment_sq_brackets: bool = True) -> List[str]:
    """Segments a SMILES string into its tokens.

    Parameters
    ----------
    smiles : str
        Input SMILES string.
    segment_sq_brackets : bool, optional
        Whether to segment expressions within square brackets (*e.g.* [C@@H], [Rb]), too. 
        Set to `True` to have square brackets and the tokens inside as standalone tokens,
        *e.g.* ["[", "C", "@", "@", "H", "]"]. 
        When set to `False`, whole expression is returned as a single token, *e.g.* "[C@@H]" .
        Defaults to `True`.

    Returns
    -------
    List[str]
        Each element of the SMILES string as a list.
    """
    regex = _RE_PATTERNS["segmentation_sq"]
    if not segment_sq_brackets:
        regex = _RE_PATTERNS["segmentation"]
    return regex.findall(smiles)


def segment_smiles_batch(
    smiles_batch: List[str], segment_sq_brackets=True
) -> List[List[str]]:
    """Segments multiple SMILES strings with a single call by wrapping `sequence.smiles_processing.segment_smiles`.

    Parameters
    ----------
    smiles_batch : List[str]
        List of input SMILES strings.
    segment_sq_brackets : bool, optional
        Whether to segment expressions within square brackets. 
        See `sequence.smiles_processing.segment_smiles` for a more detailed explanation.
        Defaults to `True`.

    Returns
    -------
    List[List[str]]
        A 2D list of strings where element $[i][j]$ corresponds to the $j^{th}$ token of the $i^{th}$ input.
    """
    return [segment_smiles(smiles, segment_sq_brackets) for smiles in smiles_batch]


def learn_unichar_encoding(smiles_corpus: List[str]) -> Dict:
    unichar_start_ix, unichar_end_ix = 33, 126
    target_vocab = {chr(ix) for ix in range(unichar_start_ix, unichar_end_ix + 1)}
    vocabulary_size = len(target_vocab)
    segmented_corpus = segment_smiles_batch(smiles_corpus)

    token_counts_by_smi = [Counter(example) for example in segmented_corpus]
    token_counts = sum(token_counts_by_smi, Counter())

    if len(token_counts) > vocabulary_size:
        source_vocab = [token for token, count in token_counts.most_common(vocabulary_size)]
    else:
        source_vocab = list(token_counts.keys())
    source_to_target_mapping = {
        token: token for token in source_vocab if token in target_vocab
    }
    source_to_target_mapping["[OOV]"] = OOV_TOKEN
    target_vocab.remove(OOV_TOKEN)  # Not use it for encoding any element
    remainings = target_vocab - set(source_vocab)
    for token in source_vocab:
        if token not in source_to_target_mapping:
            source_to_target_mapping[token] = remainings.pop()
    return source_to_target_mapping


def load_smiles_to_unichar_encoding() -> Dict[str, str]:
    return load_json(
        f"{package_path}/data/word_identification/ligand/chembl27_encoding.json"
    )


def load_unichar_to_smiles_encoding() -> Dict[str, str]:
    smi_to_unichar_encoding = load_smiles_to_unichar_encoding()
    return {v: k for k, v in smi_to_unichar_encoding.items()}


def smiles_to_unichar(smiles: str, encoding=Union[str, Dict]) -> List[str]:
    if encoding is isinstance(encoding, str):
        encoding = load_json(encoding)
    segments = segment_smiles(smiles)
    return "".join([encoding.get(segment, encoding["[OOV]"]) for segment in segments])


def smiles_to_unichar_batch(
    smiles_batch: List[str], encoding=Union[str, Dict]
) -> List[List[str]]:

    if encoding is isinstance(encoding, str):
        encoding = load_json(encoding)
    return [smiles_to_unichar(smiles, encoding) for smiles in smiles_batch]


def unichar_to_smiles(unichar: str, encoding=Union[str, Dict]) -> List[str]:
    if encoding is isinstance(encoding, str):
        encoding = load_json(encoding)
    decoding = {v: k for k, v in encoding.items()}

    return "".join([decoding.get(char, char) for char in unichar])


def unichar_to_smiles_batch(
    unichars: List[str], encoding=Union[str, Dict]
) -> List[List[str]]:

    if encoding is isinstance(encoding, str):
        encoding = load_json(encoding)
    decoding = {v: k for k, v in encoding.items()}

    return [
        "".join([decoding.get(char, char) for char in unichar]) for unichar in unichars
    ]
