### pydebiaseddta
-----------------

Repository for `pydebiaseddta`, Python implementation of the DebiasedDTA training framework for robust generalization in drug-target affinity prediction models [1]. The documentation of the library is available [here](https://rizaozcelik.github.io/pydebiaseddta/).

### Installation
----------------
The package can be installed using the \texttt{pip} package management system. Before installing the package, we highly recommend creation of a virtual environment with the appropriate Python version, using the Anaconda software framework:
```bash
conda create --name debiaseddta python=3.9.7
conda activate debiaseddta
python3 -m pip install pydebiaseddta
```
### Quickstart
We highly recommend starting from the Jupyter notebook file `examples/quickstart.ipynb`, which provides a practical yet comprehensive introduction to `pydebiaseddta`.

To interact with this file, open this file with an IDE that can read notebook files, such as VS Code or Jupyter Lab. In this program, you will have to select the interpreter as `debiaseddta`, corresponding to the name of the conda environment you just installed. The cells of the notebook will onboard you to the main functionalities of the repository.

To see examples of a broader range of functionalities, check the notebook file `examples/additional_use_cases.ipynb`. This notebook illustrates uses of various additional experiment settings.

### Additional Data
The repository includes a small sample from the BDB dataset [2], which can be loaded using the function `pydebiaseddta.utils.load_sample_dta_data`. This allows the user to experiment with `pydebiaseddta`'s features without needing external data.

For experiments with real datasets we make the full BDB and KIBA [3] datasets available to download from the following [link](https://drive.google.com/drive/folders/1ihpWgYqugjEKEN9ceyTKCIQxpp1DP_XS?usp=drive_link). The datasets have been separated into splits called `train`, `test_warm`, `test_cold_prot`, `test_cold_lig`, and `test_cold_both`. The various test splits serve to test algorithm performance in previously encountered proteins and ligands, novel proteins, novel ligands, and both novel proteins and ligands. Validation counterparts of the test splits have also been provided for model selection. Five different splits have been provided to allow the user to assess the effect of randomization on results.

### Citation
------------
If you use `pydebiaseddta` in your research, please consider citing:

```bibtex
@article{ozccelik2022debiaseddta,
    title={DebiasedDTA: DebiasedDTA: Improving the Generalizability of Drug-Target Affinity
Prediction Models},
    author={{\"O}z{\c{c}}elik, R{\i}za and Ba{\u{g}}, Alperen and At{\i}l, Berk and Barsbey, Melih and {\"O}zg{\"u}r, Arzucan and {\"O}zk{\i}r{\i}ml{\i}, Elif},
    journal={arXiv preprint arXiv:2107.05556},
    year={2023}
}
```
### References
[1] Özçelik, Rıza, Alperen Bağ, Berk Atıl, Melih Barsbey, Arzucan Özgür, and Elif Özkırımlı. “DebiasedDTA: A Framework for Improving the Generalizability of Drug-Target Affinity Prediction Models.” arXiv, January 8, 2023. http://arxiv.org/abs/2107.05556.
[2] Özçelik, Rıza, Hakime Öztürk, Arzucan Özgür, and Elif Ozkirimli. “ChemBoost: A Chemical Language Based Approach for Protein - Ligand Binding Affinity Prediction.” _Molecular Informatics 40_, no. 5 (May 2021): e2000212. https://doi.org/10.1002/minf.202000212.
[3] Tang, Jing, Agnieszka Szwajda, Sushil Shakyawar, Tao Xu, Petteri Hintsanen, Krister Wennerberg, and Tero Aittokallio. “Making Sense of Large-Scale Kinase Inhibitor Bioactivity Data Sets: A Comparative and Integrative Analysis.” _Journal of Chemical Information and Modeling 54_, no. 3 (March 24, 2014): 735–43. https://doi.org/10.1021/ci400709d.

