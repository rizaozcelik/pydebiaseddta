"""
The module that contains the DebiasedDTA training framework.

DebiasedDTA training framework comprises two-stages, which we call the “guide” and “predictor” models.
The guide learns a weighting of the training dataset such that a model trained thereupon can learn a robust relationship 
between biomolecules and binding affinity, instead of spurious associations.
The predictor then uses the weights produced by the guide to progressively weight the
training data during its training, in order to obtain a predictor that can generalize well to unseen molecules.


DebiasedDTA leverages the guides to identify protein-ligand pairs that bear more information about the
mechanisms of protein-ligand binding. We hypothesize that, if the guides, models designed to learn misleading spurious
patterns, perform poorly on a protein-ligand pair, then the pair is more likely to bear generalizable information on binding
and deserves higher attention by the DTA predictors.

DebiasedDTA adopts k-fold cross-validation (`k= 1 / mini_val_frac`) to measure the performance of a guide on the training interactions. First, it randomly
divide the training set into five folds and construct five different mini-training and mini-validation sets. DebiasedDTA trains the guide
on each mini-training set and compute the squared errors on the corresponding mini-validation set. 
One run of cross-validation yields one squared-error measurement per protein-ligand pair as each pair is placed in the mini-validation set
exactly once. In order to better estimate the performance on each sample, DebiasedDTA runs the cross-validation `n_bootstrapping`  times and
obtains `n_bootstrapping` error measurements per sample. DebiasedDTA computes the median of the `n_bootstrapping` squared errors
and calls it the "importance coefficient" of a protein-ligand pair. The importance coefficients guide the training of the predictor after being converted
into training weights.

In the DebiasedDTA training framework, the predictor is the model that will be trained with the training samples weighted
by the guide to ultimately predict target protein-chemical affinities. The predictor can adopt any biomolecule representation,
but has to be able to weight the training samples during training to comply with the weight adaptation strategy proposed in DebiasedDTA.

The proposed strategy initializes the training sample weights to 1 and updates them at each epoch such that the
weight of each training sample converges to its importance coefficient at the last epoch. When trained with this strategy,
the predictor attributes higher importance to samples with more information on binding rules (*i.e.* samples with higher
importance coefficient) as the learning continues. Our weight adaptation strategy is formulated as

$$\\vec{w}_e = (1 - \\frac{e}{E}) + \\vec{i} \\times \\frac{e}{E}, $$

where $\\vec{w}_e$ is the vector of training sample weights at epoch $e$, $E$ is the number of training epochs, 
and $\\vec{i}$ is the importance coefficients vector. Here, $e/E$ increases as the training continues,
and so does the impact of $\\vec{i}$, importance coefficients, on the sample weights.
"""
from .debiaseddta import DebiasedDTA
