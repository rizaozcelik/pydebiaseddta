"""
The submodule that contains the predictors, *i.e.*, drug-target affinity (DTA) prediction models, implemented in DebiasedDTA study.
The implemented predictors are BPEDTA, DeepDTA, and LMDTA. 
Abstract classes are also available to quickly train a custom DTA prediction model with DebiasedDTA.
"""
from .bpedta import BPEDTA
from .deepdta import DeepDTA
from .lmdta import LMDTA
