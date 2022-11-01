"""
The submodule that contains the guides, *i.e.*, 
the weak learners in DebiasedDTA that learn a weighting of the training set to improve generalizability.
The implemented guides are IDDTA and BoWDTA, and an abstract classes is also available to quickly implement custom guides.
"""
from .abstract_guide import Guide
from .bowdta import BoWDTA
from .iddta import IDDTA