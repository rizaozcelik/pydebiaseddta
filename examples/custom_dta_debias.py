from pydebiaseddta.predictors import IDDTA, DebiasedDTA
from pydebiaseddta.utils import load_sample_dta_data

class CustomDTAModel:
    def __init__(self, n_epochs):
        self.n_epochs = n_epochs

    def train(self, train_chemicals, train_proteins, train_labels, sample_weights_by_epoch):
        pass

train_chemicals, train_proteins, train_labels = load_sample_dta_data(mini=True)['train']
debiaseddta = DebiasedDTA(IDDTA, CustomDTAModel, predictor_params={'n_epochs': 100})
debiaseddta.train(train_chemicals, train_proteins, train_labels)
