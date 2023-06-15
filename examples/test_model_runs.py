# %%
from pydebiaseddta.guides import BoWDTA, IDDTA
from pydebiaseddta.debiasing import DebiasedDTA
from pydebiaseddta.predictors import DeepDTA, BPEDTA, LMDTA
from pydebiaseddta.utils import load_sample_dta_data
from pydebiaseddta.evaluation import evaluate_predictions

# %%
train_ligands, train_proteins, train_labels = load_sample_dta_data(mini=True)["train"]
# %%
bpedta = BPEDTA(n_epochs=2)
bpedta.train(train_ligands, train_proteins, train_labels)
# %%
deepdta = DeepDTA(n_epochs=2)
deepdta.train(train_ligands, train_proteins, train_labels)
# %%
lmdta = LMDTA(n_epochs=2)
lmdta.train(train_ligands, train_proteins, train_labels)
# %%
debiaseddta_bow = DebiasedDTA(BoWDTA, BPEDTA, predictor_params={"n_epochs": 2})
debiaseddta_bow.train(train_ligands, train_proteins, train_labels)
# %%
debiaseddta_id = DebiasedDTA(IDDTA, DeepDTA, predictor_params={"n_epochs": 2})
debiaseddta_id.train(train_ligands, train_proteins, train_labels)
# %%
preds = lmdta.predict(train_ligands, train_proteins)
scores = evaluate_predictions(train_labels, preds, metrics=["ci", "mse", "r2"])
# %%
