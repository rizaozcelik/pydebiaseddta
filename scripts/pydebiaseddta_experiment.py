from pydebiaseddta.debiasing import DebiasedDTA
from pydebiaseddta.guides import BoWDTA, IDDTA, RFDTA, OutDTA
from pydebiaseddta.predictors import DeepDTA, BPEDTA, LMDTA
from pydebiaseddta.evaluation import evaluate_predictions
from pydebiaseddta.utils import save_json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import argparse
import importlib.util
import pandas as pd
from pathlib import Path
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", help="Path to configuration file")
args = parser.parse_args()

spec = importlib.util.spec_from_file_location("config", args.config_path)
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

if config.results_save_folder:
    if config.results_save_folder[-1] == "/":
        config.results_save_folder[:-1]
    weights_save_path = config.results_save_folder + "/importance_weights.coef"
    Path(os.path.dirname(weights_save_path)).mkdir(parents=True, exist_ok=True)
else:
    raise Exception("This script requires a results folder to be provided.")


config_dict = {k: v for k, v in config.__dict__.items() if not k.startswith('__')}

# Save this dictionary into a JSON file
save_json(config_dict, config.results_save_folder + '/config.json')

models = { 
    "bowdta": BoWDTA, "iddta": IDDTA, "rfdta": RFDTA, "outdta": OutDTA,
    "deepdta": DeepDTA, "bpedta": BPEDTA, "lmdta": LMDTA, "none": None,
    }

print("Loading data.")
train_data = pd.read_csv(config.train_data_path)
val_splits = {
    split: [pd.read_csv(data_path)[key].tolist() for key in ["smiles", "aa_sequence", "affinity_score"]] for split, data_path in config.val_data_paths.items()
    }
test_splits = {
    split: [pd.read_csv(data_path)[key].tolist() for key in ["smiles", "aa_sequence", "affinity_score"]] for split, data_path in config.test_data_paths.items()
    }
print("Data loaded. Initializing DebiasedDTA model.")
debiaseddta = DebiasedDTA(models[config.guide.lower()], models[config.predictor.lower()], **config.debiaseddta_params)
print("Initialized DebiasedDTA model, starting training.")
train_hist = debiaseddta.train(train_data["smiles"].tolist(),
                               train_data["aa_sequence"].tolist(),
                               train_data["affinity_score"].tolist(),
                               val_splits=val_splits,
                               metrics_tracked=config.metrics_tracked,
                               weights_save_path=weights_save_path,
                               predictor_save_folder=config.results_save_folder,
                               weights_load_path=config.weights_load_path,
                               predictor_load_folder=config.predictor_load_folder)
save_json(train_hist, config.results_save_folder + "/train_hist.json")

print("Starting evaluation.")
results = {}
for split in test_splits.keys():
    test_ligands, test_proteins, test_labels = test_splits[split]
    test_preds = debiaseddta.predictor_instance.predict(test_ligands, test_proteins)
    results[split] = evaluate_predictions(test_labels, test_preds, metrics=["ci", "mse", "r2", "mae", "rmse"])    
save_json(train_hist, config.results_save_folder + "/results.json")
print("Completed evaluation, saving results.")