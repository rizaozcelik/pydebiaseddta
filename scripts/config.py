from pydebiaseddta import _PACKAGE_PATH as pydebiaseddta_package_path

results_save_folder = "./temp/example_run" # If None, no output will be provided.
# Path to the training .csv file. See the sample files for expected format.
train_data_path = pydebiaseddta_package_path + "/data/dta/dta_sample_data_mini/train.csv"
# Dictionary of validation .csv files.
val_data_paths = {"val": pydebiaseddta_package_path  + "/data/dta/dta_sample_data_mini/val.csv"} 
# Dictionary of test .csv files.
test_data_paths = { 
    "test_1": pydebiaseddta_package_path  + "/data/dta/dta_sample_data/test.csv",
    "test_2": pydebiaseddta_package_path  + "/data/dta/dta_sample_data_mini/test.csv",
    } 

guide = "BoWDTA" # 
predictor = "DeepDTA" # 
debiaseddta_params = {
    "guide_params": {   
        "max_depth": 4,
        "min_samples_split": 5,
    },
    "predictor_params": {
        "early_stopping_metric": "mse",
        "early_stopping_metric_threshold": 1.6,
        "early_stopping_split": "train",
    },
    "seed": 0
}

# Training parameters
metrics_tracked = ["mae", "mse", "r2"]

# If previously existing weights and/or predictors are to be loaded.
weights_load_path = None
predictor_load_folder = None