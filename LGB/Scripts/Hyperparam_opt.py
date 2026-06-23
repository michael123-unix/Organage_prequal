#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 31 16:16:05 2025

@author: michael
"""

import optuna
import lightgbm as lgb
import numpy as np
import pandas as pd
import json
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from optuna.visualization import plot_param_importances, plot_optimization_history
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
import os
from multiprocessing import Pool
import matplotlib.pyplot as plt
import optuna.visualization.matplotlib as vis

#for reduced memory usage
def reduce_mem_usage(df):
    for col in df.columns:
        if df[col].dtype == "float64":
            df[col]= df[col].astype(np.float32)
        if df[col].dtype == "int64":
            df[col]=df[col].astype(np.int32)
    return(df)


# 1. Prepare your data
print("loading and preparing data...")
df = pd.read_csv(snakemake.input.train_data)

df.index = df["index"]
df_1 = df.drop(columns="index")
df_1.columns
          
reduce_mem_usage(df_1)

#train val split
np.random.seed(42)
indices = df_1.index.to_list()
type(indices)

np.random.shuffle(indices)
indices_train = indices[0:int(len(indices)*0.8)]
indices_test = indices[int(len(indices)*0.8):len(indices)]
len(indices_test)

df_train = df_1.loc[indices_train,:]

df_test = df_1.loc[indices_test,:]

y_train = df_train["Age"]
y_val = df_test["Age"]
X_train = df_train.drop("Age", axis=1)
X_val = df_test.drop("Age", axis=1)


#varriable parameters
varr_params = {
    "learning_rate": [1e-7, 1],
    "num_leaves": [5, 100],
   # "max_depth": [3, 100],
    "min_child_samples": [5, 500],
    "feature_fraction": [0.4, 1.0],
    "bagging_fraction": [0.4, 1.0],
    "bagging_freq":[1, 10],
    "lambda_l1": [1e-6, 10.0],
    "lambda_l2": [1e-6, 10.0], 
   }

def objective(trial):
     # Define the search space for numeric regression
    param = {
        "objective": "regression",
        "metric": "rmse",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "random_state": 42,
        "max_depth": 50,
                 # Hyperparameters to optimize
        "learning_rate": trial.suggest_float("learning_rate", varr_params["learning_rate"][0], varr_params["learning_rate"][1], log=True),
        "num_leaves": trial.suggest_int("num_leaves", varr_params["num_leaves"][0], varr_params["num_leaves"][1]),
        #"max_depth": trial.suggest_int("max_depth", varr_params["max_depth"][0], varr_params["max_depth"][1]),
        "min_child_samples": trial.suggest_int("min_child_samples", varr_params["min_child_samples"][0], varr_params["min_child_samples"][1]),
        "feature_fraction": trial.suggest_float("feature_fraction", varr_params["feature_fraction"][0], varr_params["feature_fraction"][1]),
        "bagging_fraction": trial.suggest_float("bagging_fraction", varr_params["bagging_fraction"][0], varr_params["bagging_fraction"][1]),
        "bagging_freq": trial.suggest_int("bagging_freq", varr_params["bagging_freq"][0], varr_params["bagging_freq"][1]),
        "lambda_l1": trial.suggest_float("lambda_l1", varr_params["lambda_l1"][0], varr_params["lambda_l1"][1], log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", varr_params["lambda_l2"][0], varr_params["lambda_l2"][1], log=True),
        "max_bin": trial.suggest_int("max_bin", 30, 511),
            }
         
     # Add the pruning callback to stop bad trials early
    pruning_callback = optuna.integration.LightGBMPruningCallback(trial, "rmse")
     
     # Create the LightGBM datasets
    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

     # Train the model
    gbm = lgb.train(
             param, 
             dtrain, 
             valid_sets=[dval], 
             callbacks=[lgb.early_stopping(stopping_rounds=5), pruning_callback],
             num_boost_round=400
             )
    
    #define function output to minimize
    preds = gbm.predict(X_val)
    rmse = (np.sqrt(sum((preds - y_val)**2)/len(preds)))
    return rmse         
    #store the number of itterations
    trial.set_user_attr("best_iteration", gbm.best_iteration)

print("starting optimisation")    
study = optuna.create_study(direction="minimize", study_name="LGBM_Regression") 
study.optimize(objective, n_trials=35)

# See which parameters had the most impact on RMSE
print("saving optimisation history")
param_importance_plot = vis.plot_param_importances(study) #note that the matplotlib functionality (vis) was used since plotly is made for html unsing applications

plt.title("Param Importance")
plt.savefig(snakemake.output.param_importance_plot)
plt.close()
    
# See how the RMSE improved over the 50 trials
history_plot = vis.plot_optimization_history(study)
plt.title("Optimisation history")
plt.savefig(snakemake.output.opt_hist_plot)
plt.close()

best = study.best_params
# Save the master tracking file
pd.DataFrame(best, index=(1,1)).to_csv(snakemake.output.opt_hist, index=False)  

with open(snakemake.output.best_params, "w") as f:
    json.dump(best, f, indent=4)

print("Parameters saved successfully!")
  
    
  
 













