#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 12 13:21:47 2026

@author: michael
"""
import copy
import json
import pickle
import graphviz
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb


#for reduced memory usage
def reduce_mem_usage(df):
    for col in df.columns:
        if df[col].dtype == "float64":
            df[col]= df[col].astype(np.float32)
        if df[col].dtype == "int64":
            df[col]=df[col].astype(np.int32)
    return(df)


# load dataset
print("Loading data...")

# 1. Prepare your data
df = pd.read_csv(snakemake.input.train_data)
df.index = df["index"]
df_1 = df.drop(columns="index")
          
reduce_mem_usage(df_1)

#train test split
np.random.seed(42)
indices = df_1.index.to_list()
type(indices)

np.random.shuffle(indices)
indices_train = indices[0:int(len(indices)*0.8)]
indices_test = indices[int(len(indices)*0.8):len(indices)]
len(indices_test)

df_train = df_1.loc[indices_train,:]
df_train.shape
df_train.columns
df_test = df_1.loc[indices_test,:]

y_train = df_train["Age"]
y_test = df_test["Age"]
X_train = df_train.drop("Age", axis=1)
X_test = df_test.drop("Age", axis=1)

# create dataset for lightgbm
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

#loading the parameters
static_params = {
   "objective": "regression",
   "metric": ["l1","rmse"],
   "verbosity": -1,
   "boosting_type": "gbdt",
   "random_state": 42}

with open(snakemake.input.best_params, "r") as f:
    opt_params = json.load(f)

params = {**static_params, **opt_params}

#-
#for recording evaluation results
    #this generates an emty dictionary for the callback function reccording the metrics
    #a callback function runs at the end of each itteration
    #the callback function "record_evaluation" collects evaluation metrics for each itteration
    #the recorded metrics are the ones specified in the params under "metric", more than one are possible such as mae and rmse

evals_result = {} 
#-

# train

    #Note: log_evaluation prints the evaluation metric to the console for live monitoring
print("Starting training...")
gbm = lgb.train(
    params, lgb_train, num_boost_round=200, valid_sets=lgb_eval, 
    callbacks=[lgb.early_stopping(stopping_rounds=5), lgb.log_evaluation(10), lgb.record_evaluation(evals_result)
])

feature_importance = list(gbm.feature_importance())
best_mae = gbm.best_score
best_iter = gbm.best_iteration

# save model to file
print("Saving model...")
gbm.save_model(snakemake.output.model)

#%%plot parameter importances
print("Plotting metrics recorded during training...")
fig,ax = plt.subplots(2,2,figsize= (15,12))

lgb.plot_metric(evals_result, metric="l1",ax=ax[0,0])
lgb.plot_importance(gbm, importance_type="gain", max_num_features=10, ax=ax[0,1])
lgb.plot_split_value_histogram(gbm, feature="Lars2", bins="auto", ax =ax[1,0])
lgb.plot_tree(gbm, tree_index=1, figsize=(15, 15), show_info=["split_gain"], ax=ax[1,1])

ax[0,0].set_title("Error Reduction")
ax[0,0].set_xlabel("Iteration")
ax[0,0].set_ylabel("PRMSE")

ax[0,1].set_title(" Feature importances")
ax[0,1].set_xlabel("total gain")
ax[0,1].set_ylabel("Feature")

ax[1,0].set_title("Split Values")
ax[1,0].set_xlabel("Feature split value Lars2")
ax[1,0].set_ylabel("Clunt")

ax[1,1].set_title("Tree 1")

fig.savefig(snakemake.output.model_metrics)





