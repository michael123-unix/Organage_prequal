#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 11 16:54:56 2026

@author: michael
"""

import copy
import json
import pickle
import graphviz
import multiprocessing as mp
import seaborn as sn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as st
from matplotlib_venn import venn3
import lightgbm as lgb
import shap
import os
import psutil

#%%Function for memory usage tracking
def get_memory_usage():
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / (1024 ** 2)
    return f"Python is currently using: {mem_mb:.2f} MB"

#for reduced memory usage
def reduce_mem_usage(df):
    for col in df.columns:
        if df[col].dtype == "float64":
            df[col]= df[col].astype(np.float32)
        if df[col].dtype == "int64":
            df[col]=df[col].astype(np.int32)
    return(df)
            
#%%load data
# 1. Prepare your data
print("preparing data...")
train_data = pd.read_csv(snakemake.input.train_data)
test_data = pd.read_csv(snakemake.input.test_data)

train_data.index = train_data["index"]
train_data= train_data.drop(columns="index")
train_data.columns

test_data.index = test_data["index"]
test_data = test_data.drop(columns="index")
test_data.columns

reduce_mem_usage(train_data)

#prepare the train data for generating attributions and interactions
Y_train = train_data.pop("Age")
X_data_train = train_data
X_data_train.columns
Full_train_data = X_data_train.assign(Age = Y_train).copy()

#generate test data for prediction and error assessment
Y_test = test_data.pop("Age")
X_data_test = test_data
X_data_test.columns

#%%Prediction on test data

#load model
gbm= lgb.Booster(model_file=snakemake.input.model)

#Predict
print("start predicting")     
preds = gbm.predict(X_data_test)
Full_test_data = X_data_test.assign(Age = Y_test, Age_predicted = preds).copy()
Full_test_data["Error"] = Full_test_data["Age"] - Full_test_data["Age_predicted"]
RMSE= round(np.sqrt(sum(Full_test_data["Error"]**2)/len(Full_test_data["Error"])),2)

pred_3m = Full_test_data[Full_test_data["Age"]==3]["Age_predicted"]
pred_18m = Full_test_data[Full_test_data["Age"]==18]["Age_predicted"]
pred_24m = Full_test_data[Full_test_data["Age"]==24]["Age_predicted"]

plt.figure()
Full_test_data.boxplot("Age_predicted", by="Age")
plt.suptitle("")
plt.title("Age vs Age predicted")
plt.xlabel("Age predicted")
plt.ylabel("Age")
plt.grid(visible=False)
plt.text(2.5,1,f"RMSE = {RMSE}")
plt.savefig(snakemake.output.Age_vs_Age_predicted)
plt.close()

#%%shap values and overview importance plots
print("calculating SHAP values")
explainer = shap.TreeExplainer(gbm)

shap_values = explainer(X_data_train)
type(shap_values)
X_data_train.columns
shap_values.feature_names

#for explanations on each age group, subset the data and calculate averages of SHAP values for all features
print("subset shap matrix")

    #generate masks for subsetting the Explainer object
mask_3= (Full_train_data["Age"] == 3).values
mask_18= (Full_train_data["Age"] == 18).values
mask_24= (Full_train_data["Age"] == 24).values
type(mask_3)

len(Full_train_data["Age"]) == mask_3.sum() + mask_18.sum() + mask_24.sum()

    #generate age slices
shap_3m = shap_values[mask_3]
shap_18m = shap_values[mask_18]
shap_24m = shap_values[mask_24]
type(shap_3m)

shap_3m.shape[0] + shap_18m.shape[0] + shap_24m.shape[0] == shap_values.shape[0]

    #generate mean shap values Note: to reconstruc a shap explanation object for plotting,
    #the mean for all subarrays have to be calculated: base, data and shap value
    #value means
mean_shap_3m_value = shap_3m.values.mean(axis=0)
mean_shap_18m_value = shap_18m.values.mean(axis=0)
mean_shap_24m_value = shap_24m.values.mean(axis=0)

    #base (background) means
mean_shap_3m_base = shap_3m.base_values.mean(axis=0)
mean_shap_18m_base = shap_18m.base_values.mean(axis=0)
mean_shap_24m_base = shap_24m.base_values.mean(axis=0)

    #data means
mean_shap_3m_data = shap_3m.data.mean(axis=0)
mean_shap_18m_data = shap_18m.data.mean(axis=0)
mean_shap_24m_data = shap_24m.data.mean(axis=0)

    #generating new explanation objects
mean_shap_3m = shap.Explanation(values=mean_shap_3m_value, base_values=mean_shap_3m_base, data= mean_shap_3m_data, 
                                feature_names=shap_3m.feature_names)
mean_shap_18m = shap.Explanation(values=mean_shap_18m_value, base_values=mean_shap_18m_base, data= mean_shap_18m_data, 
                                feature_names=shap_18m.feature_names)
mean_shap_24m = shap.Explanation(values=mean_shap_24m_value, base_values=mean_shap_24m_base, data= mean_shap_24m_data, 
                                feature_names=shap_24m.feature_names)
type(mean_shap_3m)
mean_shap_3m.shape
mean_shap_18m.shape
mean_shap_24m.shape
mean_shap_3m.feature_names[20000]

#waterfall plot of average values
print("generating waterfall plots...") 
waterfall_3m, ax= plt.subplots()
ax=shap.plots.waterfall(mean_shap_3m, show=False)
waterfall_3m.savefig(snakemake.output.waterfall_3m)
plt.close()

waterfall_18m, ax= plt.subplots()
ax=shap.plots.waterfall(mean_shap_18m, show=False)
waterfall_18m.savefig(snakemake.output.waterfall_18m)
plt.close()

waterfall_24m, ax= plt.subplots()
ax=shap.plots.waterfall(mean_shap_24m, show=False)
waterfall_18m.savefig(snakemake.output.waterfall_24m)
plt.close()

#beeswarm plots for shap value to value interactions:
print("generating beeswarm plots...")
beeswarm_3m, ax = plt.subplots()  
ax=shap.plots.beeswarm(shap_3m, show=False)
beeswarm_3m.savefig(snakemake.output.beeswarm_3m)
plt.close()

beeswarm_18m, ax = plt.subplots()  
ax=shap.plots.beeswarm(shap_18m, show=False)
beeswarm_18m.savefig(snakemake.output.beeswarm_18m)
plt.close()

beeswarm_24m, ax = plt.subplots()  
ax=shap.plots.beeswarm(shap_24m, show=False)
beeswarm_24m.savefig(snakemake.output.beeswarm_24m)
plt.close()


#%%Top 50 features across all sets
print("extracting top 50 features...")
#extract 50 most important features globally and for age groups
#global
idx_shap_top_50 = np.argsort(np.abs(shap_values.values.mean(axis=0)))[::-1][:50] #::-1 reverses the array

top_50_genes = X_data_train.columns[idx_shap_top_50]
top_50_values = shap_values.values.mean(axis=0)[idx_shap_top_50]

top_genes = pd.DataFrame({"Top_50_Genes": top_50_genes,                         
                          "Values": top_50_values})
top_genes.to_csv(snakemake.output.top_50_genes_global)

#top genes for age groups
idx_shap_top_50_3m = np.argsort(np.abs(mean_shap_3m.values))[::-1][:50] #::-1 reverses the array
idx_shap_top_50_18m = np.argsort(np.abs(mean_shap_18m.values))[::-1][:50] #::-1 reverses the array
idx_shap_top_50_24m = np.argsort(np.abs(mean_shap_24m.values))[::-1][:50] #::-1 reverses the array

#3m
top_50_genes_3m = X_data_train.columns[idx_shap_top_50_3m]
top_50_values_3m = shap_values.values.mean(axis=0)[idx_shap_top_50_3m]

top_genes_3m = pd.DataFrame({"Top_50_Genes": top_50_genes_3m,
                             "Values": top_50_values_3m})
top_genes_3m.to_csv(snakemake.output.top_50_genes_3m)

#18m
top_50_genes_18m = X_data_train.columns[idx_shap_top_50_18m]
top_50_values_18m = shap_values.values.mean(axis=0)[idx_shap_top_50_18m]

top_genes_18m = pd.DataFrame({"Top_50_Genes": top_50_genes_18m,
                          "Values": top_50_values_18m})
top_genes_18m.to_csv(snakemake.output.top_50_genes_18m)

#24m
top_50_genes_24m = X_data_train.columns[idx_shap_top_50_24m]
top_50_values_24m = shap_values.values.mean(axis=0)[idx_shap_top_50_24m]

top_genes_24m = pd.DataFrame({"Top_50_Genes": top_50_genes_24m,
                          "Values": top_50_values_24m})
top_genes_24m.to_csv(snakemake.output.top_50_genes_24m)

#venn diagrams of most important genes 
print("generating venn diagram of top 50 genes across age groups")
plt.figure()
venn3((set(top_50_genes_3m), set(top_50_genes_18m), set(top_50_genes_24m)), ("3m", "18m", "24m"))
#plt.savefig("/Users/michael/Python/Organage/Results/venn_top_50_genes.png")
plt.savefig(snakemake.output.venn_top_genes)
plt.close()

#%%gene gene interactions

#generate df subsets based on age 
Full_3m = Full_train_data[Full_train_data["Age"]==3]
Full_18m = Full_train_data[Full_train_data["Age"]==18]
Full_24m = Full_train_data[Full_train_data["Age"]==24]

X_3m = Full_3m.drop(columns=["Age"])
X_18m = Full_18m.drop(columns=["Age"])
X_24m = Full_24m.drop(columns=["Age"])

#interaction plots

#global dependence plots of 20 most important genes
print("generating dependence plots for top 20 genes")

for i,path in enumerate(snakemake.output.top_genes_scatter):
    shap.plots.scatter(shap_values[:, i], color=shap_values, show=False)
    plt.savefig(path)
    plt.close()

#for age groups: calculate top three interaction partners for top 50 genes for each age group
    #Note: approximation with shap.utils.approximate_interactions, gives index of most likely interaction partner

print("estimating and extracting top gene interactions")
Interactions_3m = pd.DataFrame()
for i,j in zip(top_50_genes_3m, range(len(top_50_genes_3m))):
    sorted_idx = shap.utils.approximate_interactions(i, shap_3m.values, X_3m)
    top_idx = sorted_idx[0:3]
    interactors = X_3m.columns[top_idx]
    Interactions_3m.insert(loc=j, column=i, value=interactors)
Interactions_3m.to_csv(snakemake.output.top_50_interactions_3m)

Interactions_18m = pd.DataFrame()
for i,j in zip(top_50_genes_18m, range(len(top_50_genes_18m))):
    sorted_idx = shap.utils.approximate_interactions(i, shap_18m.values, X_18m)
    top_idx = sorted_idx[0:3]
    interactors = X_18m.columns[top_idx]
    Interactions_18m.insert(loc=j, column=i, value=interactors)
Interactions_18m.to_csv(snakemake.output.top_50_interactions_18m)

Interactions_24m = pd.DataFrame()
for i,j in zip(top_50_genes_24m, range(len(top_50_genes_24m))):
    sorted_idx = shap.utils.approximate_interactions(i, shap_24m.values, X_24m)
    top_idx = sorted_idx[0:3]
    interactors = X_24m.columns[top_idx]
    Interactions_24m.insert(loc=j, column=i, value=interactors)
Interactions_24m.to_csv(snakemake.output.top_50_interactions_24m)

print("pipeline finished successfully")

"""
#heatmap (shows only little correllations), so far useless

idx_shap_top_100_3m_hm = np.argsort(np.abs(mean_shap_3m.values))[::-1][:50] #::-1 reverses the array
idx_shap_top_100_18m_hm = np.argsort(np.abs(mean_shap_18m.values))[::-1][:50] #::-1 reverses the array
idx_shap_top_100_24m_hm = np.argsort(np.abs(mean_shap_24m.values))[::-1][:50] #::-1 reverses the array

X_3m_small_hm = X_3m.iloc[:,idx_shap_top_100_3m_hm].sample(axis=0, frac=0.1)
X_3m_small.shape
X_18m_small_hm = X_18m.iloc[:,idx_shap_top_100_18m_hm].sample(axis=0, frac=0.16)
X_18m_small.shape
X_24m_small_hm = X_24m.iloc[:,idx_shap_top_100_24m_hm].sample(axis=0, frac=0.085)
X_24m_small.shape

corr_matrix_3m_hm = np.array(X_3m_small_hm.corr())

corr_df_3m_hm = pd.DataFrame((corr_matrix_3m_hm))
corr_df_3m_hm.columns = X_3m_small_hm.columns
corr_df_3m_hm.index = X_3m_small_hm.columns
corr_df_3m_hm.shape

sn.heatmap(corr_df_3m_hm)

"""














