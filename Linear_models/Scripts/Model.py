#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 20:30:37 2026

@author: michael
"""

#%% imports
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
from matplotlib_venn import venn3
from matplotlib_venn import venn2

#%% load data
#load data
train_data = pd.read_csv(snakemake.input.train_data)
test_data = pd.read_csv(snakemake.input.test_data)

#%%
print("loading and preparing data")
#train_data=pd.read_csv("/Users/michael/Desktop/Organage_prequal/Linear_models/Data/Train_data.csv")
#test_data=pd.read_csv("/Users/michael/Desktop/Organage_prequal/Linear_models/Data/Test_data.csv")

#%%

#handle index
train_data.index = train_data["index"]
train_data = train_data.drop(columns="index")
test_data.index = test_data["index"]
test_data = test_data.drop(columns="index")

#prepare x and y for training
X_train = train_data.drop(columns="Age")
Y_train = train_data["Age"]
X_test = test_data.drop(columns="Age")
Y_test = test_data["Age"]

#check dtypes
np.sum(X_train.dtypes != "float64")

#z-scoring
for i in X_train.columns:
    mean, std = np.mean(X_train[i]), np.std(X_train[i])
    X_train[i] = (X_train[i]-mean)/std

for i in X_test.columns:
    mean, std = np.mean(X_test[i]), np.std(X_test[i])
    X_test[i] = (X_test[i]-mean)/std

np.mean(X_train.iloc[:,0])
np.std(X_train.iloc[:,0])

#fill na 
np.sum(X_train.isna())
X_train.fillna(0, inplace=True)
X_test.fillna(0, inplace=True)
np.sum(X_train.isna())

#%% lin regression
#parameter optimisation
param = [{'alpha': np.logspace(0,2,20)}]
lasso_regressor = Lasso()

optimizer = GridSearchCV(lasso_regressor, param, cv=3)
optimized = optimizer.fit(X_train, Y_train)

best_score = optimized.best_score_
best_alpha = optimized.best_estimator_
best_parameters = optimized.best_params_

#model training
model = Lasso(alpha=1)
model.fit(X_train, Y_train)

model_coefs = model.coef_

#%%
def get_top_k_coefs(coefs, k):
    idx = np.argsort(abs(coefs))[::-1][:k]
    top_coeffs = coefs[idx]
    top_genes = X_train.columns[idx]
    df = pd.DataFrame({"gene": top_genes, "coef": top_coeffs})
    return df

top_50_coefs = get_top_k_coefs(model_coefs, 50)

#prediction and evaluaton
preds = model.predict(X_test) #score is r2
test_data["Age_predicted"] = preds
test_data["Error"] = test_data["Age"]- test_data["Age_predicted"]
rmse = np.sqrt(np.sum(test_data["Error"]**2)/test_data.shape[0])

#%%logistic regression
#parameter optimisation
Y_train_log = Y_train.astype("category")
param_log = [{'C': np.logspace(-3,1,20)}]
model_logistic = LogisticRegression(penalty="l1")

optimizer_log = GridSearchCV(model_logistic, param_log, cv=3)
optimized_log = optimizer.fit(X_train, Y_train_log)

optimized_log.best_params_
optimized_log.best_estimator_
optimized_log.cv_results_

#model training
model_logistic= LogisticRegression(C=1)
model_logistic.fit(X_train, Y_train_log)
model_coefs_log = model_logistic.coef_

#%%
#parameter analysis
train_data["Age"].value_counts()

top_50_c1 = set(get_top_k_coefs(model_coefs_log[0],100)["gene"])
top_50_c2 = set(get_top_k_coefs(model_coefs_log[1],100)["gene"])
top_50_c3 = set(get_top_k_coefs(model_coefs_log[2],100)["gene"])

top_100_3m = get_top_k_coefs(model_coefs_log[0],50)
top_100_3m.to_csv(snakemake.output.top_50_genes_logistic_3m)

top_100_18m = get_top_k_coefs(model_coefs_log[0],100)
top_100_3m.to_csv(snakemake.output.top_50_genes_logistic_18m)

top_100_24m = get_top_k_coefs(model_coefs_log[0],100)
top_100_3m.to_csv(snakemake.output.top_50_genes_logistic_24m)

len(set.intersection(top_50_c1, top_50_c2, top_50_c3))

plt.figure()
venn3((top_50_c1, top_50_c2, top_50_c3), set_labels=["3m", "18m", "24m"])
plt.title("Age Biomarkers logistic regression")
plt.savefig(snakemake.output.venn_logistic)

plt.close()
top_50_coefs_log = get_top_k_coefs(model_coefs_log.sum(axis=0), 50)

a=model_coefs_log[0].sort()

preds_log = model_logistic.predict(X_test) 
test_data["Age_predicted_log"] = preds_log
test_data["Discovery_truth"] = (test_data["Age"].astype("category") == test_data["Age_predicted_log"])
FDR = 1-np.sum(test_data["Discovery_truth"])/test_data.shape[0]

Model_performance = pd.DataFrame({"lin_regr": [rmse],
                                  "logistic_regr": [FDR]})
Model_performance.to_csv(snakemake.output.Model_performance)

print("pipeline done")

# %%
