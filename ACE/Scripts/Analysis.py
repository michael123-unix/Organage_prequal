#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 21:09:13 2026

@author: michael
"""
#%% imports

from ace.model import ACE
from pytorch_lightning.utilities.seed import seed_everything
import torch
import scipy 
import matplotlib.pyplot as plt
from matplotlib_venn import venn3
import scanpy as sc
import anndata 
import numpy as np
import pandas as pd
import warnings
import sklearn.linear_model as lm
from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings('ignore')
import os
from pytorch_lightning.loggers import TensorBoardLogger
from path_explain import PathExplainerTorch, scatter_plot, summary_plot
from pathlib import Path

np.random.seed(42)
path = Path.cwd()

#%% load data
print("loading data and model...")
#train_data = anndata.read(path.parent.parent/"Data/Train_data.h5ad")
#test_data = anndata.read(path.parent.parent/"Data/Test_data.h5ad")
train_data = anndata.read(snakemake.input.train_data)
test_data = anndata.read(snakemake.input.test_data)

#%%load model
model = ACE.load(snakemake.input.model, adata=train_data)
#model = ACE.load(path.parent/"Model/model", adata=train_data)

#%% evaluate model on test set
pred = model.get_latent_representation(test_data)

#test_loss
pred_w_age = pd.DataFrame(pred, index=test_data.obs_names, columns=["z1", "z2", "z3"])
pred_w_age["Age"] = test_data.obs["age"]

#plot latent variables vs age
plt.figure()
plt.plot(pred_w_age["Age"],pred_w_age["z1"], linewidth = 0.01, label = "z1")
plt.plot(pred_w_age["Age"], pred_w_age["z2"],linewidth = 0.01, label = "z2")
plt.plot(pred_w_age["Age"], pred_w_age["z3"],linewidth = 0.01, label = "z3")
plt.title("Latent Dimensions for Age")
plt.xlabel("Age")
leg = plt.legend()
for line in leg.get_lines():
    line.set_linewidth(2.0)
    
plt.savefig(snakemake.output.embeddings_vs_age)
plt.close()

#%%ridge regression for age estimation
#Note RidgeCV automaticaly chooses the alpha parameter using CV

#prepare data
print("preparing Data for ridge regression")
train_size = int(pred_w_age.shape[0]*0.8)
indices = np.arange(pred_w_age.shape[0])
np.random.seed(42)
np.random.shuffle(indices)
pred_w_age_train = pred_w_age.iloc[indices[:train_size],]
pred_w_age_train.shape
pred_w_age_test = pred_w_age.iloc[indices[train_size:],]
pred_w_age_test.shape
pred_w_age_train.columns

train_ridge_X = np.array(pred_w_age_train.iloc[:,0:3])
train_ridge_X.shape

train_ridge_Y = np.array((pred_w_age_train["Age"]))
train_ridge_Y.shape

np.dtype(train_ridge_X[0,0])
np.dtype(train_ridge_Y[0])

#Ridge
print("starting ridge regression")
alphas= np.logspace(-5,5, 10)
ridge_model = lm.RidgeCV(alphas=alphas, cv=5)
ridge_model.fit(train_ridge_X, train_ridge_Y)           

age_ridge_test = pred_w_age_test.pop("Age")
ridge_preds = ridge_model.predict(pred_w_age_test)

ridge_test = pred_w_age_test
ridge_test["Age"] = age_ridge_test
ridge_test["Age_pred"] = ridge_preds
ridge_test["Error"] = ridge_test["Age"]-ridge_test["Age_pred"]

ridge_rmse = np.sqrt(sum(ridge_test["Error"]**2)/ridge_test.shape[0])

#error for age groups
ridge_test_3m = ridge_test[ridge_test["Age"]==3]
ridge_test_18m = ridge_test[ridge_test["Age"]==18]
ridge_test_24m = ridge_test[ridge_test["Age"]==24]

ridge_rmse_3m = np.sqrt(sum(ridge_test_3m["Error"]**2)/ridge_test_3m.shape[0])
ridge_rmse_18m = np.sqrt(sum(ridge_test_18m["Error"]**2)/ridge_test_18m.shape[0])
ridge_rmse_24m = np.sqrt(sum(ridge_test_24m["Error"]**2)/ridge_test_24m.shape[0])

ridge_metrics = pd.DataFrame({"Metric": ["Koeff z1", "Koeff z2", "Koeff z3", "RMSE global",
                                      "RMSE 3m", "RMSE 18m", "RMSE 24m"],
                              "Value": [ridge_model.coef_[0], ridge_model.coef_[1],ridge_model.coef_[2],ridge_rmse, ridge_rmse_3m, ridge_rmse_18m, ridge_rmse_24m]
                           })

ridge_metrics.to_csv(snakemake.output.ridge_metrics)
#ridge_metrics.to_csv(path.parent/"Explain_Results/ridge_metrics.csv")

print("ridge regression finished successfully")
#latent libsize: the latent library is a learned variable, it estimates the number of transcripts per cell to normalize by it
#since this varies strongly for cells. variable l in paper

# get latent representations for the test data for validation
latent_salient= anndata.AnnData(X=model.get_latent_representation(test_data, representation_kind = "salient"), 
                                obs=test_data.obs)
 
latent_background = anndata.AnnData(X=model.get_latent_representation(test_data, representation_kind= "background"), 
                                    obs=test_data.obs)

#calculate neighbours and umap for both
print("calculating PCA and UMAPS")
sc.pp.neighbors(latent_salient)
sc.tl.umap(latent_salient)
sc.pp.neighbors(latent_background)
sc.tl.umap(latent_background)

#%%feature attributions global (expected gradinets)

#model for explaining the latent space (path explain needs a single target, not a dictionary which is the standard output of 
# the VAE. Additionally, to be able to use tensors (as required by path explain) and not adata objects, the module encoder will
#be used:) 
   
def model_latent_space(x):
    outputs = model.module.inference(x, batch_index=torch.zeros(x.shape[0]))
    return outputs["qz_m"]
                                                              
explainer = PathExplainerTorch(model_latent_space)

#data prep: validate on train data. background samples, drawn from the train data.
#samples come in batches, background for each batch contains k times as many samples

#global analysis
#convert data to tensor
test_as_array = test_data.X.toarray().astype('float32').copy()
test_as_array.shape
x_test_tensor = torch.from_numpy(test_as_array)
x_test_tensor.shape

#generate subsamples of test data for evaluation
#samples
eval_idx = np.arange(test_data.shape[0])
np.random.shuffle(eval_idx)
eval_tensor = x_test_tensor[eval_idx[:5000],]
eval_tensor.requires_grad_(True)
eval_tensor.shape

#baseline as subset of samples
baseline_idx=np.arange(eval_tensor.shape[0])
np.random.shuffle(baseline_idx)
baseline_tensor = eval_tensor[baseline_idx[:500],]
baseline_tensor.requires_grad_(True)

print("Start feature attribution calculation")
batch_size = 5
dims = [1,2,3]
results = {}
for j in dims:
    attributions=[]
    out_indices = torch.full((batch_size,),j,device=eval_tensor.device)

    for i in range(0,eval_tensor.shape[0],batch_size):    
        attribution = explainer.attributions(input_tensor=eval_tensor[i:i+batch_size,], baseline=baseline_tensor, 
                                       num_samples=20, use_expectation=True,output_indices=out_indices)
        attributions.append(attribution.detach().cpu())
        del attribution
        
    results[j] = torch.cat(attributions)

shap_3D = (sum(results[i] for i in dims)/3).detach().cpu().numpy().copy()
eval_tensor_detatched = np.ascontiguousarray(eval_tensor.detach().cpu().numpy().copy())

#%%Extract the top k attributions
#Note: the class below takes the top attributions as a sum of the three attribution matrices.
    # The attribution of each gene is then summed over all samples.
       
class get_top_attributions:
    def __init__(self, data, k, names):
        mean_data = np.mean(abs(data), axis=0)
        top_k_gene_idx = mean_data.argsort()[::-1][:k]
        k_genes = names[top_k_gene_idx]
        k_values = mean_data[top_k_gene_idx]
        self.df = pd.DataFrame({"gene": k_genes, "value": k_values})

top50_attr_global = get_top_attributions(shap_3D, 50, test_data.var_names)
top50_attr_global.df.to_csv(snakemake.output.top_50_attr_global)
#top50_attr_global.df.to_csv(path.parent/"Explain_Results/top_50_attr_global.csv")

#%%Summary plot plots top k features
summary_plot(shap_3D, eval_tensor_detatched, feature_names=test_data.var_names, plot_top_k = 10)
plt.savefig(snakemake.output.global_attributions)
#plt.savefig(path.parent/"Explain_Results/top_10_global_attributions.png")

#%%data prep for attributions and interactions age specific
#subset anndata object
test_3m = test_data[test_data.obs["age"]==3]
test_3m.shape  
test_18m = test_data[test_data.obs["age"]==18]
test_18m.shape  
test_24m = test_data[test_data.obs["age"]==24]
test_24m.shape  

#convert adata to tensor
test_as_array_3m = test_3m.X.toarray().astype('float32').copy()
x_test_tensor_3m = torch.from_numpy(test_as_array_3m)
x_test_tensor_3m.shape  
    
test_as_array_18m = test_18m.X.toarray().astype('float32').copy()
x_test_tensor_18m = torch.from_numpy(test_as_array_18m)
x_test_tensor_18m.shape  

test_as_array_24m = test_24m.X.toarray().astype('float32').copy()
x_test_tensor_24m = torch.from_numpy(test_as_array_24m)
x_test_tensor_24m.shape  
    
#define sample and baseline tensors
eval_idx_3m = np.arange(test_3m.shape[0])
np.random.shuffle(eval_idx_3m)
eval_tensor_3m = x_test_tensor_3m[eval_idx_3m[:2000],]
eval_tensor_3m.requires_grad_(True)
eval_tensor_3m.shape

eval_idx_18m = np.arange(test_18m.shape[0])
np.random.shuffle(eval_idx_18m)
eval_tensor_18m = x_test_tensor_18m[eval_idx_18m[:2000],]
eval_tensor_18m.requires_grad_(True)

eval_idx_24m = np.arange(test_24m.shape[0])
np.random.shuffle(eval_idx_24m)
eval_tensor_24m = x_test_tensor_24m[eval_idx_24m[:2000],]
eval_tensor_24m.requires_grad_(True)

#baseline as subset of samples
baseline_idx_3m=np.arange(eval_tensor_3m.shape[0])
np.random.shuffle(baseline_idx_3m)
baseline_tensor_3m = eval_tensor_3m[baseline_idx_3m[:500],]
baseline_tensor_3m.requires_grad_(True)

baseline_idx_18m=np.arange(eval_tensor_18m.shape[0])
np.random.shuffle(baseline_idx_18m)
baseline_tensor_18m = eval_tensor_18m[baseline_idx_18m[:500],]
baseline_tensor_18m.requires_grad_(True)

baseline_idx_24m=np.arange(eval_tensor_24m.shape[0])
np.random.shuffle(baseline_idx_24m)
baseline_tensor_24m = eval_tensor_24m[baseline_idx_24m[:500],]
baseline_tensor_24m.requires_grad_(True)

#%% calculate attributionstype(shap_3D_3m)
print("caluclate 3m attributions")

dims = [1,2,3]
batch_size = 5
results_3m = {}
for j in dims:
    attributions=[]
    out_indices = torch.full((batch_size,),j,device=eval_tensor_3m.device) # the attribution function insists on a beeing provided with a vector of same lenght as the input. This vector denotes the variable for which the interacitons will be calculated

    for i in range(0,eval_tensor_3m.shape[0],batch_size):    
        attribution = explainer.attributions(input_tensor=eval_tensor_3m[i:i+batch_size,], baseline=baseline_tensor_3m, 
                                       num_samples=20, use_expectation=True,output_indices=out_indices)
        attributions.append(attribution.detach().cpu())
        del attribution
        
    results_3m[j] = torch.cat(attributions)

#summary plot only for the average attributions over all dimensions (else: three plots for each group)
shap_3D_3m = (sum(results_3m[i] for i in dims)/3).detach().cpu().numpy().copy()
eval_tensor_detatched_3m = np.ascontiguousarray(eval_tensor_3m.detach().cpu().numpy().copy())
summary_plot(shap_3D_3m, eval_tensor_detatched_3m, feature_names=test_data.var_names, plot_top_k = 10)
plt.savefig(snakemake.output.attr_3m_plot)
plt.close()

top_50_attr_3m_avg = get_top_attributions(shap_3D_3m, 50, test_3m.var_names)
top_50_attr_3m_avg.df.to_csv(snakemake.output.top_50_attr_3m)

print("caluclate 18m attributions")

batch_size = 5
dims = [1,2,3]
batch_size = 5
results_18m = {}
for j in dims:
    attributions=[]
    out_indices = torch.full((batch_size,),j,device=eval_tensor_18m.device)

    for i in range(0,eval_tensor_18m.shape[0],batch_size):    
        attribution = explainer.attributions(input_tensor=eval_tensor_18m[i:i+batch_size,], baseline=baseline_tensor_18m, 
                                       num_samples=20, use_expectation=True,output_indices=out_indices)
        attributions.append(attribution.detach().cpu())
        del attribution
        
    results_18m[j] = torch.cat(attributions)


#summary plot only for the average attributions over all dimensions 
shap_3D_18m = (sum(results_18m[i] for i in dims)/3).detach().cpu().numpy().copy()
eval_tensor_detatched_18m = np.ascontiguousarray(eval_tensor_18m.detach().cpu().numpy().copy())
summary_plot(shap_3D_18m, eval_tensor_detatched_18m, feature_names=test_data.var_names, plot_top_k = 10)
plt.savefig(snakemake.output.attr_18m_plot)
plt.close()

top_50_attr_18m_avg = get_top_attributions(shap_3D_18m, 50, test_18m.var_names)
top_50_attr_18m_avg.df.to_csv(snakemake.output.top_50_attr_18m)

print("caluclate 24m attributions")

batch_size = 5
dims = [1,2,3]
batch_size = 5
results_24m = {}
for j in dims:
    attributions=[]
    out_indices = torch.full((batch_size,),j,device=eval_tensor_24m.device)

    for i in range(0,eval_tensor_24m.shape[0],batch_size):    
        attribution = explainer.attributions(input_tensor=eval_tensor_24m[i:i+batch_size,], baseline=baseline_tensor_24m, 
                                       num_samples=20, use_expectation=True,output_indices=out_indices)
        attributions.append(attribution.detach().cpu())
        del attribution
        
    results_24m[j] = torch.cat(attributions)


#summary plot only for the average attributions over all dimensions
shap_3D_24m = (sum(results_24m[i] for i in dims)/3).detach().cpu().numpy().copy()
eval_tensor_detatched_24m = np.ascontiguousarray(eval_tensor_24m.detach().cpu().numpy().copy())
summary_plot(shap_3D_24m, eval_tensor_detatched_24m, feature_names=test_data.var_names, plot_top_k = 10)
plt.savefig(snakemake.output.attr_24m_plot)
plt.close()

top_50_attr_24m_avg = get_top_attributions(shap_3D_24m, 50, test_24m.var_names)
top_50_attr_24m_avg.df.to_csv(snakemake.output.top_50_attr_24m)


#%% calculate interactions
#define sample and baseline tensors

"""
Note: one cannot use only 250 genes for a network trained on 22000 genes. use a different approach for interacitons. 
Computational ressources not available for full interaction calculation

#function for selecting the k most important features by attribution
def get_top_attributions_for_interactions(data,k):
    mean_shap = data.mean(axis=0).detach().numpy()
    abs_data = abs(mean_shap)
    idx = abs_data.flatten().argsort()[::-1]
    top_k_idx = idx[0:k]
    row, col = np.unravel_index(top_k_idx, (data.shape))
    return col

index_3m = get_top_attributions_for_interactions(eval_tensor_3m, 250)
eval_idx_3m = np.arange(x_test_tensor_3m.shape[0])
np.random.shuffle(eval_idx_3m)
eval_tensor_3m = x_test_tensor_3m[eval_idx_3m[:500]][:,index_3m]
eval_tensor_3m.requires_grad_(True)
eval_tensor_3m.shape

index_18m = get_top_attributions_for_interactions(eval_tensor_18m, 250)
eval_idx_18m = np.arange(x_test_tensor_18m.shape[0])
np.random.shuffle(eval_idx_18m)
eval_tensor_18m = x_test_tensor_18m[eval_idx_18m[:500]][:,index_18m]
eval_tensor_18m.requires_grad_(True)

index_24m = get_top_attributions_for_interactions(eval_tensor_24m, 250)
eval_idx_24m = np.arange(x_test_tensor_24m.shape[0])
np.random.shuffle(eval_idx_24m)
eval_tensor_24m = x_test_tensor_24m[eval_idx_24m[:500]][:,index_24m]
eval_tensor_24m.requires_grad_(True)

#baseline as subset of samples
baseline_idx_3m=np.arange(x_test_tensor_3m.shape[0])
np.random.shuffle(baseline_idx_3m)
baseline_tensor_3m = x_test_tensor_3m[baseline_idx_3m[:500]][:,index_3m]
baseline_tensor_3m.requires_grad_(True)
baseline_tensor_3m.shape

baseline_idx_18m=np.arange(x_test_tensor_18m.shape[0])
np.random.shuffle(baseline_idx_18m)
baseline_tensor_18m = x_test_tensor_18m[baseline_idx_18m[:500]][:,index_18m]
baseline_tensor_18m.requires_grad_(True)

baseline_idx_24m=np.arange(x_test_tensor_24m.shape[0])
np.random.shuffle(baseline_idx_24m)
baseline_tensor_24m = x_test_tensor_24m[baseline_idx_24m[:500]][:,index_24m]
baseline_tensor_24m.requires_grad_(True)
"""
"""
Path_3m = "/Users/michael/Python/Organage/ACE/Interaction_batches_3m"
os.makedirs(Path_3m)
batch_size = 2
dims = [1,2,3]
for j in dims:
    out_indices = torch.full((batch_size,),j,device=eval_tensor_3m.device)

    for i in range(0,eval_tensor_3m.shape[0],batch_size):
        batchfile = os.path.join(Path_3m, f"batch_{i}_dim_{j}.pt")
        if os.path.exists(batchfile):
            continue
        interaction = explainer.interactions(input_tensor=eval_tensor_3m[i:i+batch_size,], baseline=baseline_tensor_3m, 
                                       num_samples=20, use_expectation=True,output_indices=out_indices)
        torch.save(interaction.detach().cpu(), batchfile)        
        del interaction

shap_3D_3m_int = (sum(results_int_3m[i] for i in dims)/3).detach().cpu().numpy().copy()
eval_tensor_detatched_3m = np.ascontiguousarray(eval_tensor_3m.detach().cpu().numpy().copy())
shap_3D_3m_int.shape
eval_tensor_3m.shape
"""

#%%correlation matrices
#Note: this class calculates the corellation matrix and extracts the top k correlated genes and their
    #corellation value. Takes a 2-D shap matrix (numpy array) as input and a gene vector with the same gene sequence
    #as the original gene data frame.

class get_attribution_correlation:
    def __init__(self, k, data, names):
        self.corr_matrix = np.corrcoef(data)
        np.fill_diagonal(self.corr_matrix, 0)
        self.topk_idx = np.argsort(abs(self.corr_matrix).flatten())[::-1][:k*2]
        self.row,self.col = np.unravel_index(self.topk_idx, self.corr_matrix.shape)
        self.colgenes = names[self.col]
        self.rowgenes = names[self.row]
        values = self.corr_matrix[self.row,self.col]
        self.df_int = pd.DataFrame({"gene1": self.colgenes,
                               "gene2": self.rowgenes,
                               "value": values
            })
        self.correlations = self.df_int[0:k*2:2]

print("calulate corellation matrices")
#calcualte corellations among the top 500 gene attribution values and extract top 50 corellations:

attr_corr_top50_3m = get_attribution_correlation(50, shap_3D_3m, test_3m.var_names)
attr_corr_top50_3m.correlations.to_csv(snakemake.output.top_50_corr_3m)
#%%
attr_corr_top50_18m = get_attribution_correlation(50, shap_3D_18m, test_18m.var_names)
attr_corr_top50_18m.correlations.to_csv(snakemake.output.top_50_corr_18m)

attr_corr_top50_24m = get_attribution_correlation(50, shap_3D_24m, test_24m.var_names)
attr_corr_top50_24m.correlations.to_csv(snakemake.output.top_50_corr_24m)



# %%
