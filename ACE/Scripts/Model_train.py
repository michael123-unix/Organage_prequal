#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 19:11:00 2026

@author: michael
"""
#%% imports
from ace.model import ACE
from pytorch_lightning.utilities.seed import seed_everything
import torch
import matplotlib.pyplot as plt
import scanpy as sc
import anndata 
import numpy as np
import pandas as pd
import warnings
from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings('ignore')
import os
from pytorch_lightning.loggers import TensorBoardLogger
from pathlib import Path

print("Loading and preparing data...")

#%%
path = Path.cwd()
print(path)
#%%imort data

#train_data = anndata.read(path.parent.parent/"Data/Train_data.h5ad")
#test_data = anndata.read(path.parent.parent/"Data/Test_data.h5ad")
train_data = anndata.read(snakemake.input.train_data)
test_data = anndata.read(snakemake.input.test_data)

#background and foreground
background = ["cell_ontology_class", "sex", "tissue"]
salient = ["age"]

#%%Plotting

#Age hist
plt.hist(train_data.obs["age"], bins = 24)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("count")
plt.savefig(snakemake.output.age_dist)
plt.close()

#cell types as hist
plt.hist(train_data.obs["cell_ontology_class"], bins=train_data.obs["cell_ontology_class"].nunique())
plt.xticks(rotation=45, fontsize=3)                 
plt.tight_layout()
plt.close()

#reduce cell types even more for plotting
cell_filter = train_data.obs["cell_ontology_class"].value_counts()>1000
abundant_cells_index = cell_filter[cell_filter]
abundant_cells_index.index.nunique()
abundant_cells = abundant_cells_index.index.tolist()
type(abundant_cells)

print("visualizing data...")

#violin plots
violin,ax = plt.subplots(1,2, figsize=(10,5))
for i,j in zip(["n_genes", "n_counts"], [0,1]):
    sc.pl.violin(train_data,i, jitter=0.3, ax=ax[j], show=False)                
violin.savefig(snakemake.output.violin)
#violin.savefig(path.parent/"Train_Results/Plots/violin.png")
plt.close()

#PCA Plots
PCA_vars,ax = plt.subplots(2,2, figsize=(20,10))

ax_flat = ax.flatten()
for plot_feature, i in zip(["age", "sex", "tissue"],[2,1,0]):
    sc.pl.pca(test_data, color=plot_feature, ax=ax_flat[i], show=False, legend_loc='right margin',
        legend_fontsize='small')
       
for plot_feature in ["cell_ontology_class"]:    #extra since it has to be restricted to fewer cell types
    sc.pl.pca(test_data, color=plot_feature, groups=abundant_cells, palette="tab20", ax=ax_flat[3], show=False, legend_loc='right margin',
        legend_fontsize='small')
plt.tight_layout()
PCA_vars.savefig(snakemake.output.PCA_vars)
#PCA_vars.savefig(path.parent/"Train_Results/Plots/PCA_vars.png")
plt.close()

#UMAP Plots
umap_vars,ax = plt.subplots(2,2, figsize=(20,10))

ax_flat = ax.flatten()
for plot_feature, i in zip(["age", "sex", "tissue"],[2,1,0]):
    sc.pl.umap(test_data, color=plot_feature, ax=ax_flat[i], show=False, legend_loc='right margin',
        legend_fontsize='small')
    
for plot_feature in ["cell_ontology_class"]:    #extra since it has to be restricted to fewer cell types
    sc.pl.umap(test_data, color=plot_feature, groups=abundant_cells, palette="tab20", ax=ax_flat[3], show=False, legend_loc='right margin',
        legend_fontsize='small')

plt.tight_layout()   
umap_vars.savefig(snakemake.output.umap_vars)
#umap_vars.savefig(path.parent/"Train_Results/Plots/umap_vars.png")

#%%model setup and train
"""
Options for variable types:
    categorical phenotype_keys, countinuous_phenopype_keys
    categorical_background_keys, continuous_background_keys
    pheno_continuous_recon_penalty, pheno_categorical_recon_penalty
    background_continuous-recon_penalty, pheno_continuous_recon_penalty
        The penalties for reconstruction, for categorical variables these have to be orders of magnitude higer
"""

#Model initialisation
print("initializing model...")
ACE.setup_anndata(train_data, layer="raw_counts", continuous_phenotype_keys=salient,
categorical_background_keys=background,
batch_key=None,
)

#model initialisation with softplus activation function
model = ACE(
    train_data,
    n_salient_latent=3,
    n_background_latent=17,
    use_observed_lib_size=True,
    hsic_loss_penalty=1e+6,
    pheno_continuous_recon_penalty=50, 
    back_categorical_recon_penalty=[4000, 4000, 12000], 
    dropout_rate_encoder=0,
    dropout_rate_pheno=0.5,
    dropout_rate_back=0.5,
    var_activation=torch.nn.Softplus(beta=10, threshold=400)
)

print("starting model training...")
model.train(
    use_gpu=False,
    check_val_every_n_epoch=1,
    train_size=0.8,
    early_stopping=True,
    early_stopping_monitor="validation_loss",
    early_stopping_patience=45,
    max_epochs=500,
    plan_kwargs=dict(lr=0.0001),
)

#%%model export
model.save(snakemake.output.model, overwrite=True)
#model.save(path.parent/"Model/model")
print("model training finished successfully")

#%%post training analysis
# get latent representations for the test data for validation

print("visualizing latent variables")
#data registration
model.setup_anndata(test_data, source_registry=model.registry_)

latent_salient= anndata.AnnData(X=model.get_latent_representation(test_data, representation_kind = "salient"), 
                                obs=test_data.obs)

latent_background = anndata.AnnData(X=model.get_latent_representation(test_data, representation_kind= "background"), 
                                    obs=test_data.obs)

#calculate neighbours and umap for both
sc.pp.neighbors(latent_salient)
sc.tl.umap(latent_salient)
sc.pp.neighbors(latent_background)
sc.tl.umap(latent_background)

#plot salient umaps
umap_salient_embeddings,ax = plt.subplots(2,2, figsize=(20,10))

ax_flat = ax.flatten()
for plot_feature, i in zip(["age", "sex", "tissue"], [2,1,0]):
    sc.pl.umap(latent_salient, color=plot_feature, show=False, ax=ax_flat[i])
       
sc.pl.umap(latent_salient, color = "cell_ontology_class", groups=abundant_cells, palette="tab20", show=False, ax=ax_flat[3])
umap_salient_embeddings.savefig(snakemake.output.umap_salient_embeddings)

#plot background umaps
umap_background_embeddings,ax = plt.subplots(2,2, figsize=(20,10))

ax_flat = ax.flatten()
for plot_feature, i in zip(["age", "sex", "tissue"], [2,1,0]):
    sc.pl.umap(latent_background, color=plot_feature, show=False, ax=ax_flat[i])
       
sc.pl.umap(latent_background, color = "cell_ontology_class", groups=abundant_cells, palette="tab20", show=False, ax=ax_flat[3])
umap_background_embeddings.savefig(snakemake.output.umap_background_embeddings)

print("Model establishment finished")

