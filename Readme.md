# Description

The project within this directory aims at evaluating three non linear models for their capability to identify age related biomarkers by inferring age and explaining the models using the [SHAP](https://shap.readthedocs.io/en/latest/overviews.html) framework for explaining AI models.  
The evaluated models were the [lightGBM](https://lightgbm.readthedocs.io/en/stable/) gradient boosted desition tree, the [ACE](https://github.com/suinleelab/ACE) variational autoencoder developed by the Sun Lee Lab and a simple linear classifier.  
The pipeline further emits the top 50 feature attribution in csv format for each model, outputs some data summary statistics and model parameters and even gene- gene interactions.
The snakmake pipline includes hyperparameter optimisaiton for lightGBM and the linear model while for the VAE the default parameters were used, model training and identifying the most informative features as judged by each model. Finally three venn diagrams visualize the overlap of the top 50 features of each model for mice aged 3, 16 and 24 months. 
As training data a small, publicly accessible subset of the [tabula muris senis](https://tabula-muris-senis.sf.czbiohub.org/) single cell sequencing data set was used.  

# Basic usage
The pipeline is currently set up to analyse a dataset containing specifically three age groups, namely 3, 18 and 24 months and is thus not capable to generalize without modification.   

# Results

## ACE 
The ACE model generates embeddings for Age (3D) and for Background (17D) with a loss function optimized for disentangeling the relationships of features to age and other covariates such as batch, cell type etc. The following plots show UMAPS of the initial features and the subsequent age- embeddings for the variables age, sex, cell type and tissue type:  
  
UMAPS of native features
![UMAP variables](ACE/Train_Results/Plots/umap_vars.png) 
  
UMAPS of Age- embeddings 
![UMAP age embeddings](ACE/Train_Results/Plots/umap_salient_embeddings.png)

The following plots depict the 10 most important features for the age groups 3, 18 and 24 months and their interaction partners:
  
Mice aged 3 months:
![Top attributions 3m](ACE/Explain_Results/Plots/attr_3m.png) 
  
Mice aged 18 months:
![Top attributions 18m](ACE/Explain_Results/Plots/attr_18m.png)
  
Mice aged 24 months:
![Top attributions 24m](ACE/Explain_Results/Plots/attr_24m.png)

## lightGBM
The following plots show the top 10 feature attributions of the lightGBM model.

Mice aged 3 months:
![Top attributions 3m](LGB/Results/Analysis/Plots/Beeswarm_3m.png) 
  
Mice aged 18 months:
![Top attributions 18m](LGB/Results/Analysis/Plots/Beeswarm_18m.png)
  
Mice aged 24 months:
![Top attributions 24m](LGB/Results/Analysis/Plots/Beeswarm_24m.png)

## Overall comparison
The following venn diagrams depict the overlap in top 50 features used by each of the three models for each of the three age groups:

![Top attributions 3m](Comparison/Results/venn_models_3m.png) 
![Top attributions 18m](Comparison/Results/venn_models_18m.png)
![Top attributions 24m](Comparison/Results/venn_models_24m.png)
